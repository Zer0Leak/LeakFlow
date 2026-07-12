#include "leakflow/plot/pipeline_graph.hpp"

#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/log/logger.hpp"
#include "leakflow/plot/trace_view.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace leakflow::plot {

bool progress_animation_enabled(PipelineSessionState state) noexcept {
    return state == PipelineSessionState::Running;
}

namespace {

constexpr auto max_recent_events = std::size_t{80};

[[nodiscard]] std::string exception_message(const std::exception_ptr &failure, std::string_view fallback) {
    if (!failure) {
        return std::string(fallback);
    }

    try {
        std::rethrow_exception(failure);
    } catch (const std::exception &error) {
        return error.what();
    } catch (...) {
        return std::string(fallback);
    }
}

void observe_worker_event(PipelineSession &session, PipelineEventKind kind, std::string message) {
    const auto observer = session.observer();
    if (!observer) {
        return;
    }

    try {
        observer->observe(PipelineEvent{
            .kind = kind,
            .message = std::move(message),
        });
    } catch (...) {
    }
}

void report_graph_run_failure(PipelineSession &session, std::string_view message) {
    observe_worker_event(session, PipelineEventKind::Error, std::string(message));

    log::LogRecord record{
        .level = log::LogLevel::Error,
        .component = "pipeline",
        .message = "pipeline graph run failed",
        .fields =
            {
                {"error", std::string(message)},
            },
    };
    log::write(std::move(record));
}

[[nodiscard]] std::string_view pad_direction_name(PadDirection direction) {
    switch (direction) {
    case PadDirection::Input:
        return "input";
    case PadDirection::Output:
        return "output";
    }

    return "unknown";
}

[[nodiscard]] std::string_view pad_presence_name(PadPresence presence) {
    switch (presence) {
    case PadPresence::Required:
        return "required";
    case PadPresence::Optional:
        return "optional";
    case PadPresence::OnRequest:
        return "on-request";
    }

    return "unknown";
}

[[nodiscard]] std::string_view event_kind_name(PipelineEventKind kind) {
    switch (kind) {
    case PipelineEventKind::TopologySnapshot:
        return "topology";
    case PipelineEventKind::Started:
        return "started";
    case PipelineEventKind::Stopped:
        return "stopped";
    case PipelineEventKind::Error:
        return "error";
    case PipelineEventKind::ElementStarted:
        return "element-started";
    case PipelineEventKind::ElementStopped:
        return "element-stopped";
    case PipelineEventKind::BufferObserved:
        return "buffer";
    case PipelineEventKind::PropertyChanged:
        return "property";
    case PipelineEventKind::TelemetryChanged:
        return "telemetry";
    case PipelineEventKind::ProgressReported:
        return "progress";
    case PipelineEventKind::CommandAccepted:
        return "command-accepted";
    case PipelineEventKind::CommandRejected:
        return "command-rejected";
    case PipelineEventKind::CommandApplied:
        return "command-applied";
    }

    return "unknown";
}

[[nodiscard]] std::string element_label(const PipelineElementSnapshot &element) {
    return element.type_name + "@" + element.name;
}

struct KlassColors {
    ImU32 fill;
    ImU32 border;
    ImU32 accent;
};

[[nodiscard]] ImU32 color_u32(float r, float g, float b, float a = 1.0F) {
    return ImGui::GetColorU32(ImVec4(r, g, b, a));
}

[[nodiscard]] KlassColors klass_colors_from_rgb(
    std::array<float, 3> fill,
    std::array<float, 3> border,
    std::array<float, 3> accent)
{
    return KlassColors{
        .fill = color_u32(fill[0], fill[1], fill[2], 1.0F),
        .border = color_u32(border[0], border[1], border[2], 0.92F),
        .accent = color_u32(accent[0], accent[1], accent[2], 0.96F),
    };
}

[[nodiscard]] bool klass_has_prefix(std::string_view klass, std::string_view prefix) {
    return klass == prefix ||
           (klass.size() > prefix.size() && klass.starts_with(prefix) && klass[prefix.size()] == '/');
}

[[nodiscard]] KlassColors klass_colors(std::string_view klass) {
    // Match from specific roles to broad forwarding profiles. Keep this palette in
    // sync with docs/design/metadata_klass_taxonomy.md when new klass families land.
    const auto matches = [klass](std::string_view prefix) { return klass_has_prefix(klass, prefix); };

    if (matches("Source/Live")) {
        return klass_colors_from_rgb({0.05F, 0.23F, 0.32F}, {0.13F, 0.83F, 0.93F}, {0.38F, 0.93F, 1.00F});
    }
    if (matches("Source/App")) {
        return klass_colors_from_rgb({0.13F, 0.18F, 0.42F}, {0.51F, 0.55F, 0.96F}, {0.64F, 0.68F, 1.00F});
    }
    if (matches("Source/Test")) {
        return klass_colors_from_rgb({0.17F, 0.22F, 0.30F}, {0.58F, 0.67F, 0.78F}, {0.74F, 0.82F, 0.92F});
    }
    if (matches("Source/File")) {
        return klass_colors_from_rgb({0.08F, 0.20F, 0.37F}, {0.23F, 0.51F, 0.96F}, {0.38F, 0.65F, 1.00F});
    }
    if (matches("Source")) {
        return klass_colors_from_rgb({0.08F, 0.22F, 0.36F}, {0.28F, 0.62F, 0.97F}, {0.56F, 0.79F, 1.00F});
    }

    if (matches("PassThrough/Flow/Sync")) {
        return klass_colors_from_rgb({0.14F, 0.20F, 0.28F}, {0.38F, 0.65F, 0.98F}, {0.58F, 0.77F, 1.00F});
    }
    if (matches("PassThrough/Flow/Tee")) {
        return klass_colors_from_rgb({0.16F, 0.19F, 0.27F}, {0.50F, 0.55F, 0.96F}, {0.68F, 0.72F, 1.00F});
    }
    if (matches("PassThrough/Flow")) {
        return klass_colors_from_rgb({0.15F, 0.20F, 0.25F}, {0.58F, 0.64F, 0.72F}, {0.79F, 0.86F, 0.95F});
    }
    if (matches("PassThrough/Inspect")) {
        return klass_colors_from_rgb({0.16F, 0.19F, 0.23F}, {0.80F, 0.84F, 0.88F}, {0.94F, 0.97F, 1.00F});
    }
    if (matches("PassThrough")) {
        return klass_colors_from_rgb({0.14F, 0.20F, 0.24F}, {0.55F, 0.65F, 0.72F}, {0.75F, 0.86F, 0.94F});
    }

    if (matches("Convert/PlotAnnotation")) {
        return klass_colors_from_rgb({0.07F, 0.25F, 0.21F}, {0.37F, 0.92F, 0.83F}, {0.60F, 1.00F, 0.91F});
    }
    if (matches("Convert/Tensor")) {
        return klass_colors_from_rgb({0.07F, 0.24F, 0.23F}, {0.08F, 0.72F, 0.65F}, {0.28F, 0.90F, 0.82F});
    }
    if (matches("Convert/Signal")) {
        return klass_colors_from_rgb({0.08F, 0.26F, 0.18F}, {0.20F, 0.83F, 0.51F}, {0.47F, 1.00F, 0.70F});
    }
    if (matches("Convert/Predict")) {
        return klass_colors_from_rgb({0.11F, 0.27F, 0.17F}, {0.45F, 0.90F, 0.45F}, {0.70F, 1.00F, 0.70F});
    }
    if (matches("Convert")) {
        return klass_colors_from_rgb({0.07F, 0.24F, 0.22F}, {0.18F, 0.77F, 0.69F}, {0.45F, 0.96F, 0.87F});
    }

    if (matches("Analyze/SCA/Evaluation")) {
        return klass_colors_from_rgb({0.29F, 0.23F, 0.09F}, {0.99F, 0.90F, 0.54F}, {1.00F, 0.95F, 0.67F});
    }
    if (matches("Analyze/SCA/Attack")) {
        return klass_colors_from_rgb({0.29F, 0.15F, 0.06F}, {0.98F, 0.45F, 0.09F}, {1.00F, 0.62F, 0.30F});
    }
    if (matches("Analyze/SCA/PoI")) {
        return klass_colors_from_rgb({0.25F, 0.23F, 0.07F}, {0.98F, 0.80F, 0.08F}, {1.00F, 0.90F, 0.29F});
    }
    if (matches("Analyze/SCA/Score")) {
        return klass_colors_from_rgb({0.29F, 0.22F, 0.05F}, {0.92F, 0.70F, 0.03F}, {0.99F, 0.84F, 0.27F});
    }
    if (matches("Analyze/SCA/Hypothesis")) {
        return klass_colors_from_rgb({0.30F, 0.16F, 0.07F}, {0.92F, 0.34F, 0.05F}, {1.00F, 0.52F, 0.20F});
    }
    if (matches("Analyze/SCA/Leakage")) {
        return klass_colors_from_rgb({0.29F, 0.18F, 0.04F}, {0.85F, 0.47F, 0.02F}, {0.98F, 0.64F, 0.15F});
    }
    if (matches("Analyze/SCA/Label")) {
        return klass_colors_from_rgb({0.25F, 0.25F, 0.08F}, {0.84F, 0.84F, 0.23F}, {0.94F, 0.94F, 0.40F});
    }
    if (matches("Analyze/SCA/Model")) {
        return klass_colors_from_rgb({0.30F, 0.19F, 0.07F}, {0.96F, 0.58F, 0.20F}, {1.00F, 0.74F, 0.40F});
    }
    if (matches("Analyze/SCA/Oracle")) {
        return klass_colors_from_rgb({0.32F, 0.15F, 0.07F}, {0.96F, 0.39F, 0.20F}, {1.00F, 0.58F, 0.38F});
    }
    if (matches("Analyze/SCA/Solver")) {
        return klass_colors_from_rgb({0.31F, 0.21F, 0.08F}, {0.95F, 0.62F, 0.18F}, {1.00F, 0.78F, 0.42F});
    }
    if (matches("Analyze")) {
        return klass_colors_from_rgb({0.28F, 0.20F, 0.06F}, {0.92F, 0.60F, 0.16F}, {1.00F, 0.78F, 0.35F});
    }

    if (matches("Sink/Plot/AttackScoreboard")) {
        return klass_colors_from_rgb({0.23F, 0.16F, 0.41F}, {0.77F, 0.71F, 0.99F}, {0.88F, 0.84F, 1.00F});
    }
    if (matches("Sink/Plot/AttackScore")) {
        return klass_colors_from_rgb({0.20F, 0.13F, 0.37F}, {0.55F, 0.36F, 0.96F}, {0.72F, 0.56F, 1.00F});
    }
    if (matches("Sink/Plot")) {
        return klass_colors_from_rgb({0.18F, 0.13F, 0.34F}, {0.65F, 0.55F, 0.98F}, {0.78F, 0.69F, 1.00F});
    }
    if (matches("Sink/File")) {
        return klass_colors_from_rgb({0.16F, 0.12F, 0.30F}, {0.46F, 0.35F, 0.86F}, {0.64F, 0.55F, 0.96F});
    }
    if (matches("Sink/Test")) {
        return klass_colors_from_rgb({0.19F, 0.17F, 0.27F}, {0.58F, 0.54F, 0.74F}, {0.74F, 0.70F, 0.88F});
    }
    if (matches("Sink/Evaluation")) {
        return klass_colors_from_rgb({0.20F, 0.14F, 0.34F}, {0.79F, 0.70F, 0.99F}, {0.90F, 0.84F, 1.00F});
    }
    if (matches("Sink")) {
        return klass_colors_from_rgb({0.18F, 0.12F, 0.31F}, {0.62F, 0.50F, 0.93F}, {0.78F, 0.68F, 1.00F});
    }

    if (matches("Control/Fault")) {
        return klass_colors_from_rgb({0.34F, 0.10F, 0.15F}, {0.96F, 0.32F, 0.48F}, {1.00F, 0.52F, 0.64F});
    }
    if (matches("Control")) {
        return klass_colors_from_rgb({0.29F, 0.11F, 0.16F}, {0.89F, 0.32F, 0.48F}, {1.00F, 0.54F, 0.66F});
    }

    return klass_colors_from_rgb({0.20F, 0.23F, 0.27F}, {0.56F, 0.62F, 0.70F}, {0.75F, 0.81F, 0.88F});
}

[[nodiscard]] ImU32 link_color(bool has_buffer) {
    if (has_buffer) {
        return ImGui::GetColorU32(ImVec4(0.20F, 0.76F, 0.46F, 0.95F));
    }

    return ImGui::GetColorU32(ImVec4(0.54F, 0.58F, 0.64F, 0.70F));
}

void tooltip_heading(std::string_view text) {
    ImGui::TextColored(ImVec4(0.42F, 0.74F, 1.00F, 1.0F), "%s", std::string(text).c_str());
}

void tooltip_field(std::string_view key, std::string_view value) {
    ImGui::TextColored(ImVec4(0.72F, 0.78F, 0.86F, 1.0F), "%s", std::string(key).c_str());
    ImGui::SameLine();
    ImGui::TextUnformatted(std::string(value).c_str());
}

[[nodiscard]] std::string effect_summary(const PropertyEffect &effect) {
    auto summary = std::string(property_effect_kind_name(effect.kind));
    summary += " / ";
    summary += property_invalidation_scope_name(effect.scope);
    if (!effect.output_pads.empty()) {
        summary += " [";
        for (std::size_t index = 0; index < effect.output_pads.size(); ++index) {
            if (index != 0) {
                summary += ",";
            }
            summary += effect.output_pads[index];
        }
        summary += "]";
    }
    return summary;
}

void tooltip_caps(std::string_view title, const Caps &caps) {
    tooltip_heading(title);
    ImGui::Indent(12.0F);
    tooltip_field("type", caps.type());
    for (const auto &[key, value] : caps.params()) {
        tooltip_field(key, value);
    }
    ImGui::Unindent(12.0F);
}

void tooltip_pad_list(std::string_view title, const std::vector<PipelinePadSnapshot> &pads) {
    tooltip_heading(title);
    ImGui::Indent(12.0F);
    if (pads.empty()) {
        ImGui::TextUnformatted("none");
    }
    for (const auto &pad : pads) {
        const auto text = pad.name + " (" + std::string(pad_presence_name(pad.presence)) + ") " + pad.caps.to_string();
        ImGui::TextUnformatted(text.c_str());
    }
    ImGui::Unindent(12.0F);
}

// Forward declarations: the tooltip and the pinned panel render the same bodies,
// differing only by `interactive` (the tooltip cannot be clicked).
void draw_element_info_body(PipelineGraphRuntime &runtime, const PipelineElementSnapshot &element, bool interactive);
void draw_link_info_body(PipelineGraphRuntime &runtime, const PipelineLinkSnapshot &link,
                         const std::optional<PipelineBufferObservation> &latest, std::uint64_t observed_count,
                         bool interactive);
[[nodiscard]] std::map<std::uint32_t, std::string> build_slot_labels(const PipelineTopologySnapshot &topology);
[[nodiscard]] std::string decode_clock(const std::map<std::uint32_t, std::string> &labels,
                                       const std::vector<std::uint32_t> &clock);

// A collapsible section header whose open/closed state is shared (via the
// runtime) between the interactive pinned panel and the non-interactive hover
// tooltip. In a panel it is a real, clickable CollapsingHeader; in a tooltip it
// is a static disclosure line that simply mirrors the panel's choice — so
// collapsing a section in the panel also shrinks the tooltip.
bool draw_section_header(PipelineGraphRuntime &runtime, const char *label, bool interactive, bool default_open) {
    const bool open = runtime.section_open(label, default_open);
    if (interactive) {
        ImGui::SetNextItemOpen(open, ImGuiCond_Always);
        const bool now_open = ImGui::CollapsingHeader(label);
        if (now_open != open) {
            runtime.set_section_open(label, now_open);
        }
        return now_open;
    }
    ImGui::TextColored(ImVec4(0.42F, 0.74F, 1.00F, 1.0F), "%s %s", open ? "[-]" : "[+]", label);
    return open;
}

void draw_element_tooltip(PipelineGraphRuntime &runtime, const PipelineElementSnapshot &element) {
    ImGui::BeginTooltip();
    tooltip_heading(element_label(element));
    ImGui::Separator();
    draw_element_info_body(runtime, element, /*interactive=*/false);
    ImGui::Separator();
    ImGui::TextDisabled("click to pin; collapse sections in the panel");
    ImGui::EndTooltip();
}

void draw_link_tooltip(PipelineGraphRuntime &runtime, const PipelineLinkSnapshot &link,
                       const std::optional<PipelineBufferObservation> &latest, std::uint64_t observed_count) {
    ImGui::BeginTooltip();
    tooltip_heading(link.id);
    ImGui::Separator();
    draw_link_info_body(runtime, link, latest, observed_count, /*interactive=*/false);
    ImGui::Separator();
    ImGui::TextDisabled("click to pin; collapse sections in the panel");
    ImGui::EndTooltip();
}

// Element info with collapsible sections shared between the pinned panel
// (interactive) and the hover tooltip (mirrors the panel's collapse state).
void draw_element_info_body(PipelineGraphRuntime &runtime, const PipelineElementSnapshot &element, bool interactive) {
    tooltip_field("type", element.type_name);
    tooltip_field("name", element.name);
    tooltip_field("klass", element.klass.empty() ? "unspecified" : element.klass);
    if (element.provenance_slots == 1) {
        tooltip_field("clock slot", std::to_string(element.provenance_base));
    } else if (element.provenance_slots > 1) {
        tooltip_field("clock slots", std::to_string(element.provenance_base) + ".." +
                                          std::to_string(element.provenance_base + element.provenance_slots - 1));
    }

    if (draw_section_header(runtime, "Properties", interactive, /*default_open=*/true)) {
        ImGui::Indent(12.0F);
        if (element.properties.empty()) {
            ImGui::TextUnformatted("none");
        }
        for (const auto &property : element.properties) {
            const auto value = property.name + " = " + property.value + " (" + property.value_type + ")";
            ImGui::TextUnformatted(value.c_str());
            if (!property.writable) {
                ImGui::SameLine();
                ImGui::TextDisabled("[read-only]");
            }
            if (property.effect.kind != PropertyEffectKind::UiControl ||
                property.effect.scope != PropertyInvalidationScope::None || !property.effect.output_pads.empty()) {
                ImGui::Indent(12.0F);
                tooltip_field("effect", effect_summary(property.effect));
                ImGui::Unindent(12.0F);
            }
        }
        ImGui::Unindent(12.0F);
    }

    // Telemetry (size/monitoring) and Profile (duration/timing) are shown in
    // separate sections, distinguished by TelemetryKind. See docs/design/profiling.md.
    const auto render_telemetry_fields = [&element](TelemetryKind kind) {
        bool any = false;
        for (const auto &field : element.telemetry) {
            if (field.kind != kind) {
                continue;
            }
            any = true;
            auto value = field.name + " = " + field.value + " (" + field.value_type + ")";
            if (!field.unit.empty()) {
                value += " [" + field.unit + "]";
            }
            ImGui::TextUnformatted(value.c_str());
            if (!field.description.empty()) {
                ImGui::Indent(12.0F);
                tooltip_field("description", field.description);
                ImGui::Unindent(12.0F);
            }
        }
        return any;
    };

    if (draw_section_header(runtime, "Telemetry", interactive, /*default_open=*/true)) {
        ImGui::Indent(12.0F);
        if (!runtime.topology().runtime_telemetry_enabled) {
            ImGui::TextDisabled("disabled for this run");
            ImGui::Indent(12.0F);
            ImGui::TextDisabled("use --telemetry, or omit --no-telemetry in --graph");
            ImGui::Unindent(12.0F);
        } else if (!render_telemetry_fields(TelemetryKind::Size)) {
            ImGui::TextUnformatted("none");
        }
        ImGui::Unindent(12.0F);
    }

    if (draw_section_header(runtime, "Profile", interactive, /*default_open=*/true)) {
        ImGui::Indent(12.0F);
        if (!render_telemetry_fields(TelemetryKind::Duration)) {
            ImGui::TextDisabled("no timing");
            ImGui::Indent(12.0F);
            ImGui::TextDisabled("use --print-profile or --profile-file");
            ImGui::Unindent(12.0F);
        }
        ImGui::Unindent(12.0F);
    }

    if (draw_section_header(runtime, "Pads", interactive, /*default_open=*/false)) {
        tooltip_pad_list("input pads", element.input_pads);
        tooltip_pad_list("output pads", element.output_pads);
        if (!element.pad_templates.empty()) {
            tooltip_pad_list("pad templates", element.pad_templates);
        }
    }
}

// Link info with collapsible sections shared between panel and tooltip.
void draw_link_info_body(PipelineGraphRuntime &runtime, const PipelineLinkSnapshot &link,
                         const std::optional<PipelineBufferObservation> &latest, std::uint64_t observed_count,
                         bool interactive) {
    tooltip_field("source", link.source.element_name + "." + link.source.pad_name);
    tooltip_field("sink", link.sink.element_name + "." + link.sink.pad_name);
    tooltip_field("observed", std::to_string(observed_count));

    if (draw_section_header(runtime, "Declared caps", interactive, /*default_open=*/false)) {
        tooltip_caps("source", link.source_caps);
        tooltip_caps("sink", link.sink_caps);
    }

    if (!latest) {
        ImGui::TextUnformatted("latest buffer: not observed yet");
        return;
    }

    if (draw_section_header(runtime, "Latest buffer", interactive, /*default_open=*/true)) {
        ImGui::Indent(12.0F);
        tooltip_caps("caps", latest->buffer.caps);
        tooltip_field("sequence", std::to_string(latest->sequence));
        tooltip_field("generation", std::to_string(latest->generation));
        tooltip_field("clock", decode_clock(build_slot_labels(runtime.topology()), latest->provenance));
        ImGui::Unindent(12.0F);
    }

    if (draw_section_header(runtime, "Metadata", interactive, /*default_open=*/false)) {
        ImGui::Indent(12.0F);
        if (latest->buffer.metadata.empty()) {
            ImGui::TextUnformatted("none");
        }
        for (const auto &[key, value] : latest->buffer.metadata) {
            tooltip_field(key, value);
        }
        ImGui::Unindent(12.0F);
    }

    if (draw_section_header(runtime, "Payload", interactive, /*default_open=*/false)) {
        ImGui::Indent(12.0F);
        tooltip_field("type", latest->buffer.payload_type);
        tooltip_field("present", latest->buffer.has_payload ? "true" : "false");
        for (const auto &line : latest->buffer.payload_summary) {
            ImGui::TextUnformatted(line.c_str());
        }
        ImGui::Unindent(12.0F);
    }
}

// Map each allocated vector-clock slot to a human label: the element name for a
// single-slot element, or `element.pad` when the element claims one slot per
// output pad (the future per-pad case). Slot 0 is reserved and never mapped.
[[nodiscard]] std::map<std::uint32_t, std::string> build_slot_labels(const PipelineTopologySnapshot &topology) {
    std::map<std::uint32_t, std::string> labels;
    for (const auto &element : topology.elements) {
        if (element.provenance_slots <= 0) {
            continue;
        }
        const bool per_pad = static_cast<std::size_t>(element.provenance_slots) == element.output_pads.size();
        for (int s = 0; s < element.provenance_slots; ++s) {
            const auto index = element.provenance_base + static_cast<std::uint32_t>(s);
            if (element.provenance_slots == 1) {
                labels[index] = element.name;
            } else if (per_pad) {
                labels[index] = element.name + "." + element.output_pads[static_cast<std::size_t>(s)].name;
            } else {
                labels[index] = element.name + "[" + std::to_string(s) + "]";
            }
        }
    }
    return labels;
}

// Decode a buffer vector clock into "label=count" for every non-zero slot (slot 0
// reserved). Unmapped slots fall back to "#index".
[[nodiscard]] std::string decode_clock(const std::map<std::uint32_t, std::string> &labels,
                                       const std::vector<std::uint32_t> &clock) {
    std::string out;
    for (std::uint32_t index = 1; index < clock.size(); ++index) {
        if (clock[index] == 0) {
            continue;
        }
        if (!out.empty()) {
            out += "   ";
        }
        const auto found = labels.find(index);
        out += (found != labels.end() ? found->second : "#" + std::to_string(index)) + "=" +
               std::to_string(clock[index]);
    }
    return out.empty() ? std::string("(empty)") : out;
}

// A panel listing the current production count per vector-clock slot, labelled by
// the owning element (or element.pad). Slot 0 is reserved and skipped.
void draw_provenance_panel(const PipelineGraphRuntime &runtime) {
    if (!ImGui::CollapsingHeader("Vector Clock (provenance counts)")) {
        return;
    }

    const auto labels = build_slot_labels(runtime.topology());
    const auto &counts = runtime.max_provenance();
    if (counts.size() <= 1) {
        ImGui::TextUnformatted("no buffers observed yet");
        return;
    }

    if (ImGui::BeginTable("provenance_counts", 3,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 48.0F);
        ImGui::TableSetupColumn("Element");
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 64.0F);
        ImGui::TableHeadersRow();
        for (std::uint32_t index = 1; index < counts.size(); ++index) {
            const auto label_it = labels.find(index);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%u", index);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(label_it != labels.end() ? label_it->second.c_str() : "(unassigned)");
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%u", counts[index]);
        }
        ImGui::EndTable();
    }
}

// Per-buffer vector-clock debug panel: the latest buffer on every link, with its
// clock decoded into element (or element.pad) = count for each non-zero slot.
void draw_buffer_clocks_panel(const PipelineGraphRuntime &runtime) {
    if (!ImGui::CollapsingHeader("Buffer Clocks (per link)")) {
        return;
    }

    if (runtime.latest_buffers().empty()) {
        ImGui::TextUnformatted("no buffers observed yet");
        return;
    }

    const auto labels = build_slot_labels(runtime.topology());
    if (ImGui::BeginTable("buffer_clocks", 2,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Link");
        ImGui::TableSetupColumn("Vector clock (element = count)");
        ImGui::TableHeadersRow();
        for (const auto &[link_id, observation] : runtime.latest_buffers()) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(link_id.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(decode_clock(labels, observation.provenance).c_str());
        }
        ImGui::EndTable();
    }
}

// Item 1: draw the pinned (collapsible, mouse-interactable) info windows for the
// elements/links the user clicked, with a close button that unpins.
void draw_pinned_info_windows(PipelineGraphRuntime &runtime) {
    // Spawn pinned windows at the right edge of the viewport, cascading, so they
    // are clearly visible (and distinct from the cursor-following tooltip) and the
    // user can drag them afterwards.
    const auto *viewport = ImGui::GetMainViewport();
    float cascade = 0.0F;
    const auto place_next = [&]() {
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 360.0F,
                                       viewport->WorkPos.y + 30.0F + cascade),
                                ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340.0F, 0.0F), ImGuiCond_FirstUseEver);
        cascade += 36.0F;
    };

    const std::vector<std::string> elements(runtime.pinned_elements().begin(), runtime.pinned_elements().end());
    for (const auto &name : elements) {
        const PipelineElementSnapshot *snapshot = nullptr;
        for (const auto &element : runtime.topology().elements) {
            if (element.name == name) {
                snapshot = &element;
                break;
            }
        }
        if (snapshot == nullptr) {
            runtime.unpin_element(name);
            continue;
        }
        bool open = true;
        place_next();
        if (ImGui::Begin(("Element: " + name).c_str(), &open)) {
            draw_element_info_body(runtime, *snapshot, /*interactive=*/true);
        }
        ImGui::End();
        if (!open) {
            runtime.unpin_element(name);
        }
    }

    const std::vector<std::string> links(runtime.pinned_links().begin(), runtime.pinned_links().end());
    for (const auto &link_id : links) {
        const PipelineLinkSnapshot *snapshot = nullptr;
        for (const auto &link : runtime.topology().links) {
            if (link.id == link_id) {
                snapshot = &link;
                break;
            }
        }
        if (snapshot == nullptr) {
            runtime.unpin_link(link_id);
            continue;
        }
        std::optional<PipelineBufferObservation> latest;
        if (const auto found = runtime.latest_buffers().find(link_id); found != runtime.latest_buffers().end()) {
            latest = found->second;
        }
        bool open = true;
        place_next();
        if (ImGui::Begin(("Link: " + link_id).c_str(), &open)) {
            draw_link_info_body(runtime, *snapshot, latest, runtime.observed_count(link_id), /*interactive=*/true);
        }
        ImGui::End();
        if (!open) {
            runtime.unpin_link(link_id);
        }
    }
}

[[nodiscard]] float distance_to_segment(ImVec2 point, ImVec2 begin, ImVec2 end) {
    const auto vx = end.x - begin.x;
    const auto vy = end.y - begin.y;
    const auto wx = point.x - begin.x;
    const auto wy = point.y - begin.y;
    const auto length_squared = vx * vx + vy * vy;
    if (length_squared <= 0.0001F) {
        const auto dx = point.x - begin.x;
        const auto dy = point.y - begin.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    const auto t = std::clamp((wx * vx + wy * vy) / length_squared, 0.0F, 1.0F);
    const auto projection = ImVec2(begin.x + t * vx, begin.y + t * vy);
    const auto dx = point.x - projection.x;
    const auto dy = point.y - projection.y;
    return std::sqrt(dx * dx + dy * dy);
}

[[nodiscard]] std::optional<std::size_t> element_index_by_name(const PipelineTopologySnapshot &topology,
                                                               std::string_view name) {
    for (std::size_t index = 0; index < topology.elements.size(); ++index) {
        if (topology.elements[index].name == name) {
            return index;
        }
    }

    return std::nullopt;
}

[[nodiscard]] std::optional<PipelineBufferObservation> latest_buffer_for(const PipelineGraphRuntime &runtime,
                                                                         std::string_view link_id) {
    const auto found = runtime.latest_buffers().find(std::string(link_id));
    if (found == runtime.latest_buffers().end()) {
        return std::nullopt;
    }

    return found->second;
}

void draw_recent_events(const PipelineGraphRuntime &runtime) {
    if (runtime.recent_events().empty()) {
        return;
    }

    if (!ImGui::CollapsingHeader("Recent Events")) {
        return;
    }

    const auto &events = runtime.recent_events();
    const auto begin = events.size() > 12 ? events.size() - 12 : 0U;
    for (std::size_t index = begin; index < events.size(); ++index) {
        const auto &event = events[index];
        auto line = "#" + std::to_string(event.sequence) + " ";
        line += event_kind_name(event.kind);
        if (!event.element_name.empty()) {
            line += " ";
            line += event.element_name;
        }
        if (event.property_change) {
            line += " ";
            line += event.property_change->property_name;
            line += "=";
            line += event.property_change->new_value;
            line += " ";
            line += effect_summary(event.property_change->effect);
        }
        if (event.telemetry_change) {
            line += " ";
            line += event.telemetry_change->telemetry_name;
            line += "=";
            line += event.telemetry_change->new_value;
            if (!event.telemetry_change->unit.empty()) {
                line += " ";
                line += event.telemetry_change->unit;
            }
        }
        if (!event.message.empty()) {
            line += " ";
            line += event.message;
        }
        ImGui::TextUnformatted(line.c_str());
    }
}

[[nodiscard]] std::string trim_to_string(std::string_view text) {
    const auto begin = text.find_first_not_of(" \t\n\r");
    if (begin == std::string_view::npos) {
        return {};
    }

    const auto end = text.find_last_not_of(" \t\n\r");
    return std::string(text.substr(begin, end - begin + 1));
}

[[nodiscard]] std::string control_state_key(std::string_view element_name, std::string_view property_name) {
    return std::string(element_name) + "." + std::string(property_name);
}

[[nodiscard]] const IntRangeConstraint *int_range_constraint(const PropertySpec &spec) {
    return std::get_if<IntRangeConstraint>(&spec.constraint);
}

[[nodiscard]] const DoubleRangeConstraint *double_range_constraint(const PropertySpec &spec) {
    return std::get_if<DoubleRangeConstraint>(&spec.constraint);
}

[[nodiscard]] const StringEnumConstraint *string_enum_constraint(const PropertySpec &spec) {
    return std::get_if<StringEnumConstraint>(&spec.constraint);
}

[[nodiscard]] std::vector<std::string> split_list_text(std::string_view text) {
    auto trimmed = trim_to_string(text);
    if (trimmed.size() >= 2 && trimmed.front() == '[' && trimmed.back() == ']') {
        trimmed = trimmed.substr(1, trimmed.size() - 2);
    }

    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= trimmed.size()) {
        const auto comma = trimmed.find(',', begin);
        const auto end = comma == std::string::npos ? trimmed.size() : comma;
        auto part = trim_to_string(std::string_view(trimmed).substr(begin, end - begin));
        if (!part.empty() && part.front() == '"' && part.back() == '"' && part.size() >= 2) {
            part = part.substr(1, part.size() - 2);
        }
        if (!part.empty()) {
            parts.push_back(std::move(part));
        }
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }

    return parts;
}

[[nodiscard]] std::int64_t parse_integer(std::string_view text) {
    const auto trimmed = trim_to_string(text);
    if (trimmed.empty()) {
        throw std::invalid_argument("integer value cannot be empty");
    }

    std::size_t consumed = 0;
    const auto value = std::stoll(trimmed, &consumed);
    if (consumed != trimmed.size()) {
        throw std::invalid_argument("integer value contains trailing text");
    }
    return value;
}

[[nodiscard]] double parse_double(std::string_view text) {
    const auto trimmed = trim_to_string(text);
    if (trimmed.empty()) {
        throw std::invalid_argument("double value cannot be empty");
    }

    std::size_t consumed = 0;
    const auto value = std::stod(trimmed, &consumed);
    if (consumed != trimmed.size()) {
        throw std::invalid_argument("double value contains trailing text");
    }
    return value;
}

[[nodiscard]] IntList parse_integer_list(std::string_view text) {
    IntList values;
    for (const auto &part : split_list_text(text)) {
        values.push_back(parse_integer(part));
    }
    return values;
}

[[nodiscard]] DoubleList parse_double_list(std::string_view text) {
    DoubleList values;
    for (const auto &part : split_list_text(text)) {
        values.push_back(parse_double(part));
    }
    return values;
}

[[nodiscard]] StringList parse_string_list(std::string_view text) {
    StringList values;
    for (auto part : split_list_text(text)) {
        values.push_back(std::move(part));
    }
    return values;
}

void draw_help_marker(const PropertySpec &spec) {
    if (spec.description.empty() && spec.unit.empty() && spec.value_hint.empty() &&
        std::holds_alternative<std::monostate>(spec.constraint)) {
        return;
    }

    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (!ImGui::IsItemHovered()) {
        return;
    }

    ImGui::BeginTooltip();
    if (!spec.description.empty()) {
        ImGui::TextUnformatted(spec.description.c_str());
    }
    tooltip_field("type", property_value_type_name(spec.default_value));
    tooltip_field("access", spec.writable ? "read/write" : "read-only");
    if (!spec.unit.empty()) {
        tooltip_field("unit", spec.unit);
    }
    if (!spec.value_hint.empty()) {
        tooltip_field("hint", spec.value_hint);
    }
    tooltip_field("effect", effect_summary(spec.effect));
    if (const auto *constraint = int_range_constraint(spec)) {
        tooltip_field("range", std::to_string(constraint->min) + ".." + std::to_string(constraint->max));
    } else if (const auto *constraint = double_range_constraint(spec)) {
        tooltip_field("range", std::to_string(constraint->min) + ".." + std::to_string(constraint->max));
    } else if (const auto *constraint = string_enum_constraint(spec)) {
        std::string values;
        for (const auto &value : constraint->allowed_values) {
            if (!values.empty()) {
                values += ", ";
            }
            values += value;
        }
        tooltip_field("values", values);
    }
    ImGui::EndTooltip();
}

void sync_text_state(PipelineControlRuntime &runtime, std::string_view key, std::string_view observed_value) {
    auto &state = runtime.text_edit_state(key);
    if (!state.initialized || (!state.dirty && state.observed_value != observed_value)) {
        state.text = std::string(observed_value);
        state.observed_value = std::string(observed_value);
        state.dirty = false;
        state.initialized = true;
    }
}

void commit_text_property(PipelineControlRuntime &runtime, const Element &element, const PropertySpec &spec,
                          std::string_view state_key, PropertyValue value) {
    auto &state = runtime.text_edit_state(state_key);
    if (runtime.set_property(element.name(), spec.name, std::move(value))) {
        state.observed_value = property_value_to_string(element.property(spec.name));
        state.text = state.observed_value;
        state.dirty = false;
    }
}

void draw_string_editor(PipelineControlRuntime &runtime, const Element &element, const PropertySpec &spec,
                        const std::string &value) {
    const auto key = control_state_key(element.name(), spec.name);
    sync_text_state(runtime, key, value);
    auto &state = runtime.text_edit_state(key);
    if (ImGui::InputText("##value", &state.text)) {
        state.dirty = true;
    }
    if (ImGui::IsItemDeactivated() && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        // Escape discards the in-progress edit and restores the live value.
        state.text = state.observed_value;
        state.dirty = false;
    } else if (state.dirty && ImGui::IsItemDeactivatedAfterEdit()) {
        commit_text_property(runtime, element, spec, key, std::string(state.text));
    }
}

void draw_list_editor(PipelineControlRuntime &runtime, const Element &element, const PropertySpec &spec,
                      const PropertyValue &value) {
    const auto key = control_state_key(element.name(), spec.name);
    sync_text_state(runtime, key, property_value_to_string(value));
    auto &state = runtime.text_edit_state(key);
    if (ImGui::InputText("##value", &state.text)) {
        state.dirty = true;
    }
    if (ImGui::IsItemDeactivated() && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        // Escape discards the in-progress edit and restores the live value.
        state.text = state.observed_value;
        state.dirty = false;
        return;
    }
    if (!state.dirty || !ImGui::IsItemDeactivatedAfterEdit()) {
        return;
    }

    try {
        if (std::holds_alternative<IntList>(value)) {
            commit_text_property(runtime, element, spec, key, parse_integer_list(state.text));
        } else if (std::holds_alternative<DoubleList>(value)) {
            commit_text_property(runtime, element, spec, key, parse_double_list(state.text));
        } else if (std::holds_alternative<StringList>(value)) {
            commit_text_property(runtime, element, spec, key, parse_string_list(state.text));
        }
    } catch (const std::exception &error) {
        runtime.set_last_error(error.what());
    }
}

void draw_property_editor(PipelineControlRuntime &runtime, const Element &element, const PropertySpec &spec,
                          const PropertyValue &value) {
    // Null (unset) optional property: show it and offer to give it a value.
    if (std::holds_alternative<std::monostate>(value)) {
        ImGui::TextDisabled("null");
        if (spec.optional) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Set")) {
                (void)runtime.set_property(element.name(), spec.name, spec.default_value);
            }
        }
        return;
    }

    // Non-null optional property: offer a clear-to-null button before the editor.
    if (spec.optional) {
        if (ImGui::SmallButton("x")) {
            (void)runtime.set_property(element.name(), spec.name, PropertyValue{std::monostate{}});
            return;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("set to null");
        }
        ImGui::SameLine();
    }

    if (auto color_value = std::get_if<Color>(&value)) {
        const auto key = control_state_key(element.name(), spec.name);
        auto &state = runtime.color_edit_state(key);
        // While not actively editing, mirror the live value. While editing, keep
        // the local buffer so the async command-applied value cannot snap the
        // picker back mid-pick.
        if (!state.dirty) {
            state.rgba = {color_value->r, color_value->g, color_value->b, color_value->a};
        }

        // Compact swatch; click it to open the full RGBA/HSV/hex picker popup.
        const auto flags = ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar |
                           ImGuiColorEditFlags_AlphaPreviewHalf;
        const bool changed = ImGui::ColorEdit4("##value", state.rgba.data(), flags);
        if (changed || ImGui::IsItemDeactivatedAfterEdit()) {
            state.dirty = true;
            (void)runtime.set_property(element.name(), spec.name,
                                       Color{state.rgba[0], state.rgba[1], state.rgba[2], state.rgba[3]});
        }
        // Once the applied value catches up and the widget is idle, resume mirroring.
        if (state.dirty && !ImGui::IsItemActive()) {
            const Color edited{state.rgba[0], state.rgba[1], state.rgba[2], state.rgba[3]};
            if (edited == *color_value) {
                state.dirty = false;
            }
        }
        return;
    }

    if (auto bool_value = std::get_if<bool>(&value)) {
        auto edited = *bool_value;
        if (ImGui::Checkbox("##value", &edited)) {
            (void)runtime.set_property(element.name(), spec.name, edited);
        }
        return;
    }

    // A typed numeric value commits only on finish (Enter / focus-out), never on each
    // keystroke (which would apply half-typed values like 0 while typing "0.5"). The
    // built-in InputScalar +/- steps are unreliable to detect, so we render our own
    // -/+ buttons that commit immediately.
    if (auto integer_value = std::get_if<std::int64_t>(&value)) {
        auto edited = *integer_value;
        ImGui::SetNextItemWidth(140.0F);
        ImGui::InputScalar("##value", ImGuiDataType_S64, &edited, nullptr); // no built-in steps
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            (void)runtime.set_property(element.name(), spec.name, edited);
        }
        ImGui::SameLine();
        if (ImGui::Button("-")) {
            (void)runtime.set_property(element.name(), spec.name, *integer_value - 1);
        }
        ImGui::SameLine();
        if (ImGui::Button("+")) {
            (void)runtime.set_property(element.name(), spec.name, *integer_value + 1);
        }
        return;
    }

    if (auto double_value = std::get_if<double>(&value)) {
        auto edited = *double_value;
        ImGui::InputDouble("##value", &edited, 0.0, 0.0, "%.6g"); // step 0: no built-in buttons
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            (void)runtime.set_property(element.name(), spec.name, edited);
        }
        return;
    }

    if (auto string_value = std::get_if<std::string>(&value)) {
        if (const auto *constraint = string_enum_constraint(spec)) {
            auto selected = *string_value;
            if (ImGui::BeginCombo("##value", selected.c_str())) {
                for (const auto &allowed_value : constraint->allowed_values) {
                    const auto is_selected = allowed_value == selected;
                    if (ImGui::Selectable(allowed_value.c_str(), is_selected)) {
                        (void)runtime.set_property(element.name(), spec.name, std::string(allowed_value));
                        selected = allowed_value;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        } else {
            draw_string_editor(runtime, element, spec, *string_value);
        }
        return;
    }

    if (auto interval = std::get_if<IntInterval>(&value)) {
        auto edited = *interval;
        bool commit = false;
        ImGui::InputScalar("begin", ImGuiDataType_S64, &edited.begin, nullptr);
        commit = ImGui::IsItemDeactivatedAfterEdit() || commit;
        ImGui::SameLine();
        ImGui::InputScalar("end", ImGuiDataType_S64, &edited.end, nullptr);
        commit = ImGui::IsItemDeactivatedAfterEdit() || commit;
        if (commit) {
            (void)runtime.set_property(element.name(), spec.name, edited);
        }
        return;
    }

    if (auto interval = std::get_if<DoubleInterval>(&value)) {
        auto edited = *interval;
        bool commit = false;
        ImGui::InputDouble("begin", &edited.begin, 0.0, 0.0, "%.6g");
        commit = ImGui::IsItemDeactivatedAfterEdit() || commit;
        ImGui::SameLine();
        ImGui::InputDouble("end", &edited.end, 0.0, 0.0, "%.6g");
        commit = ImGui::IsItemDeactivatedAfterEdit() || commit;
        if (commit) {
            (void)runtime.set_property(element.name(), spec.name, edited);
        }
        return;
    }

    draw_list_editor(runtime, element, spec, value);
}

void draw_element_controls(PipelineControlRuntime &runtime, const std::shared_ptr<Element> &element) {
    if (!element) {
        ImGui::TextUnformatted("element no longer exists");
        return;
    }

    ImGui::TextColored(ImVec4(0.42F, 0.74F, 1.00F, 1.0F), "%s@%s", element->element_type().c_str(),
                       element->name().c_str());
    if (!element->element_kclass().empty()) {
        ImGui::TextUnformatted(element->element_kclass().c_str());
    }
    if (!runtime.edits_enabled() && runtime.session() == nullptr) {
        ImGui::Separator();
        ImGui::TextWrapped("Controls are paused while the pipeline worker is running.");
        return;
    }

    ImGui::Separator();
    for (const auto &spec : element->property_specs()) {
        const auto found = element->properties().find(spec.name);
        const auto &value = found == element->properties().end() ? spec.default_value : found->second;
        const auto read_only = !spec.writable || (spec.name == "name" && element->name_locked());

        ImGui::PushID(spec.name.c_str());
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(spec.name.c_str());
        draw_help_marker(spec);
        ImGui::SameLine(210.0F);
        ImGui::SetNextItemWidth(260.0F);
        if (read_only) {
            ImGui::TextUnformatted(property_value_to_string(value).c_str());
        } else {
            draw_property_editor(runtime, *element, spec, value);
        }
        ImGui::PopID();
    }
}

void draw_open_element_control_windows(PipelineControlRuntime &runtime) {
    const auto open_names = runtime.open_element_names();
    if (open_names.empty()) {
        // No control windows open: never contend for the worker's element mutex.
        // The graph draws entirely from snapshots, so the main thread must not
        // block here while the worker holds the lock across a (possibly paced)
        // live sweep -- otherwise the whole UI freezes between buffers.
        return;
    }

    // Serialize live element reads against the worker thread applying commands /
    // pumping sweeps (Phase 25). Use a try-lock: if the worker is mid-sweep we
    // defer this frame's control reads rather than freezing the UI. The graph
    // itself never needs this lock.
    std::unique_lock<std::mutex> element_lock;
    if (runtime.element_mutex() != nullptr) {
        element_lock = std::unique_lock<std::mutex>(*runtime.element_mutex(), std::try_to_lock);
    }
    const bool have_elements = runtime.element_mutex() == nullptr || element_lock.owns_lock();

    for (const auto &element_name : open_names) {
        auto open = true;
        const auto title = std::string("Controls: ") + element_name;
        ImGui::SetNextWindowSize(ImVec2(560.0F, 360.0F), ImGuiCond_FirstUseEver);
        if (runtime.take_focus_request(element_name)) {
            ImGui::SetNextWindowFocus();
        }
        if (ImGui::Begin(title.c_str(), &open)) {
            if (have_elements) {
                draw_element_controls(runtime, runtime.element(element_name));
            } else {
                ImGui::TextDisabled("syncing with running pipeline...");
            }
        }
        ImGui::End();
        if (!open) {
            runtime.close(element_name);
        }
    }
}

void draw_gear_icon(ImDrawList *draw_list, ImVec2 center, ImU32 color) {
    static constexpr auto pi = 3.14159265358979323846F;
    static constexpr auto outer = 7.0F;
    static constexpr auto inner = 3.0F;
    for (int tooth = 0; tooth < 8; ++tooth) {
        const auto angle = static_cast<float>(tooth) * pi * 0.25F;
        const auto begin = ImVec2(center.x + std::cos(angle) * (outer - 1.5F),
                                  center.y + std::sin(angle) * (outer - 1.5F));
        const auto end = ImVec2(center.x + std::cos(angle) * (outer + 1.0F),
                                center.y + std::sin(angle) * (outer + 1.0F));
        draw_list->AddLine(begin, end, color, 1.2F);
    }
    draw_list->AddCircle(center, outer - 1.0F, color, 16, 1.5F);
    draw_list->AddCircle(center, inner, color, 12, 1.4F);
}

void apply_property_change_to_topology(PipelineTopologySnapshot &topology,
                                       const PipelinePropertyChangeObservation &change) {
    for (auto &element : topology.elements) {
        if (element.name != change.element.element_name) {
            continue;
        }

        for (auto &property : element.properties) {
            if (property.name == change.property_name) {
                property.value_type = change.value_type;
                property.value = change.new_value;
                property.effect = change.effect;
                return;
            }
        }

        element.properties.push_back(PipelinePropertySnapshot{
            .name = change.property_name,
            .value_type = change.value_type,
            .value = change.new_value,
            .effect = change.effect,
            .writable = true,
        });
        return;
    }
}

void apply_telemetry_change_to_topology(PipelineTopologySnapshot &topology,
                                        const PipelineTelemetryChangeObservation &change) {
    if (!topology.runtime_telemetry_enabled) {
        return;
    }

    for (auto &element : topology.elements) {
        if (element.name != change.element.element_name) {
            continue;
        }

        for (auto &field : element.telemetry) {
            if (field.name == change.telemetry_name) {
                field.value_type = change.value_type;
                field.value = change.new_value;
                field.unit = change.unit;
                field.description = change.description;
                field.value_hint = change.value_hint;
                field.kind = change.kind;
                return;
            }
        }

        element.telemetry.push_back(PipelineTelemetrySnapshot{
            .name = change.telemetry_name,
            .value_type = change.value_type,
            .value = change.new_value,
            .unit = change.unit,
            .description = change.description,
            .value_hint = change.value_hint,
            .kind = change.kind,
        });
        return;
    }
}

} // namespace

void PipelineGraphRuntime::observe(const PipelineEvent &event) {
    const auto lock = std::scoped_lock(pending_mutex_);
    pending_events_.push_back(event);
}

void PipelineGraphRuntime::drain_events() {
    std::vector<PipelineEvent> events;
    {
        const auto lock = std::scoped_lock(pending_mutex_);
        events.swap(pending_events_);
    }

    for (const auto &event : events) {
        apply_event(event);
    }
}

void PipelineGraphRuntime::clear() {
    {
        const auto lock = std::scoped_lock(pending_mutex_);
        pending_events_.clear();
    }

    topology_ = {};
    has_topology_ = false;
    running_ = false;
    stopped_ = false;
    latest_buffers_.clear();
    observed_counts_.clear();
    link_generations_.clear();
    max_provenance_.clear();
    pinned_elements_.clear();
    pinned_links_.clear();
    last_error_.reset();
    recent_events_.clear();
    element_progress_.clear();
}

bool PipelineGraphRuntime::has_topology() const { return has_topology_; }

bool PipelineGraphRuntime::running() const { return running_; }

bool PipelineGraphRuntime::stopped() const { return stopped_; }

const PipelineTopologySnapshot &PipelineGraphRuntime::topology() const { return topology_; }

const std::map<std::string, PipelineBufferObservation> &PipelineGraphRuntime::latest_buffers() const {
    return latest_buffers_;
}

std::uint64_t PipelineGraphRuntime::observed_count(std::string_view link_id) const {
    const auto found = observed_counts_.find(std::string(link_id));
    if (found == observed_counts_.end()) {
        return 0;
    }

    return found->second;
}

const std::optional<std::string> &PipelineGraphRuntime::last_error() const { return last_error_; }

const std::vector<PipelineEvent> &PipelineGraphRuntime::recent_events() const { return recent_events_; }

const std::map<std::string, PipelineGraphRuntime::ElementProgressState> &
PipelineGraphRuntime::element_progress() const {
    return element_progress_;
}

const std::vector<std::uint32_t> &PipelineGraphRuntime::max_provenance() const { return max_provenance_; }

void PipelineGraphRuntime::toggle_pinned_element(std::string_view name) {
    const auto key = std::string(name);
    if (!pinned_elements_.erase(key)) {
        pinned_elements_.insert(key);
    }
}

bool PipelineGraphRuntime::is_element_pinned(std::string_view name) const {
    return pinned_elements_.contains(std::string(name));
}

const std::set<std::string> &PipelineGraphRuntime::pinned_elements() const { return pinned_elements_; }

void PipelineGraphRuntime::unpin_element(std::string_view name) { pinned_elements_.erase(std::string(name)); }

void PipelineGraphRuntime::toggle_pinned_link(std::string_view link_id) {
    const auto key = std::string(link_id);
    if (!pinned_links_.erase(key)) {
        pinned_links_.insert(key);
    }
}

bool PipelineGraphRuntime::is_link_pinned(std::string_view link_id) const {
    return pinned_links_.contains(std::string(link_id));
}

const std::set<std::string> &PipelineGraphRuntime::pinned_links() const { return pinned_links_; }

void PipelineGraphRuntime::unpin_link(std::string_view link_id) { pinned_links_.erase(std::string(link_id)); }

bool PipelineGraphRuntime::section_open(std::string_view label, bool default_open) const {
    const auto found = section_open_.find(std::string(label));
    return found == section_open_.end() ? default_open : found->second;
}

void PipelineGraphRuntime::set_section_open(std::string_view label, bool open) {
    section_open_[std::string(label)] = open;
}

void PipelineGraphRuntime::apply_event(const PipelineEvent &event) {
    recent_events_.push_back(event);
    if (recent_events_.size() > max_recent_events) {
        recent_events_.erase(recent_events_.begin(), recent_events_.begin() + 1);
    }

    switch (event.kind) {
    case PipelineEventKind::TopologySnapshot:
        if (event.topology) {
            topology_ = *event.topology;
            has_topology_ = true;
        }
        break;
    case PipelineEventKind::Started:
        running_ = true;
        stopped_ = false;
        last_error_.reset();
        // A fresh run restarts the vector clock at 0, so drop the previous run's
        // accumulated clock display and per-link counts (they would otherwise pin the
        // panel to the old maximum, e.g. 50, instead of the new run's counts).
        max_provenance_.clear();
        observed_counts_.clear();
        link_generations_.clear();
        latest_buffers_.clear();
        element_progress_.clear();
        break;
    case PipelineEventKind::Stopped:
        running_ = false;
        stopped_ = true;
        break;
    case PipelineEventKind::Error:
        running_ = false;
        last_error_ = event.message;
        break;
    case PipelineEventKind::ElementStarted:
    case PipelineEventKind::ElementStopped:
        break;
    case PipelineEventKind::BufferObserved:
        if (event.buffer) {
            const auto &link_id = event.buffer->link_id;
            // Reset the per-link count when a new buffer generation flows so the
            // graph reflects the current generation rather than accumulating
            // across reruns. Generation is derived from the vector clock
            // (Phase 27).
            const auto generation_it = link_generations_.find(link_id);
            if (generation_it == link_generations_.end() || generation_it->second != event.buffer->generation) {
                observed_counts_[link_id] = 0;
                link_generations_[link_id] = event.buffer->generation;
            }
            latest_buffers_[link_id] = *event.buffer;
            ++observed_counts_[link_id];

            // Track the running component-wise max of every observed buffer clock
            // = the current production count per slot index (Phase 27).
            const auto &clock = event.buffer->provenance;
            if (max_provenance_.size() < clock.size()) {
                max_provenance_.resize(clock.size(), 0u);
            }
            for (std::size_t i = 0; i < clock.size(); ++i) {
                max_provenance_[i] = std::max(max_provenance_[i], clock[i]);
            }
        }
        break;
    case PipelineEventKind::PropertyChanged:
        if (event.property_change && has_topology_) {
            apply_property_change_to_topology(topology_, *event.property_change);
        }
        break;
    case PipelineEventKind::TelemetryChanged:
        if (event.telemetry_change && has_topology_) {
            apply_telemetry_change_to_topology(topology_, *event.telemetry_change);
        }
        break;
    case PipelineEventKind::ProgressReported:
        if (event.progress) {
            element_progress_[event.progress->element.element_name] = ElementProgressState{
                .fraction = event.progress->fraction,
                .message = event.progress->message,
                .index = event.progress->index,
                .total = event.progress->total,
                .status = event.progress->status,
                .updated = std::chrono::steady_clock::now(),
            };
        }
        break;
    case PipelineEventKind::CommandAccepted:
    case PipelineEventKind::CommandRejected:
    case PipelineEventKind::CommandApplied:
        // Command outcomes are surfaced through recent_events; the displayed
        // property value is updated by the paired PropertyChanged event. Surface
        // rejected/failed command details as the visible error, and clear a stale
        // error once a command applies successfully.
        if (event.command) {
            if (event.command->status == PipelineCommandStatus::Rejected
                || event.command->status == PipelineCommandStatus::Failed) {
                last_error_ = event.command->detail.empty() ? std::string("command failed") : event.command->detail;
            } else if (event.command->status == PipelineCommandStatus::Applied) {
                last_error_.reset();
            }
        }
        break;
    }
}

void PipelineControlRuntime::bind(Pipeline &pipeline) {
    const auto topology = pipeline.topology_snapshot();
    for (const auto &element_snapshot : topology.elements) {
        if (auto live_element = pipeline.find_element(element_snapshot.name)) {
            bind(std::move(live_element));
        }
    }
}

void PipelineControlRuntime::bind(std::shared_ptr<Element> element) {
    if (!element) {
        return;
    }
    elements_[element->name()] = std::move(element);
}

void PipelineControlRuntime::clear() {
    elements_.clear();
    open_elements_.clear();
    focus_element_.reset();
    text_edit_states_.clear();
    changes_.clear();
    last_error_.reset();
    edits_enabled_ = true;
}

bool PipelineControlRuntime::has_element(std::string_view element_name) const {
    return element(element_name) != nullptr;
}

std::vector<std::string> PipelineControlRuntime::element_names() const {
    std::vector<std::string> names;
    names.reserve(elements_.size());
    for (const auto &[name, weak_element] : elements_) {
        if (!weak_element.expired()) {
            names.push_back(name);
        }
    }
    return names;
}

std::shared_ptr<Element> PipelineControlRuntime::element(std::string_view element_name) const {
    const auto found = elements_.find(std::string(element_name));
    if (found == elements_.end()) {
        return {};
    }
    return found->second.lock();
}

void PipelineControlRuntime::open(std::string_view element_name) {
    if (has_element(element_name)) {
        auto name = std::string(element_name);
        open_elements_.insert(name);
        focus_element_ = std::move(name);
    }
}

void PipelineControlRuntime::close(std::string_view element_name) { open_elements_.erase(std::string(element_name)); }

bool PipelineControlRuntime::is_open(std::string_view element_name) const {
    return open_elements_.contains(std::string(element_name));
}

bool PipelineControlRuntime::take_focus_request(std::string_view element_name) {
    if (focus_element_ && *focus_element_ == element_name) {
        focus_element_.reset();
        return true;
    }
    return false;
}

void PipelineControlRuntime::set_edits_enabled(bool enabled) { edits_enabled_ = enabled; }

bool PipelineControlRuntime::edits_enabled() const { return edits_enabled_; }

void PipelineControlRuntime::set_observer(std::shared_ptr<PipelineObserver> observer) {
    observer_ = std::move(observer);
}

std::shared_ptr<PipelineObserver> PipelineControlRuntime::observer() const { return observer_; }

void PipelineControlRuntime::bind_session(PipelineSession *session) { session_ = session; }

PipelineSession *PipelineControlRuntime::session() const { return session_; }

void PipelineControlRuntime::set_element_mutex(std::mutex *element_mutex) { element_mutex_ = element_mutex; }

std::mutex *PipelineControlRuntime::element_mutex() const { return element_mutex_; }

void PipelineControlRuntime::request_start() { start_requested_.store(true); }

void PipelineControlRuntime::request_stop() {
    user_stopped_.store(true);
    std::function<void()> stopper;
    {
        const std::lock_guard<std::mutex> lock(run_stopper_mutex_);
        stopper = run_stopper_;
    }
    if (stopper) {
        stopper(); // interrupt a blocking run immediately
    }
}

void PipelineControlRuntime::request_pause() {
    if (session_ != nullptr) {
        session_->request_pause();
    }
}

void PipelineControlRuntime::request_resume() {
    if (session_ != nullptr) {
        session_->request_resume();
    }
}

void PipelineControlRuntime::request_apply() {
    if (session_ != nullptr) {
        session_->apply_staged(); // flush staged edits into the live queue
    }
}

bool PipelineControlRuntime::take_start_request() { return start_requested_.exchange(false); }

bool PipelineControlRuntime::take_user_stopped() { return user_stopped_.exchange(false); }

void PipelineControlRuntime::set_run_stopper(std::function<void()> stopper) {
    const std::lock_guard<std::mutex> lock(run_stopper_mutex_);
    run_stopper_ = std::move(stopper);
}

void PipelineControlRuntime::set_auto_recompute(bool on) {
    auto_recompute_.store(on);
    // Orthogonal to state: manual (auto off) stages edits until Apply; auto queues
    // them live for immediate effect, in every state.
    if (session_ != nullptr) {
        session_->set_manual_apply(!on);
    }
}

bool PipelineControlRuntime::auto_recompute() const { return auto_recompute_.load(); }

bool PipelineControlRuntime::set_property(std::string_view element_name, std::string_view property_name,
                                          PropertyValue value) {
    if (!edits_enabled_ && session_ == nullptr) {
        last_error_ = "control edits are disabled";
        return false;
    }

    auto live_element = element(element_name);
    if (!live_element) {
        last_error_ = "control element is not available";
        return false;
    }
    if (!live_element->has_property(property_name)) {
        last_error_ = "control property is not available";
        return false;
    }

    // Session client path (Phase 25): submit a command instead of mutating the
    // live element. The worker applies it at a safe point and emits the
    // accepted/applied events; the UI must not touch live elements directly.
    if (session_ != nullptr) {
        session_->submit(SetPropertyCommand{
            .element_name = std::string(element_name),
            .property_name = std::string(property_name),
            .value = std::move(value),
        });
        last_error_.reset();
        return true;
    }

    auto effect = PropertyEffect{};
    for (const auto &spec : live_element->property_specs()) {
        if (spec.name == property_name) {
            effect = spec.effect;
            break;
        }
    }

    const auto previous_value = property_value_to_string(live_element->property(property_name));
    try {
        live_element->set_property(property_name, std::move(value));
    } catch (const std::exception &error) {
        last_error_ = error.what();
        return false;
    }

    const auto new_value = property_value_to_string(live_element->property(property_name));
    changes_.push_back(PipelineControlChange{
        .element_name = std::string(element_name),
        .property_name = std::string(property_name),
        .previous_value = previous_value,
        .new_value = new_value,
        .effect = effect,
    });

    if (observer_) {
        auto event = PipelineEvent{
            .kind = PipelineEventKind::PropertyChanged,
            .property_change = PipelinePropertyChangeObservation{
                .element =
                    PipelineEndpointSnapshot{
                        .element_type = live_element->element_type().empty() ? std::string("Element")
                                                                             : live_element->element_type(),
                        .element_name = live_element->name(),
                        .element_klass = live_element->element_kclass(),
                    },
                .property_name = std::string(property_name),
                .value_type = property_value_type_name(live_element->property(property_name)),
                .previous_value = previous_value,
                .new_value = new_value,
                .effect = effect,
            },
            .element_name = live_element->name(),
            .message = "property changed",
        };
        event.sequence = next_event_sequence_++;
        try {
            observer_->observe(event);
        } catch (...) {
        }
    }

    last_error_.reset();
    return true;
}

std::vector<PipelineControlChange> PipelineControlRuntime::take_changes() {
    std::vector<PipelineControlChange> changes;
    changes.swap(changes_);
    return changes;
}

const std::optional<std::string> &PipelineControlRuntime::last_error() const { return last_error_; }

std::vector<std::string> PipelineControlRuntime::open_element_names() const {
    return {open_elements_.begin(), open_elements_.end()};
}

PipelineControlRuntime::TextEditState &PipelineControlRuntime::text_edit_state(std::string_view key) {
    return text_edit_states_[std::string(key)];
}

PipelineControlRuntime::ColorEditState &PipelineControlRuntime::color_edit_state(std::string_view key) {
    return color_edit_states_[std::string(key)];
}

void PipelineControlRuntime::set_last_error(std::string error) { last_error_ = std::move(error); }

namespace {

// One placed cell in the layered layout: a real element box or a virtual
// routing point for a multi-column link.
struct GraphCell {
    int layer = 0;
    int row = 0;
};

struct GraphLayout {
    std::vector<GraphCell> elements;                    // one per topology element
    std::vector<std::vector<GraphCell>> link_waypoints; // intermediate routing points per topology link
    std::vector<int> layer_sizes;                       // cells (real + virtual) per layer
    int layer_count = 1;
    int max_rows = 1;
};

// Layered DAG layout (longest-path layering + barycenter ordering). Sources
// land in the leftmost column stacked vertically; fan-out elements such as Tee
// keep their children grouped so the fork is visible. Long edges are expanded
// into chains of virtual cells so they route through their own lanes instead of
// crossing element boxes.
[[nodiscard]] GraphLayout compute_graph_layout(const PipelineTopologySnapshot &topology) {
    const auto element_count = topology.elements.size();
    GraphLayout layout;
    layout.elements.assign(element_count, GraphCell{});
    layout.link_waypoints.assign(topology.links.size(), {});
    if (element_count == 0) {
        return layout;
    }

    std::map<std::string, std::size_t> index_by_name;
    for (std::size_t i = 0; i < element_count; ++i) {
        index_by_name.emplace(topology.elements[i].name, i);
    }

    const auto find_endpoint = [&](const PipelineLinkSnapshot &link) -> std::optional<std::pair<std::size_t, std::size_t>> {
        const auto s = index_by_name.find(link.source.element_name);
        const auto t = index_by_name.find(link.sink.element_name);
        if (s == index_by_name.end() || t == index_by_name.end() || s->second == t->second) {
            return std::nullopt;
        }
        return std::pair{s->second, t->second};
    };

    // Deduplicated element edges drive the longest-path layering.
    std::vector<std::vector<std::size_t>> succ(element_count);
    std::vector<std::vector<std::size_t>> pred(element_count);
    std::set<std::pair<std::size_t, std::size_t>> seen_edges;
    for (const auto &link : topology.links) {
        const auto endpoint = find_endpoint(link);
        if (!endpoint || !seen_edges.insert(*endpoint).second) {
            continue;
        }
        succ[endpoint->first].push_back(endpoint->second);
        pred[endpoint->second].push_back(endpoint->first);
    }

    std::vector<int> layer(element_count, 0);
    {
        std::vector<int> indeg(element_count, 0);
        for (std::size_t i = 0; i < element_count; ++i) {
            indeg[i] = static_cast<int>(pred[i].size());
        }
        std::vector<std::size_t> ready;
        for (std::size_t i = 0; i < element_count; ++i) {
            if (indeg[i] == 0) {
                ready.push_back(i);
            }
        }
        for (std::size_t head = 0; head < ready.size(); ++head) {
            const auto u = ready[head];
            for (const auto v : succ[u]) {
                layer[v] = std::max(layer[v], layer[u] + 1);
                if (--indeg[v] == 0) {
                    ready.push_back(v);
                }
            }
        }
    }

    int layer_count = 1;
    for (std::size_t i = 0; i < element_count; ++i) {
        layer_count = std::max(layer_count, layer[i] + 1);
    }

    // Expand each link that spans more than one column into a chain of virtual
    // cells so it claims its own routing lane.
    std::size_t virtual_count = 0;
    for (const auto &link : topology.links) {
        const auto endpoint = find_endpoint(link);
        if (!endpoint) {
            continue;
        }
        const auto span = layer[endpoint->second] - layer[endpoint->first];
        if (span > 1) {
            virtual_count += static_cast<std::size_t>(span - 1);
        }
    }

    const auto total_cells = element_count + virtual_count;
    std::vector<int> cell_layer(total_cells, 0);
    for (std::size_t i = 0; i < element_count; ++i) {
        cell_layer[i] = layer[i];
    }
    std::vector<std::vector<std::size_t>> down(total_cells);
    std::vector<std::vector<std::size_t>> up(total_cells);
    std::vector<std::vector<std::size_t>> link_chain(topology.links.size());

    std::size_t next_virtual = element_count;
    for (std::size_t li = 0; li < topology.links.size(); ++li) {
        const auto endpoint = find_endpoint(topology.links[li]);
        if (!endpoint || layer[endpoint->second] <= layer[endpoint->first]) {
            continue;
        }
        auto prev = endpoint->first;
        for (int L = layer[endpoint->first] + 1; L < layer[endpoint->second]; ++L) {
            const auto vc = next_virtual++;
            cell_layer[vc] = L;
            link_chain[li].push_back(vc);
            down[prev].push_back(vc);
            up[vc].push_back(prev);
            prev = vc;
        }
        down[prev].push_back(endpoint->second);
        up[endpoint->second].push_back(prev);
    }

    // Order cells within each layer with barycenter sweeps to reduce crossings.
    std::vector<std::vector<std::size_t>> rows(static_cast<std::size_t>(layer_count));
    for (std::size_t c = 0; c < total_cells; ++c) {
        rows[static_cast<std::size_t>(cell_layer[c])].push_back(c);
    }

    std::vector<float> pos(total_cells, 0.0F);
    const auto refresh_pos = [&]() {
        for (auto &column : rows) {
            for (std::size_t r = 0; r < column.size(); ++r) {
                pos[column[r]] = static_cast<float>(r);
            }
        }
    };
    refresh_pos();

    // Weighted-median ordering key (Gansner et al.): pulls a cell toward the
    // median position of its neighbors in the adjacent layer; cells with no
    // neighbors keep their current position.
    const auto median_key = [&](std::size_t cell, const std::vector<std::vector<std::size_t>> &neigh) -> float {
        const auto &list = neigh[cell];
        if (list.empty()) {
            return pos[cell];
        }
        std::vector<float> values;
        values.reserve(list.size());
        for (const auto m : list) {
            values.push_back(pos[m]);
        }
        std::sort(values.begin(), values.end());
        const auto count = values.size();
        const auto mid = count / 2;
        if (count % 2 == 1) {
            return values[mid];
        }
        if (count == 2) {
            return (values[0] + values[1]) * 0.5F;
        }
        const auto left = values[mid - 1] - values[0];
        const auto right = values[count - 1] - values[mid];
        const auto denom = left + right;
        if (denom <= 0.0F) {
            return (values[mid - 1] + values[mid]) * 0.5F;
        }
        return (values[mid - 1] * right + values[mid] * left) / denom;
    };

    const auto order_layer = [&](std::size_t layer_index, const std::vector<std::vector<std::size_t>> &neigh) {
        std::vector<std::pair<float, std::size_t>> keyed;
        keyed.reserve(rows[layer_index].size());
        for (const auto cell : rows[layer_index]) {
            keyed.emplace_back(median_key(cell, neigh), cell);
        }
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (std::size_t r = 0; r < keyed.size(); ++r) {
            rows[layer_index][r] = keyed[r].second;
        }
    };

    // Crossings between edges incident to v and to w when v precedes w.
    const auto pair_crossings = [&](std::size_t v, std::size_t w,
                                    const std::vector<std::vector<std::size_t>> &neigh) {
        auto crossings = 0;
        for (const auto a : neigh[v]) {
            for (const auto b : neigh[w]) {
                if (pos[a] > pos[b]) {
                    ++crossings;
                }
            }
        }
        return crossings;
    };

    // Greedy adjacent-swap pass: swap neighbors in a layer whenever it reduces
    // crossings against both adjacent layers. Repeats until no swap helps.
    const auto transpose = [&]() {
        auto improved = true;
        auto guard = 0;
        while (improved && guard++ < 64) {
            improved = false;
            for (auto &column : rows) {
                for (std::size_t i = 0; i + 1 < column.size(); ++i) {
                    const auto v = column[i];
                    const auto w = column[i + 1];
                    const auto before = pair_crossings(v, w, up) + pair_crossings(v, w, down);
                    const auto after = pair_crossings(w, v, up) + pair_crossings(w, v, down);
                    if (after < before) {
                        std::swap(column[i], column[i + 1]);
                        pos[v] = static_cast<float>(i + 1);
                        pos[w] = static_cast<float>(i);
                        improved = true;
                    }
                }
            }
        }
    };

    const auto count_crossings = [&]() {
        auto crossings = 0;
        for (std::size_t L = 0; L + 1 < rows.size(); ++L) {
            std::vector<std::pair<float, float>> edges;
            for (const auto u : rows[L]) {
                for (const auto v : down[u]) {
                    edges.emplace_back(pos[u], pos[v]);
                }
            }
            for (std::size_t a = 0; a < edges.size(); ++a) {
                for (std::size_t b = a + 1; b < edges.size(); ++b) {
                    const auto first_before = edges[a].first < edges[b].first;
                    const auto first_after = edges[a].first > edges[b].first;
                    if ((first_before && edges[a].second > edges[b].second) ||
                        (first_after && edges[a].second < edges[b].second)) {
                        ++crossings;
                    }
                }
            }
        }
        return crossings;
    };

    // Alternate median sweeps with transpose, keeping the best ordering found.
    auto best_rows = rows;
    auto best_crossings = count_crossings();
    for (int iteration = 0; iteration < 8 && best_crossings > 0; ++iteration) {
        if (iteration % 2 == 0) {
            for (int L = 1; L < layer_count; ++L) {
                order_layer(static_cast<std::size_t>(L), up);
                refresh_pos();
            }
        } else {
            for (int L = layer_count - 2; L >= 0; --L) {
                order_layer(static_cast<std::size_t>(L), down);
                refresh_pos();
            }
        }
        transpose();
        const auto crossings = count_crossings();
        if (crossings < best_crossings) {
            best_crossings = crossings;
            best_rows = rows;
        }
    }
    rows = best_rows;
    refresh_pos();

    std::vector<int> cell_row(total_cells, 0);
    for (auto &column : rows) {
        for (std::size_t r = 0; r < column.size(); ++r) {
            cell_row[column[r]] = static_cast<int>(r);
        }
    }

    layout.layer_sizes.assign(static_cast<std::size_t>(layer_count), 0);
    for (std::size_t L = 0; L < rows.size(); ++L) {
        layout.layer_sizes[L] = static_cast<int>(rows[L].size());
        layout.max_rows = std::max(layout.max_rows, static_cast<int>(rows[L].size()));
    }
    layout.layer_count = layer_count;

    for (std::size_t i = 0; i < element_count; ++i) {
        layout.elements[i] = GraphCell{cell_layer[i], cell_row[i]};
    }
    for (std::size_t li = 0; li < link_chain.size(); ++li) {
        for (const auto vc : link_chain[li]) {
            layout.link_waypoints[li].push_back(GraphCell{cell_layer[vc], cell_row[vc]});
        }
    }
    return layout;
}

// Vertical position on a box edge for a named pad, distributing pads evenly so
// each Tee output (src_0, src_1, ...) leaves from its own point.
[[nodiscard]] float pad_anchor_y(ImVec2 box_min, ImVec2 box_max, const std::vector<PipelinePadSnapshot> &pads,
                                 std::string_view pad_name) {
    if (!pads.empty()) {
        for (std::size_t i = 0; i < pads.size(); ++i) {
            if (pads[i].name == pad_name) {
                const auto fraction = static_cast<float>(i + 1) / static_cast<float>(pads.size() + 1);
                return box_min.y + (box_max.y - box_min.y) * fraction;
            }
        }
    }
    return (box_min.y + box_max.y) * 0.5F;
}

void draw_arrowhead(ImDrawList *draw_list, ImVec2 tip, ImU32 color) {
    constexpr auto length = 8.0F;
    constexpr auto half = 4.6F;
    draw_list->AddTriangleFilled(ImVec2(tip.x - length, tip.y - half), ImVec2(tip.x - length, tip.y + half), tip, color);
}

// Progress display gate + fade: a bar shows while work is in flight, then lingers ~1.5 s after
// it completes or is cancelled and fades out, so a terminal outcome does not vanish instantly.
struct ProgressDisplay {
    bool show = false;
    float alpha = 1.0F;
};

[[nodiscard]] ProgressDisplay progress_display(const PipelineGraphRuntime::ElementProgressState &state) {
    if (state.status == ProgressStatus::Active && state.fraction < 1.0) {
        return {true, 1.0F};
    }
    static constexpr auto linger = 1.5F;
    const auto age = std::chrono::duration<float>(std::chrono::steady_clock::now() - state.updated).count();
    if (age >= linger) {
        return {false, 0.0F};
    }
    return {true, 1.0F - age / linger};
}

// Percentage overlay like "52%".
[[nodiscard]] std::string progress_percent(double fraction) {
    return std::to_string(static_cast<int>(std::clamp(fraction, 0.0, 1.0) * 100.0 + 0.5)) + "%";
}

// Draw a progress bar (track + fill) shared by the node and the panel. While `running`, a soft
// highlight band sweeps left->right across the filled portion each ~1.1 s to signal live activity;
// `alpha` fades the whole bar out once complete. Uses ImGui::GetTime(), which advances every frame
// (the plot loop renders continuously), so the sweep is smooth.
void draw_progress_bar(ImDrawList *draw_list, ImVec2 min, ImVec2 max, double fraction, float alpha, bool running,
                       ProgressStatus status) {
    const auto rounding = std::min(2.0F, (max.y - min.y) * 0.5F);
    const auto track = ImGui::GetColorU32(ImVec4(0.14F, 0.16F, 0.20F, 0.85F * alpha));
    const auto cancelled = status == ProgressStatus::Cancelled;
    const auto fill = ImGui::GetColorU32(cancelled ? ImVec4(0.88F, 0.25F, 0.28F, 0.96F * alpha)
                                                   : ImVec4(0.36F, 0.72F, 0.42F, 0.96F * alpha));
    draw_list->AddRectFilled(min, max, track, rounding);

    const auto display_fraction = cancelled ? 1.0 : fraction;
    const auto frac = static_cast<float>(std::clamp(display_fraction, 0.0, 1.0));
    const auto fill_x = min.x + (max.x - min.x) * frac;
    if (fill_x > min.x + 0.5F) {
        draw_list->AddRectFilled(min, ImVec2(fill_x, max.y), fill, rounding);
    }
    if (!running || cancelled) {
        return;
    }

    // A highlight band sweeps across the WHOLE bar so the animation stays visible even at ~1%:
    // brighter over the filled portion, a faint glow over the (dark) track. One moving band drawn
    // in two clipped passes; each half of the band fades to transparent at its outer edge.
    const auto bar_width = max.x - min.x;
    static constexpr auto period = 1.1F; // seconds per sweep
    const auto phase = static_cast<float>(std::fmod(ImGui::GetTime(), static_cast<double>(period))) / period;
    const auto band = std::max(9.0F, bar_width * 0.14F);
    const auto center = min.x - band + (bar_width + 2.0F * band) * phase;
    const auto transparent = ImGui::GetColorU32(ImVec4(0.92F, 1.0F, 0.94F, 0.0F));
    const auto sweep = [&](ImVec2 clip_min, ImVec2 clip_max, float strength) {
        if (clip_max.x <= clip_min.x + 0.5F) {
            return;
        }
        const auto bright = ImGui::GetColorU32(ImVec4(0.92F, 1.0F, 0.94F, strength * alpha));
        draw_list->PushClipRect(clip_min, clip_max, true);
        draw_list->AddRectFilledMultiColor(ImVec2(center - band, min.y), ImVec2(center, max.y),
                                           transparent, bright, bright, transparent);
        draw_list->AddRectFilledMultiColor(ImVec2(center, min.y), ImVec2(center + band, max.y),
                                           bright, transparent, transparent, bright);
        draw_list->PopClipRect();
    };
    sweep(ImVec2(fill_x, min.y), max, 0.14F);        // faint glow over the empty track
    sweep(min, ImVec2(fill_x, max.y), 0.42F);        // brighter over the filled portion
}

// Floating summary window listing every element currently (or just-recently) reporting progress,
// each with a labelled bar and its message. Only appears while something is in flight.
void draw_progress_panel(const PipelineGraphRuntime &runtime, bool animate_progress) {
    std::vector<std::pair<const std::string *, const PipelineGraphRuntime::ElementProgressState *>> active;
    for (const auto &[name, state] : runtime.element_progress()) {
        if (progress_display(state).show) {
            active.emplace_back(&name, &state);
        }
    }
    if (active.empty()) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(320.0F, 0.0F), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Progress")) {
        auto *draw_list = ImGui::GetWindowDrawList();
        for (const auto &[name, state] : active) {
            const auto display = progress_display(*state);
            const auto running = animate_progress && state->status == ProgressStatus::Active
                && state->fraction < 1.0;
            ImGui::TextUnformatted(name->c_str());

            // Full-width animated bar with the percentage centered over it.
            const auto width = ImGui::GetContentRegionAvail().x;
            const auto height = ImGui::GetTextLineHeight() + 5.0F;
            const auto origin = ImGui::GetCursorScreenPos();
            ImGui::Dummy(ImVec2(width, height));
            const ImVec2 bar_min(origin.x, origin.y);
            const ImVec2 bar_max(origin.x + width, origin.y + height);
            draw_progress_bar(draw_list, bar_min, bar_max, state->fraction, display.alpha, running, state->status);
            const auto percent = progress_percent(state->status == ProgressStatus::Cancelled ? 1.0 : state->fraction);
            const auto text_size = ImGui::CalcTextSize(percent.c_str());
            draw_list->AddText(ImVec2(bar_min.x + (width - text_size.x) * 0.5F, bar_min.y + (height - text_size.y) * 0.5F),
                               ImGui::GetColorU32(ImVec4(1.0F, 1.0F, 1.0F, 0.95F * display.alpha)), percent.c_str());

            if (state->status == ProgressStatus::Cancelled) {
                const auto message = state->message.empty() ? "cancelled" : state->message.c_str();
                ImGui::TextColored(ImVec4(0.96F, 0.32F, 0.34F, display.alpha), "%s", message);
            } else if (!state->message.empty()) {
                ImGui::TextDisabled("%s", state->message.c_str());
            }
        }
    }
    ImGui::End();
}

} // namespace

void draw_pipeline_graph(PipelineGraphRuntime &runtime) { draw_pipeline_graph(runtime, nullptr); }

void draw_pipeline_graph(PipelineGraphRuntime &runtime, PipelineControlRuntime *control_runtime) {
    runtime.drain_events();
    // PipelineGraphRuntime's Started/Stopped events do not encode Paused or the
    // held Idle state. When a session is bound, use its authoritative lifecycle
    // state so progress motion freezes immediately outside Running. Keep the
    // event-derived fallback for callers that draw a graph without a session.
    const auto animate_progress =
        control_runtime != nullptr && control_runtime->session() != nullptr
            ? progress_animation_enabled(control_runtime->session()->state())
            : runtime.running();
    if (control_runtime != nullptr) {
        control_runtime->set_edits_enabled(!runtime.running());
    }

    ImGui::SetNextWindowSize(ImVec2(1120.0F, 680.0F), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("LeakFlow Pipeline Graph")) {
        ImGui::End();
        if (control_runtime != nullptr) {
            draw_open_element_control_windows(*control_runtime);
        }
        return;
    }

    if (runtime.running()) {
        ImGui::TextColored(ImVec4(0.30F, 0.90F, 0.52F, 1.0F), "running");
    } else if (runtime.stopped()) {
        ImGui::TextColored(ImVec4(0.65F, 0.78F, 0.95F, 1.0F), "stopped");
    } else {
        ImGui::TextColored(ImVec4(0.82F, 0.72F, 0.36F, 1.0F), "waiting");
    }

    ImGui::SameLine();
    if (runtime.has_topology()) {
        const auto summary = std::to_string(runtime.topology().elements.size()) + " elements, " +
                             std::to_string(runtime.topology().links.size()) + " links";
        ImGui::TextUnformatted(summary.c_str());
    } else {
        ImGui::TextUnformatted("no topology snapshot yet");
    }

    if (runtime.last_error()) {
        ImGui::TextColored(ImVec4(1.00F, 0.42F, 0.42F, 1.0F), "error: %s", runtime.last_error()->c_str());
    }

    if (!runtime.has_topology()) {
        ImGui::Separator();
        ImGui::TextUnformatted("Waiting for pipeline events...");
        draw_recent_events(runtime);
        ImGui::End();
        if (control_runtime != nullptr) {
            draw_open_element_control_windows(*control_runtime);
        }
        return;
    }

    const auto &topology = runtime.topology();
    const auto layout = compute_graph_layout(topology);

    static constexpr auto box_width = 174.0F;
    static constexpr auto box_height = 60.0F;
    static constexpr auto h_gap = 116.0F;
    static constexpr auto v_gap = 30.0F;
    static constexpr auto margin_x = 40.0F;
    static constexpr auto margin_y = 28.0F;

    const auto span_size = [](int count, float size, float gap) {
        return static_cast<float>(count) * size + static_cast<float>(std::max(0, count - 1)) * gap;
    };
    const auto content_width = margin_x * 2.0F + span_size(layout.layer_count, box_width, h_gap);
    const auto content_height = margin_y * 2.0F + span_size(layout.max_rows, box_height, v_gap);
    const auto canvas_height = std::clamp(content_height + 12.0F, 240.0F, 640.0F);

    ImGui::BeginChild("pipeline_graph_canvas", ImVec2(0.0F, canvas_height), true, ImGuiWindowFlags_HorizontalScrollbar);

    // Node/link hover uses IsMouseHoveringRect, which ignores window z-order; gate it on the
    // canvas actually being the hovered window so clicks/tooltips do not pass through a plot
    // window sitting on top of the graph.
    const bool canvas_hovered = ImGui::IsWindowHovered();

    auto *draw_list = ImGui::GetWindowDrawList();
    const auto origin = ImGui::GetCursorScreenPos();
    const auto total_rows_height = span_size(layout.max_rows, box_height, v_gap);

    // Each layer is centered vertically so fan-outs spread symmetrically.
    const auto cell_min = [&](const GraphCell &cell) {
        const auto col_count = layout.layer_sizes.empty() ? 1 : layout.layer_sizes[static_cast<std::size_t>(cell.layer)];
        const auto col_height = span_size(col_count, box_height, v_gap);
        const auto start_y = origin.y + margin_y + (total_rows_height - col_height) * 0.5F;
        const auto x = origin.x + margin_x + static_cast<float>(cell.layer) * (box_width + h_gap);
        const auto y = start_y + static_cast<float>(cell.row) * (box_height + v_gap);
        return ImVec2(x, y);
    };
    const auto cell_center = [&](const GraphCell &cell) {
        const auto min = cell_min(cell);
        return ImVec2(min.x + box_width * 0.5F, min.y + box_height * 0.5F);
    };

    std::map<std::string, ImVec2> box_min_by_name;
    std::map<std::string, ImVec2> box_max_by_name;
    std::map<std::string, const PipelineElementSnapshot *> element_by_name;
    for (std::size_t index = 0; index < topology.elements.size(); ++index) {
        const auto &element = topology.elements[index];
        const auto min = cell_min(layout.elements[index]);
        const auto max = ImVec2(min.x + box_width, min.y + box_height);
        box_min_by_name.emplace(element.name, min);
        box_max_by_name.emplace(element.name, max);
        element_by_name.emplace(element.name, &element);
    }

    const auto mouse = ImGui::GetMousePos();

    // Nodes take click priority over links so a click over a node box pins the
    // node, not a link passing near it.
    bool mouse_over_node = false;
    for (const auto &element : topology.elements) {
        if (ImGui::IsMouseHoveringRect(box_min_by_name.at(element.name), box_max_by_name.at(element.name))) {
            mouse_over_node = true;
            break;
        }
    }

    for (std::size_t index = 0; index < topology.links.size(); ++index) {
        const auto &link = topology.links[index];
        const auto source_min = box_min_by_name.find(link.source.element_name);
        const auto source_max = box_max_by_name.find(link.source.element_name);
        const auto sink_min = box_min_by_name.find(link.sink.element_name);
        const auto sink_max = box_max_by_name.find(link.sink.element_name);
        const auto source_el = element_by_name.find(link.source.element_name);
        const auto sink_el = element_by_name.find(link.sink.element_name);
        if (source_min == box_min_by_name.end() || source_max == box_max_by_name.end() ||
            sink_min == box_min_by_name.end() || sink_max == box_max_by_name.end() ||
            source_el == element_by_name.end() || sink_el == element_by_name.end()) {
            continue;
        }

        const auto p_start = ImVec2(source_max->second.x,
                                    pad_anchor_y(source_min->second, source_max->second,
                                                 source_el->second->output_pads, link.source.pad_name));
        const auto p_end = ImVec2(sink_min->second.x,
                                  pad_anchor_y(sink_min->second, sink_max->second, sink_el->second->input_pads,
                                               link.sink.pad_name));

        std::vector<ImVec2> points;
        points.reserve(layout.link_waypoints[index].size() + 2);
        points.push_back(p_start);
        for (const auto &waypoint : layout.link_waypoints[index]) {
            points.push_back(cell_center(waypoint));
        }
        points.push_back(p_end);

        // Build an orthogonal (right-angle) polyline: each hop leaves horizontally,
        // makes a single vertical jog in the inter-column gap, then arrives
        // horizontally. Waypoints keep the horizontal runs in box-free lanes.
        std::vector<ImVec2> path;
        path.reserve(points.size() * 3);
        path.push_back(points.front());
        for (std::size_t s = 0; s + 1 < points.size(); ++s) {
            const auto a = points[s];
            const auto b = points[s + 1];
            const auto mid_x = (a.x + b.x) * 0.5F;
            path.push_back(ImVec2(mid_x, a.y));
            path.push_back(ImVec2(mid_x, b.y));
            path.push_back(b);
        }

        const auto latest = latest_buffer_for(runtime, link.id);
        const auto count = runtime.observed_count(link.id);
        const auto color = runtime.is_link_pinned(link.id)
                               ? ImGui::GetColorU32(ImVec4(1.00F, 0.82F, 0.28F, 0.98F))
                               : link_color(latest.has_value());

        auto min_distance = 1.0e9F;
        for (std::size_t s = 0; s + 1 < path.size(); ++s) {
            min_distance = std::min(min_distance, distance_to_segment(mouse, path[s], path[s + 1]));
        }
        const auto hovered = canvas_hovered && min_distance <= 6.0F;

        const auto thickness = hovered ? 3.6F : 2.4F;
        draw_list->AddPolyline(path.data(), static_cast<int>(path.size()), color, ImDrawFlags_None, thickness);
        draw_list->AddCircleFilled(p_start, 3.4F, color);
        draw_arrowhead(draw_list, p_end, color);

        // Label the originating pad on fan-out elements so each fork is identifiable.
        if (source_el->second->output_pads.size() > 1) {
            draw_list->AddText(ImVec2(p_start.x + 6.0F, p_start.y - 16.0F),
                               ImGui::GetColorU32(ImVec4(0.80F, 0.84F, 0.92F, 0.92F)), link.source.pad_name.c_str());
        }
        if (count > 0 && !path.empty()) {
            const auto mid = path[path.size() / 2];
            const auto badge = "x" + std::to_string(count);
            draw_list->AddText(ImVec2(mid.x - 6.0F, mid.y - 16.0F),
                               ImGui::GetColorU32(ImVec4(0.62F, 0.86F, 0.70F, 0.92F)), badge.c_str());
        }

        if (hovered && !mouse_over_node && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            runtime.toggle_pinned_link(link.id);
        }

        if (hovered) {
            draw_link_tooltip(runtime, link, latest, count);
        }
    }

    for (const auto &element : topology.elements) {
        const auto min = box_min_by_name.at(element.name);
        const auto max = box_max_by_name.at(element.name);
        const auto hovered = canvas_hovered && ImGui::IsMouseHoveringRect(min, max);
        const auto pinned = runtime.is_element_pinned(element.name);
        const auto colors = klass_colors(element.klass);
        const auto border = pinned ? ImGui::GetColorU32(ImVec4(1.00F, 0.82F, 0.28F, 0.98F))
                            : hovered ? ImGui::GetColorU32(ImVec4(1.00F, 1.00F, 1.00F, 0.95F))
                                      : colors.border;
        draw_list->AddRectFilled(min, max, colors.fill, 6.0F);
        draw_list->AddRectFilled(min, ImVec2(min.x + 6.0F, max.y), colors.accent, 6.0F,
                                 ImDrawFlags_RoundCornersLeft);
        draw_list->AddRect(min, max, border, 6.0F, 0, (hovered || pinned) ? 2.6F : 1.4F);

        const auto label = element_label(element);
        const auto text_size = ImGui::CalcTextSize(label.c_str());
        const auto label_position = ImVec2(min.x + std::max(8.0F, (box_width - text_size.x) * 0.5F), min.y + 14.0F);
        draw_list->AddText(ImVec2(label_position.x + 1.0F, label_position.y + 1.0F),
                           ImGui::GetColorU32(ImVec4(0.0F, 0.0F, 0.0F, 0.65F)), label.c_str());
        draw_list->AddText(label_position, ImGui::GetColorU32(ImVec4(1.0F, 1.0F, 1.0F, 0.98F)), label.c_str());

        const auto klass = element.klass.empty() ? std::string("unspecified") : element.klass;
        const auto klass_size = ImGui::CalcTextSize(klass.c_str());
        draw_list->AddText(ImVec2(min.x + std::max(8.0F, (box_width - klass_size.x) * 0.5F), min.y + 36.0F),
                           ImGui::GetColorU32(ImVec4(0.88F, 0.92F, 0.98F, 0.86F)), klass.c_str());

        // Live progress (long-running elements such as GaussianMixture): a thin bar along the
        // node's bottom edge, faded once complete. The message shows only in the Progress panel.
        if (const auto progress_it = runtime.element_progress().find(element.name);
            progress_it != runtime.element_progress().end()) {
            const auto &state = progress_it->second;
            const auto display = progress_display(state);
            if (display.show) {
                draw_progress_bar(draw_list, ImVec2(min.x + 8.0F, max.y - 7.0F), ImVec2(max.x - 8.0F, max.y - 3.0F),
                                  state.fraction, display.alpha,
                                  animate_progress && state.status == ProgressStatus::Active && state.fraction < 1.0,
                                  state.status);
            }
        }

        bool gear_hovered = false;
        if (control_runtime != nullptr && control_runtime->has_element(element.name)) {
            const auto gear_center = ImVec2(max.x - 15.0F, min.y + 15.0F);
            const auto gear_min = ImVec2(gear_center.x - 11.0F, gear_center.y - 11.0F);
            ImGui::PushID(element.name.c_str());
            ImGui::SetCursorScreenPos(gear_min);
            ImGui::InvisibleButton("element_control", ImVec2(22.0F, 22.0F));
            gear_hovered = ImGui::IsItemHovered();
            if (ImGui::IsItemClicked()) {
                control_runtime->open(element.name);
            }
            const auto gear_color =
                gear_hovered ? ImGui::GetColorU32(ImVec4(1.0F, 1.0F, 1.0F, 1.0F))
                             : ImGui::GetColorU32(ImVec4(0.92F, 0.96F, 1.0F, 0.82F));
            draw_gear_icon(draw_list, gear_center, gear_color);
            if (gear_hovered) {
                ImGui::SetTooltip("Open controls for %s", element.name.c_str());
            }
            ImGui::PopID();
        }

        // Click the node body (not the gear) to pin a collapsible info window.
        if (hovered && !gear_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            runtime.toggle_pinned_element(element.name);
        }

        if (hovered) {
            draw_element_tooltip(runtime, element);
        }
    }

    ImGui::Dummy(ImVec2(content_width, content_height));
    ImGui::EndChild();

    draw_provenance_panel(runtime);
    draw_buffer_clocks_panel(runtime);
    draw_recent_events(runtime);
    ImGui::End();

    // Pinned, collapsible info windows the user opened by clicking a node/link.
    draw_pinned_info_windows(runtime);

    // Floating summary of every element currently reporting progress (auto-hides when idle).
    draw_progress_panel(runtime, animate_progress);

    if (control_runtime != nullptr) {
        draw_open_element_control_windows(*control_runtime);
    }
}

void draw_pipeline_controls(PipelineControlRuntime &runtime) {
    ImGui::SetNextWindowSize(ImVec2(420.0F, 520.0F), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("LeakFlow Pipeline Controls")) {
        if (!runtime.edits_enabled()) {
            ImGui::TextWrapped("Controls are paused while the pipeline worker is running.");
            ImGui::Separator();
        }
        if (runtime.last_error()) {
            ImGui::TextColored(ImVec4(1.00F, 0.42F, 0.42F, 1.0F), "%s", runtime.last_error()->c_str());
            ImGui::Separator();
        }

        const auto names = runtime.element_names();
        if (names.empty()) {
            ImGui::TextUnformatted("No controls are bound.");
        }
        for (const auto &name : names) {
            auto live_element = runtime.element(name);
            if (!live_element) {
                continue;
            }

            ImGui::PushID(name.c_str());
            if (ImGui::SmallButton("Control")) {
                runtime.open(name);
            }
            ImGui::SameLine();
            ImGui::TextUnformatted((live_element->element_type() + "@" + live_element->name()).c_str());
            ImGui::PopID();
        }
    }
    ImGui::End();

    draw_open_element_control_windows(runtime);
}

std::optional<Buffer> run_pipeline_graph_until_closed(
    Pipeline &pipeline,
    PlotRuntime &plot_runtime,
    const PlotLoopOptions &options) {
    PipelineControlRuntime control_runtime;
    return run_pipeline_graph_until_closed(pipeline, plot_runtime, control_runtime, options);
}

std::optional<Buffer> run_pipeline_graph_until_closed(
    Pipeline &pipeline,
    PlotRuntime &plot_runtime,
    PipelineControlRuntime &control_runtime,
    const PlotLoopOptions &options) {
    auto graph_runtime = std::make_shared<PipelineGraphRuntime>();
    const auto previous_observer = pipeline.observer();
    const auto previous_control_observer = control_runtime.observer();
    pipeline.set_observer(graph_runtime);
    control_runtime.set_observer(graph_runtime);
    control_runtime.bind(pipeline);

    std::optional<Buffer> result;
    std::exception_ptr failure;
    std::jthread worker([&pipeline, &result, &failure](std::stop_token token) {
        try {
            // Closing the window requests the worker's stop, which interrupts a
            // paced live source mid-trace and exits the pump promptly (S11.8).
            pipeline.set_stop_token(token);
            result = pipeline.run();
        } catch (...) {
            failure = std::current_exception();
        }
    });

    try {
        run_until_closed(plot_runtime, *graph_runtime, control_runtime, options);
        worker.join();
        pipeline.set_observer(previous_observer);
        control_runtime.set_observer(previous_control_observer);
    } catch (...) {
        worker.join();
        pipeline.set_observer(previous_observer);
        control_runtime.set_observer(previous_control_observer);
        throw;
    }

    if (failure) {
        std::rethrow_exception(failure);
    }

    return result;
}

std::optional<Buffer> run_pipeline_graph_until_closed(PipelineSession &session, PlotRuntime &plot_runtime,
                                                      const PlotLoopOptions &options) {
    PipelineControlRuntime control_runtime;
    return run_pipeline_graph_until_closed(session, plot_runtime, control_runtime, options);
}

std::optional<Buffer> run_pipeline_graph_until_closed(PipelineSession &session, PlotRuntime &plot_runtime,
                                                      PipelineControlRuntime &control_runtime,
                                                      const PlotLoopOptions &options) {
    auto graph_runtime = std::make_shared<PipelineGraphRuntime>();
    const auto previous_observer = session.observer();
    const auto previous_control_observer = control_runtime.observer();
    session.set_observer(graph_runtime);
    control_runtime.set_observer(graph_runtime);
    control_runtime.bind(session.pipeline());
    control_runtime.bind_session(&session);

    std::mutex element_mutex;
    control_runtime.set_element_mutex(&element_mutex);

    // Two-way trace-index sync: moving a vertical slider submits a SetProperty so
    // the owning TracePlot's trace_index property (and the gear/graph) follows.
    plot_runtime.trace_view()->set_trace_index_listener([&session](std::string_view element_name, int trace_index) {
        session.submit(SetPropertyCommand{
            .element_name = std::string(element_name),
            .property_name = "trace_index",
            .value = static_cast<std::int64_t>(trace_index),
        });
    });

    // Overlay sub-group x-axis sync: a member's x_axis change propagates to peers.
    plot_runtime.trace_view()->set_x_axis_listener([&session](std::string_view element_name, std::string_view x_axis) {
        session.submit(SetPropertyCommand{
            .element_name = std::string(element_name),
            .property_name = "x_axis",
            .value = std::string(x_axis),
        });
    });

    // Show the pipeline structure immediately, before any run -- so a graph opened in
    // Stopped (no --auto-start) is not blank while waiting for the Start button.
    graph_runtime->observe(PipelineEvent{
        .kind = PipelineEventKind::TopologySnapshot,
        .topology = session.pipeline().topology_snapshot(),
    });

    std::exception_ptr failure;
    const bool auto_start = options.auto_start;
    std::jthread worker([&session, &control_runtime, &element_mutex, &plot_runtime, &failure,
                         auto_start](std::stop_token token) {
        using namespace std::chrono_literals;
        // A user Stop recycles the run: tear down, clear the plot snapshots and
        // slider indexes, and return to Stopped so the next Start replays from 0.
        // Element properties live on the elements and are left untouched.
        const auto user_stop = [&session, &element_mutex, &plot_runtime]() {
            {
                const std::lock_guard<std::mutex> lock(element_mutex);
                session.stop();
            }
            plot_runtime.clear();
            session.set_state(PipelineSessionState::Stopped);
        };
        try {
            // Request-driven streaming lifecycle (Stopped/Running/Paused/Idle). A
            // "run" produces until EOS (live) or one sweep (offline). Pause/Resume are
            // handled inside the run by the session's safe-point park; Stop interrupts
            // it via a run-stopper; natural completion lands in Idle (held & editable).
            bool start_pending = auto_start;
            session.set_state(PipelineSessionState::Stopped);

            while (!token.stop_requested()) {
                // ---- Stopped: wait for Start; edits only stage (no cache) ----
                if (!start_pending) {
                    start_pending = control_runtime.take_start_request();
                }
                if (!start_pending) {
                    {
                        const std::lock_guard<std::mutex> lock(element_mutex);
                        (void)session.drain_commands();
                    }
                    std::this_thread::sleep_for(8ms);
                    continue;
                }
                start_pending = false;
                (void)control_runtime.take_user_stopped(); // clear any stale stop

                // ---- Launch a run ----
                std::stop_source run_stop;
                const auto stop_the_run = [&run_stop, &session]() {
                    run_stop.request_stop();
                    session.request_resume(); // wake a paused run so it tears down
                };
                std::stop_callback window_cb(token, stop_the_run);
                control_runtime.set_run_stopper(stop_the_run);
                session.request_resume(); // never start a run paused
                session.set_stop_token(run_stop.get_token());

                try {
                    if (session.pipeline().should_run_threaded()) {
                        session.set_safe_point_mutex(&element_mutex);
                        (void)session.run_once(); // blocks; pause/stop handled inside the segments
                        session.set_safe_point_mutex(nullptr);
                    } else {
                        {
                            const std::lock_guard<std::mutex> lock(element_mutex);
                            session.start();
                            (void)session.run_sweep(); // first sweep (populates the rerun cache offline)
                        }
                        // A live-no-queue source streams single-threaded here, with pause
                        // and edits applied between buffers.
                        while (!run_stop.stop_requested() && session.pipeline().has_live_source()
                               && !session.pipeline().all_live_sources_at_eos()) {
                            session.wait_while_paused(run_stop.get_token());
                            if (run_stop.stop_requested()) {
                                break;
                            }
                            {
                                const std::lock_guard<std::mutex> lock(element_mutex);
                                (void)session.drain_commands();
                            }
                            (void)session.run_sweep();
                        }
                    }
                } catch (...) {
                    const auto run_failure = std::current_exception();
                    const auto message = exception_message(run_failure, "unknown pipeline graph run failure");

                    session.set_safe_point_mutex(nullptr);
                    control_runtime.set_run_stopper(nullptr);
                    session.request_resume();
                    (void)control_runtime.take_user_stopped();

                    report_graph_run_failure(session, message);
                    {
                        const std::lock_guard<std::mutex> lock(element_mutex);
                        session.stop();
                    }
                    session.set_state(PipelineSessionState::Stopped);
                    observe_worker_event(session, PipelineEventKind::Stopped, "failed");

                    if (token.stop_requested()) {
                        break;
                    }
                    continue;
                }

                control_runtime.set_run_stopper(nullptr);
                if (token.stop_requested()) {
                    break;
                }

                // ---- Run ended: Stopped (user) or Idle (EOS / sweep done) ----
                if (control_runtime.take_user_stopped()) {
                    user_stop();
                    continue;
                }
                session.set_state(PipelineSessionState::Idle);

                // ---- Idle: held & editable. Auto-recompute applies each edit from
                // cache immediately; manual waits for an Apply. Start is disabled in
                // Idle (the UI gates it): re-running requires Stop then Start, so a
                // stray start request is consumed and ignored here. ----
                while (!token.stop_requested() && session.state() == PipelineSessionState::Idle) {
                    (void)control_runtime.take_start_request();
                    if (control_runtime.take_user_stopped()) {
                        user_stop();
                        break;
                    }
                    // Live-queued edits recompute from cache; manual-staged edits
                    // wait in staging until Apply flushes them here. Uniform drain.
                    if (session.pending_command_count() > 0) {
                        const std::lock_guard<std::mutex> lock(element_mutex);
                        (void)session.drain_commands();
                    }
                    std::this_thread::sleep_for(8ms);
                }
            }

            const std::lock_guard<std::mutex> lock(element_mutex);
            session.stop();
        } catch (...) {
            failure = std::current_exception();
            const auto message = exception_message(failure, "unknown pipeline graph worker failure");
            session.set_safe_point_mutex(nullptr);
            control_runtime.set_run_stopper(nullptr);
            session.request_resume();
            report_graph_run_failure(session, message);
            session.set_state(PipelineSessionState::Stopped);
            observe_worker_event(session, PipelineEventKind::Stopped, "failed");
        }
    });

    const auto restore = [&]() {
        worker.request_stop();
        worker.join();
        plot_runtime.trace_view()->set_trace_index_listener(nullptr);
        plot_runtime.trace_view()->set_x_axis_listener(nullptr);
        control_runtime.set_element_mutex(nullptr);
        control_runtime.bind_session(nullptr);
        session.set_observer(previous_observer);
        control_runtime.set_observer(previous_control_observer);
    };

    try {
        run_until_closed(plot_runtime, *graph_runtime, control_runtime, options);
        restore();
    } catch (...) {
        restore();
        throw;
    }

    if (failure) {
        std::rethrow_exception(failure);
    }

    return std::nullopt;
}

} // namespace leakflow::plot
