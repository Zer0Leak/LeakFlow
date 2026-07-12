#include "leakflow/core/pipeline.hpp"

#include "leakflow/core/active_element.hpp"
#include "leakflow/core/pipeline_segments.hpp"
#include "leakflow/core/provenance.hpp"
#include "leakflow/core/summary_document.hpp"
#include "leakflow/log/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace leakflow {
namespace {

const Pad *find_pad_named(const std::vector<Pad> &pads, std::string_view name) {
    for (const auto &pad : pads) {
        if (pad.name() == name) {
            return &pad;
        }
    }

    return nullptr;
}

bool links_are_equal(const PadLink &left, const PadLink &right) {
    return left.source_element == right.source_element && left.source_pad_name == right.source_pad_name &&
           left.sink_element == right.sink_element && left.sink_pad_name == right.sink_pad_name;
}

bool source_endpoints_are_equal(const PadLink &left, const PadLink &right) {
    return left.source_element == right.source_element && left.source_pad_name == right.source_pad_name;
}

bool sink_endpoints_are_equal(const PadLink &left, const PadLink &right) {
    return left.sink_element == right.sink_element && left.sink_pad_name == right.sink_pad_name;
}

bool output_annotation_endpoints_are_equal(const OutputMetadataAnnotation &left,
                                           const OutputMetadataAnnotation &right) {
    return left.source_element == right.source_element && left.target.kind == right.target.kind &&
           left.target.direction == right.target.direction && left.target.name == right.target.name;
}

bool has_duplicate_link(const std::vector<PadLink> &links, const PadLink &candidate) {
    for (const auto &link : links) {
        if (links_are_equal(link, candidate)) {
            return true;
        }
    }

    return false;
}

bool has_source_endpoint_link(const std::vector<PadLink> &links, const PadLink &candidate) {
    for (const auto &link : links) {
        if (source_endpoints_are_equal(link, candidate)) {
            return true;
        }
    }

    return false;
}

bool has_sink_endpoint_link(const std::vector<PadLink> &links, const PadLink &candidate) {
    for (const auto &link : links) {
        if (sink_endpoints_are_equal(link, candidate)) {
            return true;
        }
    }

    return false;
}

std::string input_pad_context(const std::vector<PadLink> &incoming) {
    if (incoming.empty()) {
        return "without connected input pads";
    }
    std::ostringstream message;
    message << "with input pad";
    if (incoming.size() != 1) {
        message << 's';
    }
    message << " '";
    for (std::size_t i = 0; i < incoming.size(); ++i) {
        if (i != 0) {
            message << ',';
        }
        message << incoming[i].sink_pad_name;
    }
    message << "'";
    return message.str();
}

std::string element_label(const std::shared_ptr<Element> &element) {
    const auto type =
        element->element_type().empty() ? std::string_view("Element") : std::string_view(element->element_type());
    return std::string(type) + "@" + element->name();
}

std::string endpoint_label(const std::shared_ptr<Element> &element, std::string_view pad_name) {
    return element_label(element) + "." + std::string(pad_name);
}

std::string route_key_for(const PadLink &link) {
    return endpoint_label(link.source_element, link.source_pad_name) + " -> " +
           endpoint_label(link.sink_element, link.sink_pad_name);
}

PipelinePadSnapshot pad_snapshot(const Pad &pad) {
    return PipelinePadSnapshot{
        .name = pad.name(),
        .direction = pad.direction(),
        .presence = pad.presence(),
        .caps = pad.caps(),
    };
}

std::vector<PipelinePadSnapshot> pad_snapshots_for(const std::vector<Pad> &pads) {
    std::vector<PipelinePadSnapshot> snapshots;
    snapshots.reserve(pads.size());
    for (const auto &pad : pads) {
        snapshots.push_back(pad_snapshot(pad));
    }
    return snapshots;
}

PipelineTelemetrySnapshot telemetry_snapshot_for(const TelemetrySnapshot &telemetry) {
    return PipelineTelemetrySnapshot{
        .name = telemetry.name,
        .value_type = property_value_type_name(telemetry.value),
        .value = property_value_to_string(telemetry.value),
        .unit = telemetry.unit,
        .description = telemetry.description,
        .value_hint = telemetry.value_hint,
        .kind = telemetry.kind,
    };
}

PipelineElementSnapshot element_snapshot(const Element &element) {
    std::vector<PipelinePropertySnapshot> properties;
    properties.reserve(element.properties().size());
    for (const auto &[name, value] : element.properties()) {
        auto effect = PropertyEffect{};
        auto writable = true;
        for (const auto &spec : element.property_specs()) {
            if (spec.name == name) {
                effect = spec.effect;
                writable = spec.writable;
                break;
            }
        }
        properties.push_back(PipelinePropertySnapshot{
            .name = name,
            .value_type = property_value_type_name(value),
            .value = property_value_to_string(value),
            .effect = std::move(effect),
            .writable = writable,
        });
    }

    std::vector<PipelineTelemetrySnapshot> telemetry;
    for (const auto &snapshot : element.telemetry_snapshot()) {
        telemetry.push_back(telemetry_snapshot_for(snapshot));
    }

    return PipelineElementSnapshot{
        .type_name = element.element_type().empty() ? "Element" : element.element_type(),
        .name = element.name(),
        .klass = element.element_kclass(),
        .properties = std::move(properties),
        .telemetry = std::move(telemetry),
        .input_pads = pad_snapshots_for(element.input_pads()),
        .output_pads = pad_snapshots_for(element.output_pads()),
        .pad_templates = pad_snapshots_for(element.pad_templates()),
    };
}

PipelineEndpointSnapshot endpoint_snapshot(const Element &element, std::string_view pad_name) {
    return PipelineEndpointSnapshot{
        .element_type = element.element_type().empty() ? "Element" : element.element_type(),
        .element_name = element.name(),
        .element_klass = element.element_kclass(),
        .pad_name = std::string(pad_name),
    };
}

std::string compact_value(std::string_view value) {
    constexpr auto max_length = std::size_t{120};
    std::string compact;
    compact.reserve(value.size());

    for (const auto character : value) {
        if (character == '\n' || character == '\r' || character == '\t') {
            compact.push_back(' ');
        } else {
            compact.push_back(character);
        }
    }

    if (compact.size() > max_length) {
        compact.resize(max_length - 3);
        compact += "...";
    }

    return compact;
}

std::string exception_message(const std::exception_ptr &failure) {
    if (!failure) {
        return "unknown threaded segment failure";
    }
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception &error) {
        return error.what();
    } catch (...) {
        return "unknown threaded segment failure";
    }
}

void append_payload_field(std::vector<std::string> &lines, const SummaryField &field, std::size_t depth) {
    const auto indent = std::string(depth * 2, ' ');
    auto line = indent + field.label;
    if (!field.value.text.empty()) {
        line += "=" + field.value.text;
    }
    lines.push_back(compact_value(line));
    for (const auto &child : field.children) {
        append_payload_field(lines, child, depth + 1);
    }
}

[[nodiscard]] std::vector<std::string> payload_summary_lines(const Buffer &buffer) {
    if (!buffer.has_payload()) {
        return {};
    }

    SummarySection section("Payload");
    buffer.payload()->describe(section, log::summary_level());

    std::vector<std::string> lines;
    for (const auto &field : section.fields) {
        if (field.label == "payload") {
            continue;
        }
        append_payload_field(lines, field, 0);
    }
    return lines;
}

PipelineBufferSnapshot buffer_snapshot(const Buffer &buffer) {
    return PipelineBufferSnapshot{
        .caps = buffer.caps(),
        .metadata = buffer.metadata(),
        .has_payload = buffer.has_payload(),
        .payload_type = buffer.has_payload() ? buffer.payload()->type_name() : std::string("none"),
        .payload_summary = payload_summary_lines(buffer),
    };
}

PipelineBufferObservation buffer_observation_for(const PadLink &link, const Buffer &buffer, std::uint64_t sequence) {
    return PipelineBufferObservation{
        .link_id = pipeline_link_id(link.source_element->name(), link.source_pad_name, link.sink_element->name(),
                                    link.sink_pad_name),
        .source = endpoint_snapshot(*link.source_element, link.source_pad_name),
        .sink = endpoint_snapshot(*link.sink_element, link.sink_pad_name),
        .buffer = buffer_snapshot(buffer),
        .sequence = sequence,
        .generation = provenance_generation(buffer.provenance()),
        .provenance = buffer.provenance(),
    };
}

void append_metadata_summary(std::ostringstream &output, const Buffer &buffer) {
    output << "  metadata (" << buffer.metadata().size() << "):";
    if (buffer.metadata().empty()) {
        output << " none\n";
        return;
    }

    output << '\n';
    for (const auto &[key, value] : buffer.metadata()) {
        output << "    " << key << '=' << compact_value(value) << '\n';
    }
}

void append_payload_summary(std::ostringstream &output, const Buffer &buffer) {
    output << "  payload:\n";
    if (!buffer.has_payload()) {
        output << "    type=none\n";
        return;
    }

    output << "    type=" << buffer.payload()->type_name() << '\n';
    for (const auto &line : payload_summary_lines(buffer)) {
        output << "    " << line << '\n';
    }
    for (const auto &[key, value] : buffer.caps().params()) {
        output << "    " << key << '=' << value << '\n';
    }
}

void log_first_buffer_on_route(const PadLink &link, const Buffer &buffer, std::set<std::string> &logged_routes) {
    const auto route = route_key_for(link);
    if (!logged_routes.insert(route).second) {
        return;
    }

    std::ostringstream message;
    message << "buffer flow\n"
            << "  " << route << '\n'
            << "  caps=" << buffer.caps().to_string() << '\n';
    append_metadata_summary(message, buffer);
    append_payload_summary(message, buffer);

    auto text = message.str();
    if (!text.empty() && text.back() == '\n') {
        text.pop_back();
    }

    auto record = link.source_element->make_log_record(log::LogLevel::Info, "pipeline", std::move(text));
    record.fields.emplace("src", endpoint_label(link.source_element, link.source_pad_name));
    record.fields.emplace("sink", endpoint_label(link.sink_element, link.sink_pad_name));
    record.fields.emplace("caps", buffer.caps().to_string());
    log::write(std::move(record));
}

void apply_output_metadata_annotations(const std::vector<OutputMetadataAnnotation> &annotations, const PadLink &link,
                                       Buffer &buffer) {
    const auto apply_for_kind = [&](MetadataPadTargetKind kind) {
        for (const auto &annotation : annotations) {
            if (annotation.source_element != link.source_element || annotation.target.kind != kind) {
                continue;
            }
            if (annotation.target.direction && *annotation.target.direction != PadDirection::Output) {
                continue;
            }
            if (annotation.target.kind == MetadataPadTargetKind::PadName &&
                annotation.target.name != link.source_pad_name) {
                continue;
            }
            if (annotation.target.kind == MetadataPadTargetKind::PadTemplate &&
                !metadata_pad_template_matches(annotation.target.name, link.source_pad_name)) {
                continue;
            }

            for (const auto &[key, value] : annotation.metadata) {
                buffer.set_metadata(key, value);
            }
        }
    };

    apply_for_kind(MetadataPadTargetKind::AllPads);
    apply_for_kind(MetadataPadTargetKind::PadTemplate);
    apply_for_kind(MetadataPadTargetKind::PadName);
}

bool output_pad_template_matches_any(const Element &element, std::string_view pad_template) {
    for (const auto &declared_template : element.pad_templates()) {
        if (declared_template.direction() == PadDirection::Output && declared_template.name() == pad_template) {
            return true;
        }
    }

    for (const auto &pad : element.output_pads()) {
        if (metadata_pad_template_matches(pad_template, pad.name())) {
            return true;
        }
    }

    return false;
}

bool has_output_pad_or_template(const Element &element) {
    if (!element.output_pads().empty()) {
        return true;
    }

    for (const auto &pad_template : element.pad_templates()) {
        if (pad_template.direction() == PadDirection::Output) {
            return true;
        }
    }

    return false;
}

const PadLink *single_incoming_link_for(const std::vector<PadLink> &links, const Element &element) {
    const PadLink *incoming = nullptr;

    for (const auto &link : links) {
        if (link.sink_element.get() != &element) {
            continue;
        }
        if (incoming != nullptr) {
            return nullptr;
        }

        incoming = &link;
    }

    return incoming;
}

bool caps_are_forwarded(const Caps &caps) { return caps.type() == generic_buffer_caps_type || caps_are_any(caps); }

Caps resolved_source_caps_for(const std::vector<PadLink> &links, const Element &element,
                              std::string_view source_pad_name) {
    const auto *source_pad = find_pad_named(element.output_pads(), source_pad_name);
    if (source_pad == nullptr) {
        throw std::invalid_argument("source output pad does not exist");
    }
    if (!caps_are_forwarded(source_pad->caps())) {
        return source_pad->caps();
    }

    const auto *incoming = single_incoming_link_for(links, element);
    if (incoming == nullptr) {
        return source_pad->caps();
    }

    const auto *input_pad = find_pad_named(incoming->sink_element->input_pads(), incoming->sink_pad_name);
    if (input_pad == nullptr || !caps_are_forwarded(input_pad->caps())) {
        return source_pad->caps();
    }

    auto upstream_caps = resolved_source_caps_for(links, *incoming->source_element, incoming->source_pad_name);
    if (caps_are_forwarded(upstream_caps)) {
        return source_pad->caps();
    }

    return upstream_caps;
}

std::optional<std::size_t> find_element_index(const std::vector<std::shared_ptr<Element>> &elements,
                                              const std::shared_ptr<Element> &target) {
    for (std::size_t index = 0; index < elements.size(); ++index) {
        if (elements[index] == target) {
            return index;
        }
    }

    return std::nullopt;
}

std::string caps_mismatch_message(const Element &source_element, std::string_view source_pad_name,
                                  const Caps &source_caps, const Element &sink_element, std::string_view sink_pad_name,
                                  const Caps &sink_caps) {
    std::ostringstream message;
    message << "link caps mismatch: " << source_element.name() << '.' << source_pad_name << " ("
            << source_caps.to_string() << ") -> " << sink_element.name() << '.' << sink_pad_name << " ("
            << sink_caps.to_string() << ')';

    if (source_caps.type() == generic_buffer_caps_type && sink_caps.type() != generic_buffer_caps_type) {
        message << "; source pad declares generic " << generic_buffer_caps_type << ", but sink requires concrete "
                << sink_caps.type();
    }

    return message.str();
}

} // namespace

class Pipeline::PipelineRuntimeContext final : public RuntimeContext {
public:
    PipelineRuntimeContext(Pipeline &pipeline, const BoundaryRuntimes &boundaries, std::stop_token stop,
                           std::mutex &shared, std::exception_ptr &failure, std::mutex &failure_mutex,
                           std::stop_source &run_stop, SegmentSafePoint safe_point)
        : pipeline_(pipeline)
        , boundaries_(boundaries)
        , stop_(stop)
        , shared_(shared)
        , failure_(failure)
        , failure_mutex_(failure_mutex)
        , run_stop_(run_stop)
        , safe_point_(std::move(safe_point))
    {
    }

    [[nodiscard]] bool push(Element &element, std::string_view source_pad, Buffer buffer) override
    {
        if (stop_.stop_requested()) {
            return false;
        }
        safe_point(element);
        if (stop_.stop_requested()) {
            return false;
        }

        {
            const std::lock_guard<std::mutex> lock(shared_);
            auto provenance = buffer.provenance();
            if (provenance.size() < pipeline_.next_slot_) {
                provenance.resize(pipeline_.next_slot_, 0u);
            }
            if (const auto base_it = pipeline_.element_base_.find(&element);
                base_it != pipeline_.element_base_.end()) {
                provenance[base_it->second] = pipeline_.next_emit_count(&element);
            }
            buffer.set_provenance(std::move(provenance));
        }

        return pipeline_.route_runtime_push(element, source_pad, std::move(buffer), boundaries_, stop_, shared_,
                                            runtime_mutex_, safe_point_, logged_routes_, passive_states_);
    }

    void end_of_stream(Element &element, std::string_view source_pad) override
    {
        const std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);
        pipeline_.route_runtime_eos(element, source_pad, boundaries_, runtime_mutex_, safe_point_, passive_states_);
    }

    void report_error(Element &element, std::string message) override
    {
        const std::lock_guard<std::mutex> lock(failure_mutex_);
        if (!failure_) {
            failure_ = std::make_exception_ptr(
                std::runtime_error("active element '" + element.name() + "' failed: " + std::move(message)));
        }
        run_stop_.request_stop();
    }

    void safe_point(Element &element) override
    {
        if (!safe_point_) {
            return;
        }
        std::vector<std::shared_ptr<Element>> elements;
        elements.reserve(1);
        for (const auto &candidate : pipeline_.elements_) {
            if (candidate.get() == &element) {
                elements.push_back(candidate);
                break;
            }
        }
        safe_point_(elements, stop_);
    }

    [[nodiscard]] std::stop_token stop_token() const override { return stop_; }
    [[nodiscard]] bool stop_requested() const override { return stop_.stop_requested(); }
    [[nodiscard]] bool is_paused() const override { return false; }
    void wait_if_paused() override
    {
        if (safe_point_) {
            safe_point_({}, stop_);
        }
    }

private:
    Pipeline &pipeline_;
    const BoundaryRuntimes &boundaries_;
    std::stop_token stop_;
    std::mutex &shared_;
    std::exception_ptr &failure_;
    std::mutex &failure_mutex_;
    std::stop_source &run_stop_;
    SegmentSafePoint safe_point_;
    std::recursive_mutex runtime_mutex_;
    std::set<std::string> logged_routes_;
    RuntimePassiveStates passive_states_;
};

std::shared_ptr<Element> Pipeline::add(std::shared_ptr<Element> element) {
    if (!element) {
        throw std::invalid_argument("pipeline element must not be null");
    }
    if (element->name().empty()) {
        throw std::invalid_argument("pipeline element name cannot be empty");
    }
    if (elements_by_name_.contains(element->name())) {
        throw std::invalid_argument("duplicate element instance name");
    }

    const auto element_name = element->name();
    elements_.push_back(std::move(element));
    elements_by_name_.emplace(element_name, elements_.back());
    elements_.back()->lock_name();
    elements_.back()->set_runtime_telemetry_enabled(runtime_telemetry_enabled_);
    elements_.back()->set_profiling_enabled(profiling_enabled_);
    elements_.back()->set_trace_sink(trace_sink_);
    elements_.back()->set_pause_waiter(pause_waiter_);

    // Allocate vector-clock provenance slots (Phase 27). 0-slot elements (Tee,
    // sinks) get no base index and forward provenance verbatim.
    if (elements_.back()->provenance_slots() > 0) {
        element_base_.emplace(elements_.back().get(), next_slot_);
        next_slot_ += static_cast<std::uint32_t>(elements_.back()->provenance_slots());
    }

    log::write(elements_.back()->make_log_record(log::LogLevel::Debug, "pipeline", "added element"));
    refresh_live_driven_flags();
    return elements_.back();
}

void Pipeline::link(std::shared_ptr<Element> source_element, std::string source_pad_name,
                          std::shared_ptr<Element> sink_element, std::string sink_pad_name) {
    if (!source_element) {
        throw std::invalid_argument("source element handle must not be null");
    }
    if (!sink_element) {
        throw std::invalid_argument("sink element handle must not be null");
    }

    const auto source_index = find_element_index(elements_, source_element);
    if (!source_index) {
        throw std::invalid_argument("source element handle does not exist in pipeline");
    }

    const auto sink_index = find_element_index(elements_, sink_element);
    if (!sink_index) {
        throw std::invalid_argument("sink element handle does not exist in pipeline");
    }

    if (*source_index >= *sink_index) {
        std::ostringstream message;
        message << "cannot link source endpoint '@" << source_element->name() << '.' << source_pad_name
                << "' to sink endpoint '@" << sink_element->name() << '.' << sink_pad_name << "': ";
        if (*source_index == *sink_index) {
            message << "source and sink are the same element; links must connect distinct elements";
        } else {
            message << "source element '@" << source_element->name() << "' was declared after sink element '@"
                    << sink_element->name() << "'; declare the source before the sink";
        }
        throw std::invalid_argument(message.str());
    }

    auto *source_pad = find_pad_named(source_element->output_pads(), source_pad_name);
    if (source_pad == nullptr) {
        (void)source_element->request_output_pad(source_pad_name);
        source_pad = find_pad_named(source_element->output_pads(), source_pad_name);
    }
    if (source_pad == nullptr) {
        throw std::invalid_argument("source output pad does not exist");
    }

    auto *sink_pad = find_pad_named(sink_element->input_pads(), sink_pad_name);
    if (sink_pad == nullptr) {
        (void)sink_element->request_input_pad(sink_pad_name);
        sink_pad = find_pad_named(sink_element->input_pads(), sink_pad_name);
    }
    if (sink_pad == nullptr) {
        throw std::invalid_argument("sink input pad does not exist");
    }

    const auto source_caps = resolved_source_caps_for(links_, *source_element, source_pad_name);
    if (!caps_are_compatible(source_caps, sink_pad->caps())) {
        throw std::invalid_argument(caps_mismatch_message(*source_element, source_pad_name, source_caps, *sink_element,
                                                          sink_pad_name, sink_pad->caps()));
    }

    PadLink candidate{
        std::move(source_element),
        std::move(source_pad_name),
        std::move(sink_element),
        std::move(sink_pad_name),
    };

    if (has_duplicate_link(links_, candidate)) {
        throw std::invalid_argument("duplicate pad link");
    }
    if (has_source_endpoint_link(links_, candidate)) {
        throw std::invalid_argument("source pad is already linked");
    }
    if (has_sink_endpoint_link(links_, candidate)) {
        throw std::invalid_argument("sink pad is already linked");
    }

    auto record = candidate.source_element->make_log_record(log::LogLevel::Debug, "pipeline", "linked pads");
    record.fields.emplace("source_pad", candidate.source_pad_name);
    record.fields.emplace("sink_element", candidate.sink_element->element_type());
    record.fields.emplace("sink_element_name", candidate.sink_element->name());
    record.fields.emplace("sink_element_kclass", candidate.sink_element->element_kclass());
    record.fields.emplace("sink_pad", candidate.sink_pad_name);
    log::write(std::move(record));

    links_.push_back(std::move(candidate));
    refresh_live_driven_flags();
}

std::size_t Pipeline::size() const { return elements_.size(); }

const std::vector<std::shared_ptr<Element>> &Pipeline::elements() const { return elements_; }

std::shared_ptr<Element> Pipeline::find_element(std::string_view name) const {
    const auto found = elements_by_name_.find(std::string(name));
    if (found == elements_by_name_.end()) {
        return nullptr;
    }

    return found->second;
}

std::shared_ptr<Element> Pipeline::element(std::string_view name) const {
    auto found = find_element(name);
    if (!found) {
        throw std::invalid_argument("unknown element name");
    }

    return found;
}

std::vector<std::shared_ptr<Element>> Pipeline::elements_by_type(std::string_view type_name) const {
    const auto normalized = element_default_name_prefix(type_name);
    std::vector<std::shared_ptr<Element>> matches;

    for (const auto &element : elements_) {
        if (element->element_type().empty()) {
            continue;
        }
        if (element_default_name_prefix(element->element_type()) == normalized) {
            matches.push_back(element);
        }
    }

    return matches;
}

const std::vector<PadLink> &Pipeline::links() const { return links_; }

Caps Pipeline::source_caps(const Element &source_element, std::string_view source_pad_name) const {
    return resolved_source_caps_for(links_, source_element, source_pad_name);
}

std::optional<std::string> Pipeline::link_caps_error() const {
    for (const auto &link : links_) {
        const auto source_caps_value = resolved_source_caps_for(links_, *link.source_element, link.source_pad_name);
        const auto *sink_pad = find_pad_named(link.sink_element->input_pads(), link.sink_pad_name);
        const Caps sink_caps_value = sink_pad == nullptr ? Caps(any_caps_type) : sink_pad->caps();
        if (!caps_are_compatible(source_caps_value, sink_caps_value)) {
            return caps_mismatch_message(*link.source_element, link.source_pad_name, source_caps_value,
                                         *link.sink_element, link.sink_pad_name, sink_caps_value);
        }
    }

    return std::nullopt;
}

PipelineTopologySnapshot Pipeline::topology_snapshot() const {
    PipelineTopologySnapshot snapshot;
    snapshot.runtime_telemetry_enabled = runtime_telemetry_enabled_;
    snapshot.elements.reserve(elements_.size());
    for (const auto &element : elements_) {
        snapshot.elements.push_back(element_snapshot(*element));
        if (const auto base_it = element_base_.find(element.get()); base_it != element_base_.end()) {
            snapshot.elements.back().provenance_base = base_it->second;
            snapshot.elements.back().provenance_slots = element->provenance_slots();
        }
    }

    snapshot.links.reserve(links_.size());
    for (const auto &link : links_) {
        const auto *sink_pad = find_pad_named(link.sink_element->input_pads(), link.sink_pad_name);
        snapshot.links.push_back(PipelineLinkSnapshot{
            .id = pipeline_link_id(link.source_element->name(), link.source_pad_name, link.sink_element->name(),
                                   link.sink_pad_name),
            .source = endpoint_snapshot(*link.source_element, link.source_pad_name),
            .sink = endpoint_snapshot(*link.sink_element, link.sink_pad_name),
            .source_caps = resolved_source_caps_for(links_, *link.source_element, link.source_pad_name),
            .sink_caps = sink_pad == nullptr ? Caps(any_caps_type) : sink_pad->caps(),
        });
    }

    return snapshot;
}

void Pipeline::set_observer(std::shared_ptr<PipelineObserver> observer) { observer_ = std::move(observer); }

std::shared_ptr<PipelineObserver> Pipeline::observer() const { return observer_; }

void Pipeline::set_runtime_telemetry_enabled(bool enabled) {
    runtime_telemetry_enabled_ = enabled;
    for (const auto &element : elements_) {
        element->set_runtime_telemetry_enabled(enabled);
    }
}

bool Pipeline::runtime_telemetry_enabled() const { return runtime_telemetry_enabled_; }

void Pipeline::set_profiling_enabled(bool enabled) {
    profiling_enabled_ = enabled;
    for (const auto &element : elements_) {
        element->set_profiling_enabled(enabled);
    }
}

bool Pipeline::profiling_enabled() const { return profiling_enabled_; }

void Pipeline::set_trace_sink(TelemetryTraceSink *sink) {
    trace_sink_ = sink;
    for (const auto &element : elements_) {
        element->set_trace_sink(sink);
    }
}

TelemetryTraceSink *Pipeline::trace_sink() const { return trace_sink_; }

ElementOutputs Pipeline::timed_process_pads(Element &element, ElementInputs inputs) {
    if (!element.profiling_enabled()) {
        return element.process_pads(std::move(inputs));
    }
    RuntimeTelemetryScopedTimer timer(element.process_stat(), trace_sink_, element.name() + ".process",
                                      element.name());
    return element.process_pads(std::move(inputs));
}

namespace {

void add_output_metadata_annotation_impl(std::vector<OutputMetadataAnnotation> &annotations,
                                         const std::vector<std::shared_ptr<Element>> &elements,
                                         std::shared_ptr<Element> source_element, MetadataPadTarget target,
                                         std::map<std::string, std::string> metadata) {
    if (!source_element) {
        throw std::invalid_argument("metadata annotation source element handle must not be null");
    }
    if (!find_element_index(elements, source_element)) {
        throw std::invalid_argument("metadata annotation source element handle does not exist in pipeline");
    }

    if (target.kind == MetadataPadTargetKind::AllPads) {
        if (target.direction && *target.direction != PadDirection::Output) {
            throw std::invalid_argument("output metadata annotation target must use output pads");
        }
        if (!has_output_pad_or_template(*source_element)) {
            throw std::invalid_argument("metadata annotation source element has no output pads");
        }
        target.direction = PadDirection::Output;
    } else if (target.kind == MetadataPadTargetKind::PadName) {
        if (target.direction && *target.direction != PadDirection::Output) {
            throw std::invalid_argument("output metadata annotation target must use output pads");
        }
        if (find_pad_named(source_element->output_pads(), target.name) == nullptr &&
            !source_element->can_request_output_pad(target.name)) {
            throw std::invalid_argument("metadata annotation source output pad does not exist");
        }
        target.direction = PadDirection::Output;
    } else if (target.kind == MetadataPadTargetKind::PadTemplate) {
        if (target.direction && *target.direction != PadDirection::Output) {
            throw std::invalid_argument("output metadata annotation target must use output pads");
        }
        if (!output_pad_template_matches_any(*source_element, target.name)) {
            throw std::invalid_argument("metadata annotation source output pad "
                                        "template matches no output pads");
        }
        target.direction = PadDirection::Output;
    }

    if (metadata.empty()) {
        throw std::invalid_argument("metadata annotation cannot be empty");
    }

    OutputMetadataAnnotation candidate{
        std::move(source_element),
        std::move(target),
        std::move(metadata),
    };

    for (auto &annotation : annotations) {
        if (!output_annotation_endpoints_are_equal(annotation, candidate)) {
            continue;
        }

        for (auto &[key, value] : candidate.metadata) {
            if (annotation.metadata.contains(key)) {
                throw std::invalid_argument("duplicate metadata annotation key for output pad");
            }
            annotation.metadata.emplace(std::move(key), std::move(value));
        }
        return;
    }

    annotations.push_back(std::move(candidate));
}

} // namespace

void Pipeline::add_output_metadata_annotation(std::shared_ptr<Element> source_element,
                                                    std::string source_pad_name,
                                                    std::map<std::string, std::string> metadata) {
    add_output_metadata_annotation_impl(output_metadata_annotations_, elements_, std::move(source_element),
                                        metadata_pad(PadDirection::Output, std::move(source_pad_name)),
                                        std::move(metadata));
}

void Pipeline::add_output_metadata_annotation(std::shared_ptr<Element> source_element,
                                                    std::map<std::string, std::string> metadata) {
    add_output_metadata_annotation_impl(output_metadata_annotations_, elements_, std::move(source_element),
                                        metadata_all_pads(PadDirection::Output), std::move(metadata));
}

void Pipeline::add_output_metadata_annotation_for_pad_template(std::shared_ptr<Element> source_element,
                                                                     std::string source_pad_template,
                                                                     std::map<std::string, std::string> metadata) {
    add_output_metadata_annotation_impl(output_metadata_annotations_, elements_, std::move(source_element),
                                        metadata_pad_template(PadDirection::Output, std::move(source_pad_template)),
                                        std::move(metadata));
}

std::vector<std::shared_ptr<Element>> Pipeline::linked_execution_order() const {
    if (links_.empty()) {
        return elements_;
    }

    std::vector<std::shared_ptr<Element>> order;
    order.reserve(links_.size() + 1);
    order.push_back(links_.front().source_element);

    for (const auto &link : links_) {
        if (link.source_element != order.back()) {
            throw std::invalid_argument("pad links must form one linear chain in insertion order");
        }

        order.push_back(link.sink_element);
    }

    return order;
}

std::vector<PadLink> Pipeline::outgoing_links_for(const std::shared_ptr<Element> &element) const {
    std::vector<PadLink> outgoing;

    for (const auto &link : links_) {
        if (link.source_element == element) {
            outgoing.push_back(link);
        }
    }

    return outgoing;
}

std::vector<PadLink> Pipeline::incoming_links_for(const std::shared_ptr<Element> &element) const {
    std::vector<PadLink> incoming;

    for (const auto &link : links_) {
        if (link.sink_element == element) {
            incoming.push_back(link);
        }
    }

    return incoming;
}

void Pipeline::set_caching_enabled(bool enabled) {
    caching_enabled_ = enabled;
    if (!enabled) {
        cached_outputs_.clear();
    }
}

bool Pipeline::caching_enabled() const { return caching_enabled_; }

std::vector<std::shared_ptr<Element>> Pipeline::replay_set(const std::shared_ptr<Element> &element) const {
    std::set<const Element *> reachable;
    std::vector<const Element *> stack;
    reachable.insert(element.get());
    stack.push_back(element.get());

    while (!stack.empty()) {
        const auto *current = stack.back();
        stack.pop_back();
        for (const auto &link : links_) {
            if (link.source_element.get() == current && reachable.insert(link.sink_element.get()).second) {
                stack.push_back(link.sink_element.get());
            }
        }
    }

    std::vector<std::shared_ptr<Element>> order;
    for (const auto &candidate : elements_) {
        if (reachable.contains(candidate.get())) {
            order.push_back(candidate);
        }
    }

    return order;
}

std::optional<Buffer> Pipeline::execute(const std::vector<std::shared_ptr<Element>> &order,
                                              bool seed_from_cache) {
    if (links_.empty()) {
        std::optional<Buffer> current;
        for (const auto &element : elements_) {
            if (stop_requested()) {
                cached_outputs_.clear();
                return std::nullopt;
            }
            std::vector<std::uint32_t> provenance;
            if (current) {
                provenance = current->provenance();
            }
            current = element->process(std::move(current));
            if (stop_requested()) {
                cached_outputs_.clear();
                return std::nullopt;
            }
            if (current) {
                provenance.resize(next_slot_, 0u);
                if (const auto base_it = element_base_.find(element.get()); base_it != element_base_.end()) {
                    provenance[base_it->second] = next_emit_count(element.get());
                }
                current->set_provenance(std::move(provenance));
            }
        }
        if (!caching_enabled_) {
            cached_outputs_.clear();
        }
        return current;
    }

    // `live` is the per-(element, output pad) routing map used for input gather.
    // A full sweep starts empty and populates as it walks topological order; a
    // rerun seeds from the persisted cache so upstream branches outside the
    // replay-set supply their still-valid outputs. See
    // docs/design/pipeline_controller.md.
    std::map<const Element *, std::map<std::string, Buffer>> live =
        seed_from_cache ? cached_outputs_ : std::map<const Element *, std::map<std::string, Buffer>>{};
    std::optional<Buffer> last_terminal_output;
    std::set<std::string> logged_routes;

    for (const auto &element : order) {
        if (stop_requested()) {
            cached_outputs_.clear();
            return std::nullopt;
        }
        const auto incoming = incoming_links_for(element);
        const auto outgoing = outgoing_links_for(element);

        if (incoming.empty() && outgoing.empty()) {
            continue;
        }

        ElementInputs inputs;
        for (const auto &link : incoming) {
            const auto source_it = live.find(link.source_element.get());
            if (source_it == live.end()) {
                throw std::invalid_argument("rerun is missing a cached upstream output; a full sweep is required");
            }
            const auto pad_it = source_it->second.find(link.source_pad_name);
            if (pad_it == source_it->second.end()) {
                throw std::invalid_argument("rerun is missing a cached upstream output pad; a full sweep is required");
            }
            inputs.emplace(link.sink_pad_name, pad_it->second);
        }

        if (inputs.size() != incoming.size()) {
            throw std::invalid_argument("linked element did not receive all connected inputs");
        }

        auto process_record = element->make_log_record(log::LogLevel::Debug, "pipeline", "processing element");
        process_record.fields.emplace("connected_inputs", std::to_string(inputs.size()));
        process_record.fields.emplace("connected_outputs", std::to_string(outgoing.size()));
        log::write(std::move(process_record));

        // Fold the input vector clocks (component-wise max, conflict-detecting)
        // before consuming the inputs. In one-shot/offline this always matches;
        // in partial rerun it verifies cached + recomputed branches are
        // provenance-consistent. See docs/design/dataflow_sync_model.md Section 6.
        std::vector<const std::vector<std::uint32_t> *> input_clocks;
        input_clocks.reserve(inputs.size());
        for (const auto &input_entry : inputs) {
            if (input_entry.second) {
                input_clocks.push_back(&input_entry.second->provenance());
            }
        }
        const auto merged_provenance = merge_provenance(input_clocks, next_slot_);

        ElementOutputs outputs;
        try {
            outputs = timed_process_pads(*element, std::move(inputs));
        } catch (const std::exception &error) {
            throw std::runtime_error("element '" + element->name() + "' failed " + input_pad_context(incoming) +
                                     ": " + error.what());
        }
        if (stop_requested()) {
            cached_outputs_.clear();
            return std::nullopt;
        }

        // Stamp provenance on every produced buffer: merged inputs plus this
        // element's own slot incremented once per firing (0-slot elements such as
        // Tee forward the merged clock verbatim, keeping branches identical).
        const auto base_it = element_base_.find(element.get());
        std::optional<std::uint32_t> own_count;
        if (base_it != element_base_.end() && !outputs.empty()) {
            own_count = next_emit_count(element.get());
        }
        for (auto &entry : outputs) {
            auto provenance = merged_provenance;
            if (own_count) {
                provenance[base_it->second] = *own_count;
            }
            entry.second.set_provenance(std::move(provenance));
        }

        if (outgoing.empty()) {
            last_terminal_output = outputs.empty()
                                       ? std::nullopt
                                       : std::optional<Buffer>(std::move(outputs.begin()->second));
            continue;
        }

        for (const auto &link : outgoing) {
            const auto output_it = outputs.find(link.source_pad_name);
            if (output_it != outputs.end()) {
                Buffer routed = output_it->second;
                apply_output_metadata_annotations(output_metadata_annotations_, link, routed);
                log_first_buffer_on_route(link, routed, logged_routes);
                emit(PipelineEvent{
                    .kind = PipelineEventKind::BufferObserved,
                    .buffer = buffer_observation_for(link, routed, next_buffer_sequence_++),
                });
                live[link.source_element.get()].insert_or_assign(link.source_pad_name, std::move(routed));
            } else if (const auto source_it = live.find(link.source_element.get()); source_it != live.end()) {
                source_it->second.erase(link.source_pad_name);
            }
        }
    }

    if (caching_enabled_) {
        cached_outputs_ = std::move(live);
    } else {
        cached_outputs_.clear();
    }

    return last_terminal_output;
}

// Progress push channel (see progress_sink.hpp): forwards Element::report_progress to the
// observer bus as a ProgressReported event. Created at start_all() when `this` is stationary
// (Pipeline is a movable value type, so a back-pointer must not be captured before the move).
class Pipeline::ProgressEventSink final : public ProgressSink {
public:
    explicit ProgressEventSink(Pipeline &pipeline) : pipeline_(pipeline) {}

    void report(Element &element, const ElementProgress &progress) override {
        pipeline_.emit(PipelineEvent{
            .kind = PipelineEventKind::ProgressReported,
            .progress =
                PipelineProgressObservation{
                    .element = endpoint_snapshot(element, std::string_view{}),
                    .fraction = progress.fraction,
                    .message = progress.message,
                    .index = progress.index,
                    .total = progress.total,
                    .status = progress.status,
                },
        });
    }

private:
    Pipeline &pipeline_;
};

void Pipeline::start_all() {
    started_count_ = 0;
    // (Re)create the progress channel now that `this` is stationary and hand it to every element
    // alongside the stop token, so a long process() can push a live progress bar to observers.
    progress_sink_ = std::make_unique<ProgressEventSink>(*this);
    // A run starts fresh: reset the vector-clock production counters so each run's
    // provenance begins at 0 (a Stop -> Start cycle restarts the clocks rather than
    // continuing from the previous run). Slot allocation (element_base_/next_slot_)
    // is topology, not per-run state, so it is preserved.
    emit_counts_.clear();
    refresh_live_driven_flags();

    log::LogRecord start_record{
        .level = log::LogLevel::Debug,
        .component = "pipeline",
        .message = "starting pipeline",
    };
    start_record.fields.emplace("elements", std::to_string(elements_.size()));
    start_record.fields.emplace("links", std::to_string(links_.size()));
    log::write(std::move(start_record));
    emit(PipelineEvent{
        .kind = PipelineEventKind::TopologySnapshot,
        .topology = topology_snapshot(),
    });
    emit(PipelineEvent{
        .kind = PipelineEventKind::Started,
        .message = "started",
    });

    for (const auto &element : elements_) {
        log::write(element->make_log_record(log::LogLevel::Debug, "pipeline", "starting element"));
        element->set_stop_token(stop_token_);
        element->set_progress_sink(progress_sink_.get());
        element->reset_progress_reporting();
        element->start();
        emit(PipelineEvent{
            .kind = PipelineEventKind::ElementStarted,
            .element_name = element->name(),
            .message = "started",
        });
        ++started_count_;
    }
}

std::optional<Buffer> Pipeline::run_sweep() {
    return execute(elements_, /*seed_from_cache=*/false);
}

std::optional<Buffer> Pipeline::rerun_from(const std::shared_ptr<Element> &element) {
    if (!caching_enabled_) {
        throw std::invalid_argument("rerun_from requires caching to be enabled");
    }
    if (!find_element_index(elements_, element)) {
        throw std::invalid_argument("rerun_from element is not in the pipeline");
    }

    return execute(replay_set(element), /*seed_from_cache=*/true);
}

void Pipeline::stop_all() {
    for (std::size_t index = started_count_; index > 0; --index) {
        log::write(elements_[index - 1]->make_log_record(log::LogLevel::Debug, "pipeline", "stopping element"));
        elements_[index - 1]->stop();
        emit(PipelineEvent{
            .kind = PipelineEventKind::ElementStopped,
            .element_name = elements_[index - 1]->name(),
            .message = "stopped",
        });
    }
    started_count_ = 0;
    // Stop clears the run's vector-clock counters so the next start begins at 0.
    emit_counts_.clear();
    emit(PipelineEvent{
        .kind = PipelineEventKind::Stopped,
        .message = "stopped",
    });
}

std::optional<Buffer> Pipeline::run() {
    // One run() = a pump loop (live phase). Offline (no live source) it is one
    // sweep, identical to the historical one-shot. A live source streams: the pump
    // sweeps until every live source is at end-of-stream. "Live" is detected from
    // the graph (a declared live source), never a separate run method. See
    // docs/design/dataflow_sync_model.md S11.8.

    // Live + Queue: run threaded segments (step 4b). run_threaded manages its own
    // start_all/stop_all and asks boundary elements to own the handoff runtime.
    if (should_run_threaded()) {
        return run_threaded(stop_token_, {}, nullptr, runtime_telemetry_enabled_);
    }

    start_all();

    std::optional<Buffer> current;
    try {
        if (has_live_source()) {
            while (!all_live_sources_at_eos() && !stop_requested()) {
                current = run_sweep();
            }
        } else {
            current = run_sweep();
        }
    } catch (const std::exception &error) {
        emit(PipelineEvent{
            .kind = PipelineEventKind::Error,
            .message = error.what(),
        });
        log::LogRecord error_record{
            .level = log::LogLevel::Error,
            .component = "pipeline",
            .message = "pipeline failed",
            .fields =
                {
                    {"error", error.what()},
                },
        };
        log::write(std::move(error_record));
        stop_started(started_count_);
        started_count_ = 0;
        emit(PipelineEvent{
            .kind = PipelineEventKind::Stopped,
            .message = "failed",
        });
        throw;
    } catch (...) {
        emit(PipelineEvent{
            .kind = PipelineEventKind::Error,
            .message = "unknown error",
        });
        log::LogRecord error_record{
            .level = log::LogLevel::Error,
            .component = "pipeline",
            .message = "pipeline failed with unknown error",
        };
        log::write(std::move(error_record));
        stop_started(started_count_);
        started_count_ = 0;
        emit(PipelineEvent{
            .kind = PipelineEventKind::Stopped,
            .message = "failed",
        });
        throw;
    }

    log::LogRecord finish_record{
        .level = log::LogLevel::Debug,
        .component = "pipeline",
        .message = "finished pipeline",
        .fields =
            {
                {"has_output", current ? "true" : "false"},
            },
    };
    log::write(std::move(finish_record));
    stop_all();

    return current;
}

std::optional<Buffer> Pipeline::execute_segment(const std::vector<std::shared_ptr<Element>> &order,
                                                std::map<const Element *, std::map<std::string, Buffer>> live,
                                                const BoundaryRuntimes &boundaries, std::stop_token stop,
                                                std::mutex &shared) {
    std::optional<Buffer> last_terminal_output;
    std::set<std::string> logged_routes;

    for (const auto &element : order) {
        if (stop.stop_requested()) {
            return std::nullopt;
        }
        const auto incoming = incoming_links_for(element);
        const auto outgoing = outgoing_links_for(element);
        if (incoming.empty() && outgoing.empty()) {
            continue;
        }

        // Gather inputs. Within a segment every incoming link's source is either an
        // earlier segment element (routed into `live`) or a boundary element
        // (seeded into `live` under the boundary's identity by the consumer loop).
        // See S10.
        ElementInputs inputs;
        for (const auto &link : incoming) {
            const auto source_it = live.find(link.source_element.get());
            if (source_it == live.end()) {
                throw std::invalid_argument("segment element is missing an upstream/boundary input");
            }
            const auto pad_it = source_it->second.find(link.source_pad_name);
            if (pad_it == source_it->second.end()) {
                throw std::invalid_argument("segment element is missing an upstream/boundary input pad");
            }
            inputs.emplace(link.sink_pad_name, pad_it->second);
        }
        if (inputs.size() != incoming.size()) {
            throw std::invalid_argument("segment element did not receive all connected inputs");
        }

        std::vector<const std::vector<std::uint32_t> *> input_clocks;
        input_clocks.reserve(inputs.size());
        for (const auto &entry : inputs) {
            if (entry.second) {
                input_clocks.push_back(&entry.second->provenance());
            }
        }
        const auto merged_provenance = merge_provenance(input_clocks, next_slot_);

        ElementOutputs outputs;
        try {
            outputs = timed_process_pads(*element, std::move(inputs));
        } catch (const std::exception &error) {
            throw std::runtime_error("element '" + element->name() + "' failed " + input_pad_context(incoming) +
                                     ": " + error.what());
        }
        if (stop.stop_requested()) {
            return std::nullopt;
        }

        // Stamp provenance: merged inputs plus this element's own slot, incremented
        // once per firing. The own-count bump touches shared state, so it is locked.
        const auto base_it = element_base_.find(element.get());
        std::optional<std::uint32_t> own_count;
        {
            const std::lock_guard<std::mutex> lock(shared);
            if (base_it != element_base_.end() && !outputs.empty()) {
                own_count = next_emit_count(element.get());
            }
        }
        for (auto &entry : outputs) {
            auto provenance = merged_provenance;
            if (own_count) {
                provenance[base_it->second] = *own_count;
            }
            entry.second.set_provenance(std::move(provenance));
        }

        if (outgoing.empty()) {
            last_terminal_output = outputs.empty()
                                       ? std::nullopt
                                       : std::optional<Buffer>(std::move(outputs.begin()->second));
            continue;
        }

        // Route each output: into `live` for an intra-segment link, or into a
        // boundary runtime for a thread-boundary sink. Boundary pushes happen AFTER
        // the shared-locked emit so a parking Block push never stalls other
        // segments' provenance/observer work.
        struct QueuedPush {
            ThreadBoundaryRuntime *boundary = nullptr;
            Buffer buffer;
        };
        std::vector<QueuedPush> pushes;
        for (const auto &link : outgoing) {
            const auto output_it = outputs.find(link.source_pad_name);
            if (output_it == outputs.end()) {
                if (const auto source_it = live.find(link.source_element.get()); source_it != live.end()) {
                    source_it->second.erase(link.source_pad_name);
                }
                continue;
            }
            Buffer routed = output_it->second;
            apply_output_metadata_annotations(output_metadata_annotations_, link, routed);
            {
                const std::lock_guard<std::mutex> lock(shared);
                log_first_buffer_on_route(link, routed, logged_routes);
                emit(PipelineEvent{
                    .kind = PipelineEventKind::BufferObserved,
                    .buffer = buffer_observation_for(link, routed, next_buffer_sequence_++),
                });
            }
            if (link.sink_element->is_thread_boundary()) {
                const auto boundary_it = boundaries.find(link.sink_element.get());
                if (boundary_it == boundaries.end() || boundary_it->second == nullptr) {
                    throw std::invalid_argument("thread boundary element '" + link.sink_element->name() +
                                                "' has no prepared runtime");
                }
                pushes.push_back(QueuedPush{
                    .boundary = boundary_it->second,
                    .buffer = std::move(routed),
                });
            } else {
                live[link.source_element.get()].insert_or_assign(link.source_pad_name, std::move(routed));
            }
        }

        for (auto &push : pushes) {
            if (!push.boundary->boundary_push(std::move(push.buffer), stop)) {
                // Stop requested or queue closed: abandon this sweep cooperatively.
                return last_terminal_output;
            }
        }
    }

    return last_terminal_output;
}

bool Pipeline::route_runtime_push(Element &element, std::string_view source_pad, Buffer buffer,
                                  const BoundaryRuntimes &boundaries, std::stop_token stop,
                                  std::mutex &shared, std::recursive_mutex &runtime_mutex,
                                  const SegmentSafePoint &safe_point,
                                  std::set<std::string> &logged_routes,
                                  RuntimePassiveStates &passive_states) {
    bool routed_any = false;
    for (const auto &link : links_) {
        if (link.source_element.get() != &element || link.source_pad_name != source_pad) {
            continue;
        }
        routed_any = true;

        Buffer routed = buffer;
        apply_output_metadata_annotations(output_metadata_annotations_, link, routed);
        {
            const std::lock_guard<std::mutex> lock(shared);
            log_first_buffer_on_route(link, routed, logged_routes);
            emit(PipelineEvent{
                .kind = PipelineEventKind::BufferObserved,
                .buffer = buffer_observation_for(link, routed, next_buffer_sequence_++),
            });
        }

        if (!link.sink_element->is_thread_boundary()) {
            if (!route_runtime_passive_input(link, std::move(routed), boundaries, stop, shared, runtime_mutex,
                                             safe_point, logged_routes, passive_states)) {
                return false;
            }
            continue;
        }

        if (safe_point) {
            safe_point({link.sink_element}, stop);
            if (stop.stop_requested()) {
                return false;
            }
        }
        const auto boundary_it = boundaries.find(link.sink_element.get());
        if (boundary_it == boundaries.end() || boundary_it->second == nullptr) {
            throw std::invalid_argument("thread boundary element '" + link.sink_element->name() +
                                        "' has no prepared runtime");
        }
        if (!boundary_it->second->boundary_push(std::move(routed), stop)) {
            return false;
        }
    }

    if (!routed_any) {
        throw std::invalid_argument("RuntimeContext push from '" + element.name() + "." + std::string(source_pad) +
                                    "' has no linked downstream pad");
    }
    return true;
}

bool Pipeline::route_runtime_passive_input(const PadLink &link, Buffer buffer,
                                           const BoundaryRuntimes &boundaries, std::stop_token stop,
                                           std::mutex &shared, std::recursive_mutex &runtime_mutex,
                                           const SegmentSafePoint &safe_point,
                                           std::set<std::string> &logged_routes,
                                           RuntimePassiveStates &passive_states) {
    auto &element = *link.sink_element;
    std::unique_lock<std::recursive_mutex> runtime_lock(runtime_mutex);
    auto &state = passive_states[&element];
    state.queues[link.sink_pad_name].push_back(std::move(buffer));

    enum class Mode { Driving, Held, Latest };

    const auto incoming = incoming_links_for(link.sink_element);
    if (incoming.empty()) {
        return true;
    }

    const auto mode_for = [this, &incoming, &element](std::size_t index) {
        std::string policy = "barrier";
        if (const auto value = element.property_as<std::string>("policy")) {
            policy = *value;
        }

        std::size_t primary = 0;
        for (std::size_t i = 1; i < incoming.size(); ++i) {
            if (incoming[i].sink_pad_name < incoming[primary].sink_pad_name) {
                primary = i;
            }
        }
        if (policy == "latest") {
            return index == primary ? Mode::Driving : Mode::Latest;
        }
        if (policy == "held") {
            return index == primary ? Mode::Driving : Mode::Held;
        }
        return is_live_driven(incoming[index].source_element) ? Mode::Driving : Mode::Held;
    };

    std::vector<Mode> modes;
    modes.reserve(incoming.size());
    for (std::size_t i = 0; i < incoming.size(); ++i) {
        modes.push_back(mode_for(i));
    }

    while (!stop.stop_requested()) {
        bool complete = true;
        for (std::size_t i = 0; i < incoming.size(); ++i) {
            auto &queue = state.queues[incoming[i].sink_pad_name];
            if (modes[i] == Mode::Latest && queue.size() > 1) {
                auto newest = std::move(queue.back());
                queue.clear();
                queue.push_back(std::move(newest));
            }
            if (queue.empty()) {
                complete = false;
                break;
            }
        }
        if (!complete) {
            return true;
        }

        std::vector<std::uint32_t> max_slot(next_slot_, 0u);
        std::vector<int> shared_count(next_slot_, 0);
        std::size_t driving_count = 0;
        for (std::size_t i = 0; i < incoming.size(); ++i) {
            if (modes[i] != Mode::Driving) {
                continue;
            }
            ++driving_count;
            const auto &clock = state.queues[incoming[i].sink_pad_name].front().provenance();
            for (std::size_t slot = 0; slot < clock.size() && slot < max_slot.size(); ++slot) {
                if (clock[slot] != 0u) {
                    shared_count[slot] += 1;
                    max_slot[slot] = std::max(max_slot[slot], clock[slot]);
                }
            }
        }

        std::vector<std::size_t> behind;
        for (std::size_t i = 0; i < incoming.size(); ++i) {
            if (modes[i] != Mode::Driving) {
                continue;
            }
            const auto &clock = state.queues[incoming[i].sink_pad_name].front().provenance();
            for (std::size_t slot = 0; slot < clock.size() && slot < max_slot.size(); ++slot) {
                if (clock[slot] != 0u && shared_count[slot] >= 2 && clock[slot] < max_slot[slot]) {
                    behind.push_back(i);
                    break;
                }
            }
        }
        if (!behind.empty()) {
            if (behind.size() == driving_count) {
                throw std::invalid_argument("element '" + element.name() +
                                            "': inputs have no matchable generation");
            }
            for (const auto index : behind) {
                state.queues[incoming[index].sink_pad_name].pop_front();
            }
            continue;
        }

        ElementInputs inputs;
        std::vector<const std::vector<std::uint32_t> *> input_clocks;
        input_clocks.reserve(incoming.size());
        for (std::size_t i = 0; i < incoming.size(); ++i) {
            const auto &queue = state.queues[incoming[i].sink_pad_name];
            const Buffer &selected = modes[i] == Mode::Latest ? queue.back() : queue.front();
            inputs.emplace(incoming[i].sink_pad_name, selected);
            input_clocks.push_back(&selected.provenance());
        }
        const auto merged_provenance = merge_provenance(input_clocks, next_slot_);

        if (safe_point) {
            safe_point({link.sink_element}, stop);
            if (stop.stop_requested()) {
                return false;
            }
        }

        ElementOutputs outputs;
        try {
            outputs = timed_process_pads(element, std::move(inputs));
        } catch (const std::exception &error) {
            throw std::runtime_error("element '" + element.name() + "' failed " + input_pad_context(incoming) +
                                     ": " + error.what());
        }
        if (stop.stop_requested()) {
            return false;
        }

        const auto base_it = element_base_.find(&element);
        std::optional<std::uint32_t> own_count;
        {
            const std::lock_guard<std::mutex> lock(shared);
            if (base_it != element_base_.end() && !outputs.empty()) {
                own_count = next_emit_count(&element);
            }
        }
        for (auto &entry : outputs) {
            auto provenance = merged_provenance;
            if (own_count) {
                provenance[base_it->second] = *own_count;
            }
            entry.second.set_provenance(std::move(provenance));
        }

        for (std::size_t i = 0; i < incoming.size(); ++i) {
            auto &queue = state.queues[incoming[i].sink_pad_name];
            if (modes[i] == Mode::Driving) {
                queue.pop_front();
            } else if (modes[i] == Mode::Latest && queue.size() > 1) {
                auto newest = std::move(queue.back());
                queue.clear();
                queue.push_back(std::move(newest));
            }
        }

        runtime_lock.unlock();
        if (!route_runtime_outputs(element, std::move(outputs), boundaries, stop, shared, runtime_mutex, safe_point,
                                   logged_routes, passive_states)) {
            return false;
        }
        runtime_lock.lock();
    }
    return false;
}

bool Pipeline::route_runtime_outputs(Element &element, ElementOutputs outputs,
                                     const BoundaryRuntimes &boundaries, std::stop_token stop,
                                     std::mutex &shared, std::recursive_mutex &runtime_mutex,
                                     const SegmentSafePoint &safe_point,
                                     std::set<std::string> &logged_routes,
                                     RuntimePassiveStates &passive_states) {
    if (outputs.empty()) {
        return true;
    }

    for (const auto &link : links_) {
        if (link.source_element.get() != &element) {
            continue;
        }
        const auto output_it = outputs.find(link.source_pad_name);
        if (output_it == outputs.end()) {
            continue;
        }
        if (!route_runtime_push(element, link.source_pad_name, output_it->second, boundaries, stop, shared,
                                runtime_mutex, safe_point, logged_routes, passive_states)) {
            return false;
        }
    }
    return true;
}

void Pipeline::route_runtime_eos(Element &element, std::string_view source_pad,
                                 const BoundaryRuntimes &boundaries, std::recursive_mutex &runtime_mutex,
                                 const SegmentSafePoint &safe_point, RuntimePassiveStates &passive_states) {
    for (const auto &link : links_) {
        if (link.source_element.get() != &element || link.source_pad_name != source_pad) {
            continue;
        }
        if (link.sink_element->is_thread_boundary()) {
            if (const auto boundary_it = boundaries.find(link.sink_element.get());
                boundary_it != boundaries.end() && boundary_it->second != nullptr) {
                boundary_it->second->boundary_close();
            }
            continue;
        }
        route_runtime_passive_eos(link, boundaries, runtime_mutex, safe_point, passive_states);
    }
}

void Pipeline::route_runtime_passive_eos(const PadLink &link, const BoundaryRuntimes &boundaries,
                                         std::recursive_mutex &runtime_mutex, const SegmentSafePoint &safe_point,
                                         RuntimePassiveStates &passive_states) {
    std::vector<std::string> output_pads;
    {
        const std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
        auto &state = passive_states[link.sink_element.get()];
        state.eos_pads.insert(link.sink_pad_name);

        const auto incoming = incoming_links_for(link.sink_element);
        const bool all_eos = std::all_of(incoming.begin(), incoming.end(), [&state](const PadLink &incoming_link) {
            return state.eos_pads.contains(incoming_link.sink_pad_name);
        });
        if (!all_eos) {
            return;
        }
        for (const auto &pad : link.sink_element->output_pads()) {
            output_pads.push_back(pad.name());
        }
        passive_states.erase(link.sink_element.get());
    }

    for (const auto &pad_name : output_pads) {
        route_runtime_eos(*link.sink_element, pad_name, boundaries, runtime_mutex, safe_point, passive_states);
    }
}

std::optional<Buffer> Pipeline::run_source_segment(const PipelineSegment &segment, const BoundaryRuntimes &boundaries,
                                                   std::stop_token stop, std::mutex &shared,
                                                   const SegmentSafePoint &safe_point) {
    const bool has_live = std::any_of(segment.elements.begin(), segment.elements.end(),
                                      [](const auto &element) { return element->is_live(); });
    const auto sources_at_eos = [&segment]() {
        for (const auto &element : segment.elements) {
            if (element->is_live() && !element->at_end_of_stream()) {
                return false;
            }
        }
        return true;
    };

    std::optional<Buffer> last;
    if (has_live) {
        // Pump one buffer per sweep until this segment's live sources reach EOS.
        // Between buffers, apply pending property changes (forward-apply, S11.5).
        while (!stop.stop_requested() && !sources_at_eos()) {
            if (safe_point) {
                safe_point(segment.elements, stop);
            }
            last = execute_segment(segment.elements, {}, boundaries, stop, shared);
        }
    } else if (!stop.stop_requested()) {
        // One-run source behind a boundary: a single sweep, like the offline model.
        last = execute_segment(segment.elements, {}, boundaries, stop, shared);
    }

    // Signal EOS to downstream segments so they drain then finish.
    for (const auto &queue : segment.output_queues) {
        if (const auto it = boundaries.find(queue.get()); it != boundaries.end() && it->second != nullptr) {
            it->second->boundary_close();
        }
    }
    return last;
}

std::optional<Buffer> Pipeline::run_consumer_segment(const PipelineSegment &segment,
                                                     const BoundaryRuntimes &boundaries,
                                                     std::stop_token stop, std::mutex &shared,
                                                     const SegmentSafePoint &safe_point) {
    // The aggregator (S11.1-2). One held "head" per input queue. Each fire it
    // Barrier-matches the heads by vector clock (default policy): heads sharing a
    // clock slot must agree; a head strictly behind on a shared slot is an orphan
    // (its partner was dropped) and is discarded to realign; a Held input (a static
    // / non-live-driven branch) is a permanent wildcard, reused across many driving
    // fires and not consumed. A single input queue degenerates to a plain pull.
    // Per-input aggregation mode (S11.1): Driving (consume; Barrier-realign by clock),
    // Held (a static input reused as a wildcard), or Latest (sample the newest, drop
    // intermediates). Chosen from the join element's `policy` property below.
    enum class Mode { Driving, Held, Latest };
    struct Head {
        ThreadBoundaryRuntime *boundary = nullptr;
        const Element *boundary_element = nullptr;
        const PadLink *out_link = nullptr; // boundary -> consumer, for observation + seeding
        std::string source_pad;
        std::string sink_pad;   // the join input pad this boundary feeds
        bool auto_held = false; // not live-driven: Held under the default policy
        Mode mode = Mode::Driving;
        std::optional<Buffer> buffer;
        bool eos = false;
    };
    const auto retained = [](Mode mode) { return mode != Mode::Driving; };

    std::vector<Head> heads;
    heads.reserve(segment.input_queues.size());
    for (const auto &queue : segment.input_queues) {
        Head head;
        const auto it = boundaries.find(queue.get());
        head.boundary = (it != boundaries.end()) ? it->second : nullptr;
        head.boundary_element = queue.get();
        head.source_pad = queue->output_pads().empty() ? std::string() : queue->output_pads().front().name();
        for (const auto &link : links_) {
            if (link.source_element.get() == queue.get()) {
                head.out_link = &link;
                head.sink_pad = link.sink_pad_name;
            }
            // The producing branch is Held iff it is not live-driven: a static /
            // one-run input reused across the live driver's buffers (S11.3).
            if (link.sink_element.get() == queue.get()) {
                head.auto_held = !is_live_driven(link.source_element);
            }
        }
        heads.push_back(std::move(head));
    }

    // Select the join policy from the element all input queues feed (Sync exposes a
    // `policy` property, S11.4). Default Barrier. The primary input is the lowest-
    // named join pad (in_0 / in0); under held/latest the rest are reused/sampled.
    const Element *join = nullptr;
    bool single_join = !heads.empty();
    for (const auto &head : heads) {
        const Element *sink = head.out_link != nullptr ? head.out_link->sink_element.get() : nullptr;
        if (join == nullptr) {
            join = sink;
        } else if (join != sink) {
            single_join = false;
        }
    }
    std::string policy = "barrier";
    if (single_join && join != nullptr) {
        if (const auto value = join->property_as<std::string>("policy")) {
            policy = *value;
        }
    }
    std::size_t primary = 0;
    for (std::size_t i = 1; i < heads.size(); ++i) {
        if (heads[i].sink_pad < heads[primary].sink_pad) {
            primary = i;
        }
    }
    for (std::size_t i = 0; i < heads.size(); ++i) {
        if (policy == "latest") {
            heads[i].mode = (i == primary) ? Mode::Driving : Mode::Latest;
        } else if (policy == "held") {
            heads[i].mode = (i == primary) ? Mode::Driving : Mode::Held;
        } else { // barrier / zip / all-required-once: auto-Held a static input, else Driving
            heads[i].mode = heads[i].auto_held ? Mode::Held : Mode::Driving;
        }
    }

    std::optional<Buffer> last;
    while (!stop.stop_requested()) {
        // Between fires, park while paused and apply pending property changes to
        // this segment's elements (forward-apply on the segment thread, S11.5).
        if (safe_point) {
            safe_point(segment.elements, stop);
        }

        // 1a. Fill every empty head, blocking for a Driving input's next buffer (and
        // for a Latest input's very first sample). Held/Latest inputs do not end the
        // segment on EOS; a Driving input that ends, does.
        bool finished = false;
        for (auto &head : heads) {
            if (head.buffer) {
                continue;
            }
            if (head.boundary == nullptr) {
                finished = true;
                break;
            }
            if (head.eos) {
                if (!retained(head.mode)) { // a driving input ended -> the segment ends
                    finished = true;
                    break;
                }
                continue; // retained + EOS without a buffer: handled by completeness below
            }
            auto pull = head.boundary->boundary_pull(stop);
            if (pull.buffer) {
                if (head.out_link != nullptr) {
                    const std::lock_guard<std::mutex> lock(shared);
                    emit(PipelineEvent{
                        .kind = PipelineEventKind::BufferObserved,
                        .buffer = buffer_observation_for(*head.out_link, *pull.buffer, next_buffer_sequence_++),
                    });
                }
                head.buffer = std::move(pull.buffer);
            } else if (pull.end_of_stream) {
                head.eos = true;
                if (!retained(head.mode)) {
                    finished = true;
                    break;
                }
            } else {
                finished = true; // NoData -> cooperative stop
                break;
            }
        }
        if (finished) {
            break;
        }

        // 1b. Now that the Driving inputs are present, refresh Latest heads to the
        // newest buffer available at fire time (sample-and-hold, dropping
        // intermediates); never blocks.
        for (auto &head : heads) {
            if (head.mode != Mode::Latest || head.boundary == nullptr) {
                continue;
            }
            while (true) {
                auto sample = head.boundary->boundary_try_pull();
                if (sample.buffer) {
                    if (head.out_link != nullptr) {
                        const std::lock_guard<std::mutex> lock(shared);
                        emit(PipelineEvent{
                            .kind = PipelineEventKind::BufferObserved,
                            .buffer = buffer_observation_for(*head.out_link, *sample.buffer, next_buffer_sequence_++),
                        });
                    }
                    head.buffer = std::move(sample.buffer);
                    continue; // keep draining to the newest
                }
                if (sample.end_of_stream) {
                    head.eos = true;
                }
                break;
            }
        }

        // 2. Every input must present a buffer to fire (a Held input that never
        // delivered cannot complete the set).
        const bool complete = std::all_of(heads.begin(), heads.end(),
                                          [](const Head &head) { return head.buffer.has_value(); });
        if (!complete) {
            break;
        }

        // 3. Barrier realign over the Driving inputs only (Held/Latest are wildcards).
        // For every clock slot non-zero in >=2 driving heads (a shared generation
        // slot), heads below the max are orphans -- monotonicity says their partner is
        // gone -- so discard them and refill.
        std::vector<std::uint32_t> max_slot(next_slot_, 0u);
        std::vector<int> shared_count(next_slot_, 0);
        std::size_t driving_count = 0;
        for (const auto &head : heads) {
            if (retained(head.mode)) {
                continue;
            }
            ++driving_count;
            const auto &clock = head.buffer->provenance();
            for (std::size_t s = 0; s < clock.size() && s < max_slot.size(); ++s) {
                if (clock[s] != 0u) {
                    shared_count[s] += 1;
                    max_slot[s] = std::max(max_slot[s], clock[s]);
                }
            }
        }
        std::vector<std::size_t> behind;
        for (std::size_t i = 0; i < heads.size(); ++i) {
            if (retained(heads[i].mode)) {
                continue;
            }
            const auto &clock = heads[i].buffer->provenance();
            for (std::size_t s = 0; s < clock.size() && s < max_slot.size(); ++s) {
                if (clock[s] != 0u && shared_count[s] >= 2 && clock[s] < max_slot[s]) {
                    behind.push_back(i);
                    break;
                }
            }
        }
        if (!behind.empty()) {
            if (behind.size() == driving_count) {
                // No driving head is the max on every conflicting slot: the inputs are
                // mutually inconsistent, not a recoverable drop. Surface it.
                throw std::invalid_argument("aggregator: inputs have no matchable generation");
            }
            for (const auto index : behind) {
                heads[index].buffer.reset(); // discard orphan; refill next iteration
            }
            continue;
        }

        // 4. Matched set -> fire. Seed each head under its boundary identity.
        std::map<const Element *, std::map<std::string, Buffer>> live;
        for (const auto &head : heads) {
            live[head.boundary_element].insert_or_assign(head.source_pad, *head.buffer);
        }
        last = execute_segment(segment.elements, std::move(live), boundaries, stop, shared);

        // 5. Consume driving heads; retain Held/Latest heads for reuse.
        for (auto &head : heads) {
            if (!retained(head.mode)) {
                head.buffer.reset();
            }
        }
    }

    for (const auto &queue : segment.output_queues) {
        if (const auto it = boundaries.find(queue.get()); it != boundaries.end() && it->second != nullptr) {
            it->second->boundary_close();
        }
    }
    return last;
}

std::optional<Buffer> Pipeline::run_threaded(std::stop_token stop, SegmentSafePoint safe_point,
                                             std::mutex *runtime_property_mutex, bool telemetry_enabled) {
    set_runtime_telemetry_enabled(telemetry_enabled);

    // Unify the external stop with an internal one so a segment failure can stop
    // peers (otherwise a consumer could block forever on a queue the failed
    // producer never closed). Elements and queue waits observe this unified token.
    std::stop_source run_stop;
    std::stop_callback external_cb(stop, [&run_stop]() { run_stop.request_stop(); });
    const auto token = run_stop.get_token();
    set_stop_token(token);

    const auto segments = decompose_into_segments(*this);

    std::mutex shared;
    std::mutex fallback_runtime_property_mutex;
    auto &runtime_property_lock =
        runtime_property_mutex == nullptr ? fallback_runtime_property_mutex : *runtime_property_mutex;

    BoundaryRuntimes boundaries;
    std::map<std::pair<const Element *, std::string>, std::string> published_runtime_properties;
    std::map<std::pair<const Element *, std::string>, std::string> published_runtime_telemetry;
    std::jthread runtime_property_publisher;
    const auto publish_boundary_runtime_properties = [this, &boundaries, &shared,
                                                         &published_runtime_properties,
                                                         &published_runtime_telemetry,
                                                         telemetry_enabled](bool force) noexcept {
        if (!observer_ || (!telemetry_enabled && !profiling_enabled_)) {
            return;
        }

        try {
            std::vector<PipelineEvent> events;
            for (const auto &[element, boundary] : boundaries) {
                if (!telemetry_enabled) {
                    break;
                }
                if (boundary == nullptr) {
                    continue;
                }

                std::vector<PipelinePropertySnapshot> properties;
                try {
                    properties = boundary->boundary_runtime_properties();
                } catch (...) {
                    continue;
                }

                for (const auto &property : properties) {
                    const auto key = std::make_pair(element, property.name);
                    const auto found = published_runtime_properties.find(key);
                    const auto previous = found == published_runtime_properties.end() ? property.value : found->second;
                    if (!force && found != published_runtime_properties.end() && found->second == property.value) {
                        continue;
                    }
                    published_runtime_properties[key] = property.value;
                    events.push_back(PipelineEvent{
                        .kind = PipelineEventKind::PropertyChanged,
                        .property_change =
                            PipelinePropertyChangeObservation{
                                .element = endpoint_snapshot(*element, std::string_view{}),
                                .property_name = property.name,
                                .value_type = property.value_type,
                                .previous_value = previous,
                                .new_value = property.value,
                                .effect = property.effect,
                            },
                        });
                }

                std::vector<TelemetrySnapshot> telemetry;
                try {
                    telemetry = boundary->boundary_runtime_telemetry();
                } catch (...) {
                    continue;
                }

                for (const auto &field : telemetry) {
                    const auto snapshot = telemetry_snapshot_for(field);
                    const auto key = std::make_pair(element, snapshot.name);
                    const auto found = published_runtime_telemetry.find(key);
                    const auto previous = found == published_runtime_telemetry.end() ? snapshot.value : found->second;
                    if (!force && found != published_runtime_telemetry.end() && found->second == snapshot.value) {
                        continue;
                    }
                    published_runtime_telemetry[key] = snapshot.value;
                    events.push_back(PipelineEvent{
                        .kind = PipelineEventKind::TelemetryChanged,
                        .telemetry_change =
                            PipelineTelemetryChangeObservation{
                                .element = endpoint_snapshot(*element, std::string_view{}),
                                .telemetry_name = snapshot.name,
                                .value_type = snapshot.value_type,
                                .previous_value = previous,
                                .new_value = snapshot.value,
                                .unit = snapshot.unit,
                                .description = snapshot.description,
                                .value_hint = snapshot.value_hint,
                                .kind = snapshot.kind,
                            },
                    });
                }
            }

            // Per-element timing overlay: publish every element's Duration channels
            // (the built-in "process" timer plus op scopes) so the --graph telemetry
            // panel shows live per-element timing, not just Queue size telemetry.
            if (profiling_enabled_) {
                for (const auto &element : elements_) {
                    std::vector<TelemetrySnapshot> telemetry;
                    try {
                        telemetry = element->telemetry_snapshot();
                    } catch (...) {
                        continue;
                    }
                    for (const auto &field : telemetry) {
                        if (field.kind != TelemetryKind::Duration) {
                            continue;
                        }
                        const auto snapshot = telemetry_snapshot_for(field);
                        const auto key = std::make_pair(element.get(), snapshot.name);
                        const auto found = published_runtime_telemetry.find(key);
                        const auto previous =
                            found == published_runtime_telemetry.end() ? snapshot.value : found->second;
                        if (!force && found != published_runtime_telemetry.end() &&
                            found->second == snapshot.value) {
                            continue;
                        }
                        published_runtime_telemetry[key] = snapshot.value;
                        events.push_back(PipelineEvent{
                            .kind = PipelineEventKind::TelemetryChanged,
                            .telemetry_change =
                                PipelineTelemetryChangeObservation{
                                    .element = endpoint_snapshot(*element, std::string_view{}),
                                    .telemetry_name = snapshot.name,
                                    .value_type = snapshot.value_type,
                                    .previous_value = previous,
                                    .new_value = snapshot.value,
                                    .unit = snapshot.unit,
                                    .description = snapshot.description,
                                    .value_hint = snapshot.value_hint,
                                    .kind = snapshot.kind,
                                },
                        });
                    }
                }
            }

            if (events.empty()) {
                return;
            }
            const std::lock_guard<std::mutex> lock(shared);
            for (auto &event : events) {
                emit(std::move(event));
            }
        } catch (...) {
        }
    };
    const auto start_runtime_property_publisher = [&]() {
        if (!observer_ || (boundaries.empty() && !profiling_enabled_)) {
            return;
        }
        runtime_property_publisher = std::jthread([&publish_boundary_runtime_properties](std::stop_token publisher_stop) {
            static constexpr auto interval = std::chrono::milliseconds(33);
            while (!publisher_stop.stop_requested()) {
                std::this_thread::sleep_for(interval);
                if (!publisher_stop.stop_requested()) {
                    publish_boundary_runtime_properties(/*force=*/false);
                }
            }
        });
    };
    const auto stop_runtime_property_publisher = [&]() noexcept {
        if (runtime_property_publisher.joinable()) {
            runtime_property_publisher.request_stop();
            runtime_property_publisher.join();
        }
    };
    std::vector<ActiveElement *> active_elements;
    const auto clear_boundaries = [&]() noexcept {
        stop_runtime_property_publisher();
        for (const auto &[element, boundary] : boundaries) {
            (void)element;
            if (boundary != nullptr) {
                boundary->clear_thread_boundary_runtime();
            }
        }
        publish_boundary_runtime_properties(/*force=*/true);
        boundaries.clear();
        published_runtime_properties.clear();
        published_runtime_telemetry.clear();
    };
    const auto stop_active_elements = [&active_elements]() noexcept {
        for (auto it = active_elements.rbegin(); it != active_elements.rend(); ++it) {
            if (*it != nullptr) {
                (*it)->stop_active();
            }
        }
        active_elements.clear();
    };
    const auto wait_active_elements = [&active_elements]() {
        for (auto *active : active_elements) {
            if (active != nullptr) {
                active->wait_active();
            }
        }
    };

    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::vector<std::optional<Buffer>> terminal_outputs(segments.size());

    try {
        {
            const std::lock_guard<std::mutex> lock(runtime_property_lock);
            start_all();
        }
        boundaries = prepare_boundary_runtimes(&runtime_property_lock);
        if (telemetry_enabled) {
            publish_boundary_runtime_properties(/*force=*/true);
            start_runtime_property_publisher();
        }
        PipelineRuntimeContext runtime_context(*this, boundaries, token, shared, failure, failure_mutex, run_stop,
                                               safe_point);

        {
            std::vector<std::jthread> threads;
            threads.reserve(segments.size());
            for (std::size_t index = 0; index < segments.size(); ++index) {
                if (is_active_source_segment(segments[index]) || is_active_boundary_consumer_segment(segments[index])) {
                    continue;
                }
                threads.emplace_back([this, &segments, index, &boundaries, &shared, token, &terminal_outputs,
                                      &failure, &failure_mutex, &run_stop, &safe_point]() {
                    try {
                        const auto &segment = segments[index];
                        terminal_outputs[index] =
                            segment.input_queues.empty()
                                ? run_source_segment(segment, boundaries, token, shared, safe_point)
                                : run_consumer_segment(segment, boundaries, token, shared, safe_point);
                    } catch (...) {
                        {
                            const std::lock_guard<std::mutex> lock(failure_mutex);
                            if (!failure) {
                                failure = std::current_exception();
                            }
                        }
                        // Wake peers blocked on boundary waits so none hang.
                        run_stop.request_stop();
                    }
                });
            }

            std::set<ActiveElement *> active_started;
            const auto start_active_element = [&](ActiveElement *active) {
                if (active == nullptr || !active_started.insert(active).second) {
                    return;
                }
                active->start_active(runtime_context);
                active_elements.push_back(active);
            };

            for (const auto &segment : segments) {
                if (!is_active_source_segment(segment)) {
                    continue;
                }
                auto *active = dynamic_cast<ActiveElement *>(segment.elements.front().get());
                start_active_element(active);
            }

            for (const auto &segment : segments) {
                if (!is_active_boundary_consumer_segment(segment)) {
                    continue;
                }
                for (const auto &link : activation_input_links(segment)) {
                    auto *active = dynamic_cast<ActiveElement *>(link.source_element.get());
                    start_active_element(active);
                }
            }

            try {
                wait_active_elements();
            } catch (...) {
                run_stop.request_stop();
                throw;
            }
        } // jthreads join here

        stop_active_elements();
        stop_all();
        clear_boundaries();
    } catch (...) {
        run_stop.request_stop();
        stop_active_elements();
        clear_boundaries();
        if (started_count_ > 0) {
            stop_started(started_count_);
            started_count_ = 0;
            emit(PipelineEvent{
                .kind = PipelineEventKind::Stopped,
                .message = "failed",
            });
        }
        throw;
    }

    if (failure) {
        emit(PipelineEvent{.kind = PipelineEventKind::Error, .message = exception_message(failure)});
        std::rethrow_exception(failure);
    }

    std::optional<Buffer> result;
    for (auto &output : terminal_outputs) {
        if (output) {
            result = std::move(output);
        }
    }
    return result;
}

Pipeline::BoundaryRuntimes Pipeline::prepare_boundary_runtimes(std::mutex *property_mutex) {
    BoundaryRuntimes boundaries;
    try {
        for (const auto &element : elements_) {
            if (!element->is_thread_boundary()) {
                continue;
            }
            auto *boundary = dynamic_cast<ThreadBoundaryRuntime *>(element.get());
            if (boundary == nullptr) {
                throw std::invalid_argument("element '" + element->name() +
                                            "' is marked as thread boundary but does not implement "
                                            "ThreadBoundaryRuntime");
            }
            boundary->prepare_thread_boundary_runtime(property_mutex);
            boundaries.emplace(element.get(), boundary);
        }
    } catch (...) {
        for (const auto &[element, boundary] : boundaries) {
            (void)element;
            if (boundary != nullptr) {
                boundary->clear_thread_boundary_runtime();
            }
        }
        throw;
    }
    return boundaries;
}

bool Pipeline::is_active_source_segment(const PipelineSegment &segment) const {
    if (!segment.input_queues.empty() || segment.elements.size() != 1) {
        return false;
    }
    return dynamic_cast<ActiveElement *>(segment.elements.front().get()) != nullptr;
}

bool Pipeline::is_active_boundary_consumer_segment(const PipelineSegment &segment) const {
    if (segment.input_queues.empty() || segment.elements.empty()) {
        return false;
    }
    const auto input_links = activation_input_links(segment);
    if (input_links.empty()) {
        return false;
    }
    for (const auto &link : input_links) {
        if (dynamic_cast<ActiveElement *>(link.source_element.get()) == nullptr) {
            return false;
        }
    }
    for (const auto &element : segment.elements) {
        if (incoming_links_for(element).empty() && !outgoing_links_for(element).empty()) {
            return false;
        }
    }
    return true;
}

std::vector<PadLink> Pipeline::activation_input_links(const PipelineSegment &segment) const {
    std::vector<PadLink> input_links;
    if (segment.input_queues.empty()) {
        return input_links;
    }
    for (const auto &queue : segment.input_queues) {
        for (const auto &link : links_) {
            if (link.source_element.get() == queue.get()) {
                input_links.push_back(link);
            }
        }
    }
    return input_links;
}

void Pipeline::stop_started(std::size_t started_count) noexcept {
    for (std::size_t index = started_count; index > 0; --index) {
        try {
            log::write(elements_[index - 1]->make_log_record(log::LogLevel::Debug, "pipeline",
                                                             "stopping element after failure"));
            elements_[index - 1]->stop();
            emit(PipelineEvent{
                .kind = PipelineEventKind::ElementStopped,
                .element_name = elements_[index - 1]->name(),
                .message = "stopped after failure",
            });
        } catch (...) {
        }
    }
}

std::uint32_t Pipeline::next_emit_count(const Element *element) {
    auto &count = emit_counts_[element];
    // 0 stays reserved ("not downstream of this element"); wrap max -> 1.
    count = (count == std::numeric_limits<std::uint32_t>::max()) ? 1u : count + 1u;
    return count;
}

void Pipeline::set_stop_token(std::stop_token token) {
    stop_token_ = std::move(token);
    // Forward to elements already added so a token set after construction still
    // reaches blocking waits; start_all() re-forwards for elements added later.
    for (const auto &element : elements_) {
        element->set_stop_token(stop_token_);
    }
}

bool Pipeline::stop_requested() const { return stop_token_.stop_requested(); }

void Pipeline::set_pause_waiter(std::function<void(std::stop_token)> waiter) {
    pause_waiter_ = std::move(waiter);
    for (const auto &element : elements_) {
        element->set_pause_waiter(pause_waiter_);
    }
}

bool Pipeline::has_live_source() const {
    for (const auto &element : elements_) {
        if (element->is_live()) {
            return true;
        }
    }
    return false;
}

bool Pipeline::is_live_driven(const std::shared_ptr<Element> &element) const {
    // Walk backwards over incoming links from `element`; live-driven iff `element`
    // or any ancestor declares itself a live source (OR-propagation, S11.5).
    std::set<const Element *> visited;
    std::vector<const Element *> stack{element.get()};
    while (!stack.empty()) {
        const auto *current = stack.back();
        stack.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        if (current->is_live()) {
            return true;
        }
        for (const auto &link : links_) {
            if (link.sink_element.get() == current) {
                stack.push_back(link.source_element.get());
            }
        }
    }
    return false;
}

void Pipeline::refresh_live_driven_flags() {
    for (const auto &element : elements_) {
        element->set_live_driven(is_live_driven(element));
    }
}

bool Pipeline::all_live_sources_at_eos() const {
    for (const auto &element : elements_) {
        if (element->is_live() && !element->at_end_of_stream()) {
            return false;
        }
    }
    return true;
}

bool Pipeline::should_run_threaded() const {
    return has_live_source() && decompose_into_segments(*this).size() > 1;
}

void Pipeline::emit(PipelineEvent event) noexcept {
    if (!observer_) {
        return;
    }

    // Serialize dispatch: progress reports are pushed from the worker thread mid-process(), so
    // they must not race the buffer/telemetry events. The lock is the innermost in every path
    // (callers may already hold the segment `shared` mutex), so it cannot deadlock.
    try {
        const std::lock_guard<std::mutex> lock(*emit_mutex_);
        event.sequence = next_event_sequence_++;
        observer_->observe(event);
    } catch (...) {
    }
}

} // namespace leakflow
