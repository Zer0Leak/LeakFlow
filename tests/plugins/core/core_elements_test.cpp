#include "leakflow/plugins/core/core_elements.hpp"
#include "leakflow/plugins/core/descriptor_catalog.hpp"
#include "leakflow/log/logger.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

} // namespace

int main()
{
    namespace core = leakflow::plugins::core;

    const auto input_path = std::filesystem::temp_directory_path() / "leakflow_plugins_core_input.txt";
    const auto output_path = std::filesystem::temp_directory_path() / "leakflow_plugins_core_output.txt";
    {
        std::ofstream input_file(input_path, std::ios::binary | std::ios::trunc);
        input_file << "hello leakflow";
    }

    core::FileSrc file_source;
    file_source.set_property("path", input_path.string());
    auto file_buffer = file_source.process(std::nullopt);
    if (!expect(file_buffer.has_value(), "FileSrc did not produce a buffer")) {
        return 1;
    }
    if (!expect(file_buffer->has_payload(), "FileSrc did not attach a payload")) {
        return 1;
    }
    const auto bytes_payload = file_buffer->payload_as<core::BytesPayload>();
    if (!expect(bytes_payload != nullptr, "FileSrc payload type was wrong")) {
        return 1;
    }
    if (!expect(bytes_payload->bytes() == "hello leakflow", "FileSrc bytes were wrong")) {
        return 1;
    }

    core::FileSink file_sink;
    file_sink.set_property("path", output_path.string());
    auto file_sink_output = file_sink.process(file_buffer);
    if (!expect(file_sink.received(), "FileSink did not receive a buffer")) {
        return 1;
    }
    if (!expect(!file_sink_output.has_value(), "FileSink should consume the buffer")) {
        return 1;
    }
    if (!expect(file_sink.last_bytes() == "hello leakflow", "FileSink bytes were wrong")) {
        return 1;
    }
    {
        std::ifstream output_file(output_path, std::ios::binary);
        std::string output_bytes;
        std::getline(output_file, output_bytes, '\0');
        if (!expect(output_bytes == "hello leakflow", "FileSink output file bytes were wrong")) {
            return 1;
        }
    }

    core::FakeSrc source;
    source.set_property("caps_type", std::string("sca/test"));
    source.set_property("metadata_key", std::string("kind"));
    source.set_property("metadata_value", std::string("fake"));

    auto buffer = source.process(std::nullopt);
    if (!expect(buffer.has_value(), "FakeSrc did not produce a buffer")) {
        return 1;
    }
    if (!expect(buffer->caps().type() == "sca/test", "FakeSrc caps type was wrong")) {
        return 1;
    }
    if (!expect(buffer->metadata("kind") == "fake", "FakeSrc metadata was wrong")) {
        return 1;
    }

    core::Summary summary;
    auto summarized = summary.process(buffer);
    if (!expect(summarized.has_value(), "Summary did not forward the buffer")) {
        return 1;
    }
    if (!expect(summarized->has_metadata("summary"), "Summary metadata was not added")) {
        return 1;
    }
    if (!expect(summary.last_summary().find("caps=sca/test") != std::string::npos,
            "Summary text did not include caps")) {
        return 1;
    }
    if (!expect(summary.last_summary().find("payload=none") != std::string::npos,
            "Summary text did not include payload presence")) {
        return 1;
    }

    leakflow::log::LogConfig summary_disabled_config;
    summary_disabled_config.summaries_enabled = false;
    leakflow::log::configure(summary_disabled_config);

    core::Summary suppressed_summary;
    auto suppressed = suppressed_summary.process(buffer);
    if (!expect(suppressed.has_value(), "Suppressed Summary did not forward the buffer")) {
        return 1;
    }
    if (!expect(suppressed_summary.last_summary().empty(), "Suppressed Summary still rendered text")) {
        return 1;
    }
    if (!expect(!suppressed->has_metadata("summary"), "Suppressed Summary still stamped metadata")) {
        return 1;
    }

    core::Summary forced_summary;
    forced_summary.set_property("always_print", true);
    auto forced = forced_summary.process(buffer);
    if (!expect(forced.has_value(), "Forced Summary did not forward the buffer")) {
        return 1;
    }
    if (!expect(forced_summary.last_summary().find("caps=sca/test") != std::string::npos,
            "Summary(always_print=true) did not render while summaries were disabled")) {
        return 1;
    }
    leakflow::log::reset_for_tests();

    core::Queue queue;
    auto queued = queue.process(summarized);
    if (!expect(queued.has_value(), "Queue did not forward a queued buffer")) {
        return 1;
    }
    if (!expect(queue.depth() == 0, "Queue depth should be zero after synchronous forwarding")) {
        return 1;
    }

    core::FakeSink sink;
    auto sink_output = sink.process(queued);
    if (!expect(sink.received(), "FakeSink did not receive a buffer")) {
        return 1;
    }
    if (!expect(!sink_output.has_value(), "FakeSink should consume the buffer")) {
        return 1;
    }

    const auto descriptors = core::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "built-in descriptor catalog size was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].name == "leakflow_plugins_core", "core plugin descriptor name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 8, "core element descriptor count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].author == "Zer0Leak <edgard.lima@gmail.com>",
            "core plugin descriptor author was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].license == "Apache-2.0", "core plugin descriptor license was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].version == "0.10", "core plugin descriptor version was wrong")) {
        return 1;
    }
    const auto& tee_descriptor = descriptors[0].elements[5];
    if (!expect(tee_descriptor.type_name == "Tee", "core plugin Tee descriptor index changed")) {
        return 1;
    }
    if (!expect(tee_descriptor.output_pads.empty(), "Tee descriptor should not predeclare source pads")) {
        return 1;
    }
    if (!expect(tee_descriptor.pad_templates.size() == 2, "Tee descriptor pad template count changed")) {
        return 1;
    }
    if (!expect(tee_descriptor.pad_templates[0].name() == "sink",
            "Tee descriptor sink template name changed")) {
        return 1;
    }
    if (!expect(tee_descriptor.pad_templates[0].caps().type() == leakflow::any_caps_type,
            "Tee descriptor sink template caps should be ANY")) {
        return 1;
    }
    if (!expect(tee_descriptor.pad_templates[1].name() == "src_%u",
            "Tee descriptor source template name changed")) {
        return 1;
    }
    if (!expect(tee_descriptor.pad_templates[1].presence() == leakflow::PadPresence::OnRequest,
            "Tee descriptor source template should be on request")) {
        return 1;
    }
    if (!expect(tee_descriptor.pad_templates[1].caps().type() == leakflow::any_caps_type,
            "Tee descriptor source template caps should be ANY")) {
        return 1;
    }
    if (!expect(core::find_plugin_descriptor("leakflow_plugins_core") != nullptr,
            "core plugin descriptor was not findable")) {
        return 1;
    }
    if (!expect(core::find_plugin_descriptor("missing") == nullptr,
            "missing plugin descriptor was unexpectedly found")) {
        return 1;
    }

    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);

    return 0;
}
