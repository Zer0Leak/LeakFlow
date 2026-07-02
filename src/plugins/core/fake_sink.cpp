#include "leakflow/plugins/core/fake_sink.hpp"

#include "core_plugin_constants.hpp"

#include <utility>

namespace leakflow::plugins::core {

ElementDescriptor FakeSink::descriptor()
{
    return {
        .type_name = "FakeSink",
        .klass = "Sink/Fake",
        .purpose = "consume buffers for validation",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(core_pad_caps_type)),
        },
        .output_pads = {},
        .property_specs = {
            PropertySpec("label", std::string("sink"), "label used for inspection and tests"),
        },
        .keywords = {"fake", "sink", "test"},
    };
}

FakeSink::FakeSink(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> FakeSink::process(std::optional<Buffer> input)
{
    received_ = input.has_value();
    last_buffer_ = std::move(input);

    auto record = make_log_record(log::LogLevel::Debug, "element", "consumed buffer");
    record.fields.emplace("received", received_ ? "true" : "false");
    leakflow::log::write(std::move(record));
    return std::nullopt;
}

bool FakeSink::received() const
{
    return received_;
}

const std::optional<Buffer>& FakeSink::last_buffer() const
{
    return last_buffer_;
}

} // namespace leakflow::plugins::core
