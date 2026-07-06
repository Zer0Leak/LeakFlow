#include "leakflow/core/buffer.hpp"
#include "leakflow/core/metadata_forwarding.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }

    return condition;
}

leakflow::Buffer make_buffer()
{
    return leakflow::Buffer(leakflow::Caps("leakflow/torch-tensor"));
}

bool has(const leakflow::Buffer& buffer, const std::string& key, const std::string& value)
{
    return buffer.has_metadata(key) && buffer.metadata(key) == value;
}

} // namespace

int main()
{
    using leakflow::ForwardingProfile;
    using leakflow::MetadataGroup;
    using leakflow::forward_metadata;
    using leakflow::metadata_group;
    using leakflow::profile_for_klass;

    // Group classification: the leading segment is the group.
    expect(metadata_group("capture.source") == MetadataGroup::Capture, "capture.* not capture");
    expect(metadata_group("capture.dataset.name") == MetadataGroup::Capture, "capture.dataset.* not capture");
    expect(metadata_group("capture.sample_rate_hz") == MetadataGroup::Capture, "capture.sample_rate not capture");
    expect(metadata_group("origin.file.path") == MetadataGroup::Origin, "origin.file.* not origin");
    expect(metadata_group("origin.role") == MetadataGroup::Origin, "origin.role not origin");
    expect(metadata_group("origin.traces.file.path") == MetadataGroup::Origin, "fused origin not origin");
    expect(metadata_group("payload.leakage.model") == MetadataGroup::Payload, "payload.* not payload");
    expect(metadata_group("payload.poi.method") == MetadataGroup::Payload, "payload.poi not payload");
    expect(metadata_group("totally_unprefixed_key") == MetadataGroup::Payload, "unprefixed not payload default");
    expect(metadata_group("routing.element") == MetadataGroup::Routing, "routing.element not routing");
    expect(metadata_group("routing.branch") == MetadataGroup::Routing, "routing.branch not routing");

    // Profile classification by leading klass token.
    expect(profile_for_klass("Source/File/Torch") == ForwardingProfile::Source, "Source klass");
    expect(profile_for_klass("Sink/Plot/Trace") == ForwardingProfile::Sink, "Sink klass");
    expect(profile_for_klass("PassThrough/Flow/Queue") == ForwardingProfile::PassThrough, "PassThrough klass");
    expect(profile_for_klass("Convert/Tensor/Torch") == ForwardingProfile::Reframe, "Convert klass");
    expect(profile_for_klass("Analyze/SCA/Leakage/AES") == ForwardingProfile::Analyze, "Analyze klass");
    expect(profile_for_klass("Control/Fault/Voltage") == ForwardingProfile::PassThrough, "unknown klass default");

    // Reframe (single buffer): copy capture + origin as-is; drop payload + routing.
    {
        auto input = make_buffer();
        input.set_metadata("capture.sample_rate_hz", "29454545.45");
        input.set_metadata("origin.file.path", "traces.pt");
        input.set_metadata("origin.role", "traces");
        input.set_metadata("payload.leakage.model", "aes");
        input.set_metadata("routing.branch", "analysis");
        input.set_metadata("routing.element", "torchconvert0");

        auto output = make_buffer();
        forward_metadata(input, ForwardingProfile::Reframe, output, "sink");

        expect(has(output, "capture.sample_rate_hz", "29454545.45"), "Reframe dropped capture");
        expect(has(output, "origin.file.path", "traces.pt"), "Reframe dropped origin file");
        expect(has(output, "origin.role", "traces"), "Reframe dropped origin role");
        expect(!output.has_metadata("payload.leakage.model"), "Reframe kept payload");
        expect(!output.has_metadata("routing.branch"), "Reframe kept routing branch");
        expect(!output.has_metadata("routing.element"), "Reframe kept routing element");
        expect(!output.has_metadata("origin.sink.file.path"), "Reframe should not prefix origin");
    }

    // Analyze (multi input): union capture; relabel origin as origin.<pad>.<key>; drop payload + routing.
    {
        auto traces = make_buffer();
        traces.set_metadata("capture.sample_rate_hz", "29454545.45");
        traces.set_metadata("capture.source", "ChipWhisperer");
        traces.set_metadata("origin.file.path", "traces.pt");
        traces.set_metadata("routing.branch", "analysis-leakage");

        auto keys = make_buffer();
        keys.set_metadata("capture.sample_rate_hz", "29454545.45");
        keys.set_metadata("origin.file.path", "key.pt");
        keys.set_metadata("origin.role", "key");

        leakflow::ElementInputs inputs;
        inputs.emplace("traces", std::move(traces));
        inputs.emplace("keys", std::move(keys));

        auto output = make_buffer();
        forward_metadata(inputs, ForwardingProfile::Analyze, output);

        expect(has(output, "capture.sample_rate_hz", "29454545.45"), "Analyze dropped unioned capture");
        expect(has(output, "capture.source", "ChipWhisperer"), "Analyze dropped capture.source");
        expect(has(output, "origin.traces.file.path", "traces.pt"), "Analyze did not prefix traces origin");
        expect(has(output, "origin.keys.file.path", "key.pt"), "Analyze did not prefix keys origin");
        expect(has(output, "origin.keys.role", "key"), "Analyze did not prefix role");
        expect(!output.has_metadata("origin.file.path"), "Analyze left bare origin");
        expect(!output.has_metadata("routing.branch"), "Analyze kept routing");
    }

    // Analyze: conflicting capture fact across inputs is an error.
    {
        auto a = make_buffer();
        a.set_metadata("capture.sample_rate_hz", "29454545.45");
        auto b = make_buffer();
        b.set_metadata("capture.sample_rate_hz", "10000000.0");

        leakflow::ElementInputs inputs;
        inputs.emplace("a", std::move(a));
        inputs.emplace("b", std::move(b));

        auto output = make_buffer();
        bool threw = false;
        std::string message;
        try {
            forward_metadata(inputs, ForwardingProfile::Analyze, output, "poi");
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        expect(threw, "Analyze did not error on conflicting capture metadata");
        expect(message.find("element 'poi'") != std::string::npos,
            "Analyze conflict did not identify the forwarding element");
        expect(message.find("input pad 'a'") != std::string::npos
                && message.find("input pad 'b'") != std::string::npos,
            "Analyze conflict did not identify both conflicting input pads");
    }

    // Analyze: an already-forwarded origin key is flattened, not re-nested.
    {
        auto targets = make_buffer();
        targets.set_metadata("origin.keys.file.path", "key.pt");
        targets.set_metadata("origin.file.path", "leakage.pt");

        leakflow::ElementInputs inputs;
        inputs.emplace("targets", std::move(targets));

        auto output = make_buffer();
        forward_metadata(inputs, ForwardingProfile::Analyze, output);
        expect(has(output, "origin.targets.keys.file.path", "key.pt"), "Analyze did not flatten nested origin");
        expect(has(output, "origin.targets.file.path", "leakage.pt"), "Analyze did not prefix bare origin");
        expect(!output.has_metadata("origin.targets.origin.keys.file.path"), "Analyze double-nested origin");
    }

    // PassThrough: copy capture + origin + payload; drop routing only.
    {
        auto input = make_buffer();
        input.set_metadata("capture.sample_rate_hz", "1.0");
        input.set_metadata("origin.file.path", "x.pt");
        input.set_metadata("payload.leakage.model", "aes");
        input.set_metadata("routing.branch", "b");

        auto output = make_buffer();
        forward_metadata(input, ForwardingProfile::PassThrough, output, "sink");
        expect(has(output, "payload.leakage.model", "aes"), "PassThrough dropped payload");
        expect(has(output, "origin.file.path", "x.pt"), "PassThrough dropped origin");
        expect(!output.has_metadata("routing.branch"), "PassThrough kept routing");
    }

    // Source/Sink: no-op.
    {
        auto input = make_buffer();
        input.set_metadata("capture.sample_rate_hz", "1.0");
        auto output = make_buffer();
        forward_metadata(input, ForwardingProfile::Source, output, "sink");
        forward_metadata(input, ForwardingProfile::Sink, output, "sink");
        expect(output.metadata().empty(), "Source/Sink forwarded metadata");
    }

    return failures == 0 ? 0 : 1;
}
