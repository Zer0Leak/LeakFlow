#include "leakflow/core/pipeline.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

class PassthroughElement final : public leakflow::Element {
public:
    explicit PassthroughElement(std::string name)
        : Element(std::move(name))
    {
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        processed = true;

        if (input) {
            input->set_metadata(name(), "visited");
        }

        return input;
    }

    bool processed = false;
};

class SourceElement final : public leakflow::Element {
public:
    SourceElement()
        : Element("source")
    {
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            return input;
        }

        leakflow::Buffer buffer(leakflow::Caps("sca/traceset"));
        buffer.set_metadata("source", "generated");
        return buffer;
    }
};

class SinkElement final : public leakflow::Element {
public:
    SinkElement()
        : Element("sink")
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        received = input.has_value();
        return std::nullopt;
    }

    bool received = false;
};

class ValueSourceElement final : public leakflow::Element {
public:
    ValueSourceElement(std::string name, std::string value)
        : Element(std::move(name))
        , value_(std::move(value))
    {
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            return input;
        }

        leakflow::Buffer buffer(leakflow::Caps("sca/traceset"));
        buffer.set_metadata("value", value_);
        return buffer;
    }

private:
    std::string value_;
};

class MultiInputElement final : public leakflow::Element {
public:
    MultiInputElement()
        : Element("join")
    {
        add_input_pad(leakflow::Pad("left", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset")));
        add_input_pad(leakflow::Pad("right", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset")));
        add_input_pad(leakflow::Pad(
            "traces",
            leakflow::PadDirection::Input,
            leakflow::Caps("sca/traceset"),
            leakflow::PadPresence::Optional));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        throw std::invalid_argument("MultiInputElement requires named inputs");
    }

    std::optional<leakflow::Buffer> process_inputs(leakflow::ElementInputs inputs) override
    {
        processed = true;

        const auto left = inputs.find("left");
        const auto right = inputs.find("right");
        if (left == inputs.end() || !left->second || right == inputs.end() || !right->second) {
            throw std::invalid_argument("missing required multi-input buffer");
        }

        saw_optional_traces = inputs.contains("traces");

        leakflow::Buffer output(leakflow::Caps("sca/traceset"));
        output.set_metadata("left", left->second->metadata("value"));
        output.set_metadata("right", right->second->metadata("value"));
        output.set_metadata("optional_traces", saw_optional_traces ? "connected" : "unconnected");
        return output;
    }

    bool processed = false;
    bool saw_optional_traces = false;
};

std::shared_ptr<PassthroughElement> make_linkable_element(
    std::string name, std::string input_name, std::string output_name, std::string caps_type)
{
    auto element = std::make_shared<PassthroughElement>(std::move(name));
    element->add_input_pad(
        leakflow::Pad(std::move(input_name), leakflow::PadDirection::Input, leakflow::Caps(caps_type)));
    element->add_output_pad(
        leakflow::Pad(std::move(output_name), leakflow::PadDirection::Output, leakflow::Caps(std::move(caps_type))));
    return element;
}

std::shared_ptr<PassthroughElement> make_linkable_element_with_caps(
    std::string name, std::string input_name, leakflow::Caps input_caps, std::string output_name, leakflow::Caps output_caps)
{
    auto element = std::make_shared<PassthroughElement>(std::move(name));
    element->add_input_pad(leakflow::Pad(std::move(input_name), leakflow::PadDirection::Input, std::move(input_caps)));
    element->add_output_pad(
        leakflow::Pad(std::move(output_name), leakflow::PadDirection::Output, std::move(output_caps)));
    return element;
}

bool throws_invalid_argument_for_link(leakflow::Pipeline& pipeline,
    std::shared_ptr<leakflow::Element> source_element,
    std::string source_pad_name,
    std::shared_ptr<leakflow::Element> sink_element,
    std::string sink_pad_name)
{
    try {
        pipeline.link(
            std::move(source_element), std::move(source_pad_name), std::move(sink_element), std::move(sink_pad_name));
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

std::optional<std::string> invalid_argument_message_for_link(leakflow::Pipeline& pipeline,
    std::shared_ptr<leakflow::Element> source_element,
    std::string source_pad_name,
    std::shared_ptr<leakflow::Element> sink_element,
    std::string sink_pad_name)
{
    try {
        pipeline.link(
            std::move(source_element), std::move(source_pad_name), std::move(sink_element), std::move(sink_pad_name));
    } catch (const std::invalid_argument& error) {
        return std::string(error.what());
    }

    return std::nullopt;
}

} // namespace

int main()
{
    leakflow::Pipeline empty_pipeline;
    if (!expect(empty_pipeline.links().empty(), "empty pipeline should start with zero links")) {
        return 1;
    }

    leakflow::Pipeline valid_pipeline;
    auto source = valid_pipeline.add(make_linkable_element("source", "unused-in", "traces", "sca/traceset"));
    auto normalize = valid_pipeline.add(make_linkable_element("normalize", "traces", "normalized", "sca/traceset"));
    auto sink = valid_pipeline.add(make_linkable_element("sink", "normalized", "unused-out", "sca/traceset"));

    valid_pipeline.link(source, "traces", normalize, "traces");
    if (!expect(valid_pipeline.links().size() == 1, "valid link was not stored")) {
        return 1;
    }
    if (!expect(valid_pipeline.links()[0].source_element == source, "source element handle was not preserved")) {
        return 1;
    }
    if (!expect(valid_pipeline.links()[0].source_pad_name == "traces", "source pad name was not preserved")) {
        return 1;
    }
    if (!expect(valid_pipeline.links()[0].sink_element == normalize, "sink element handle was not preserved")) {
        return 1;
    }
    if (!expect(valid_pipeline.links()[0].sink_pad_name == "traces", "sink pad name was not preserved")) {
        return 1;
    }

    valid_pipeline.link(normalize, "normalized", sink, "normalized");
    if (!expect(valid_pipeline.links().size() == 2, "second valid link was not stored")) {
        return 1;
    }
    if (!expect(valid_pipeline.links()[1].source_element == normalize, "second link source order was not preserved")) {
        return 1;
    }
    if (!expect(valid_pipeline.links()[1].sink_element == sink, "second link sink order was not preserved")) {
        return 1;
    }

    const auto foreign_source = make_linkable_element("foreign-source", "unused-in", "traces", "sca/traceset");
    const auto foreign_sink = make_linkable_element("foreign-sink", "traces", "unused-out", "sca/traceset");

    if (!expect(throws_invalid_argument_for_link(valid_pipeline, foreign_source, "traces", normalize, "traces"),
            "invalid source element handle was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument_for_link(valid_pipeline, source, "traces", foreign_sink, "traces"),
            "invalid sink element handle was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument_for_link(valid_pipeline, source, "missing", normalize, "traces"),
            "missing source pad was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument_for_link(valid_pipeline, source, "traces", normalize, "missing"),
            "missing sink pad was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument_for_link(valid_pipeline, normalize, "normalized", source, "unused-in"),
            "linking to an earlier element was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument_for_link(valid_pipeline, normalize, "normalized", normalize, "traces"),
            "linking an element to itself was not rejected")) {
        return 1;
    }

    leakflow::Pipeline mismatch_pipeline;
    auto mismatch_source = mismatch_pipeline.add(
        make_linkable_element("source", "unused-in", "traces", "sca/traceset"));
    auto mismatch_sink = mismatch_pipeline.add(make_linkable_element("sink", "labels", "unused-out", "sca/labels"));
    if (!expect(throws_invalid_argument_for_link(mismatch_pipeline, mismatch_source, "traces", mismatch_sink, "labels"),
            "caps type mismatch was not rejected")) {
        return 1;
    }
    auto mismatch_message = invalid_argument_message_for_link(mismatch_pipeline, mismatch_source, "traces", mismatch_sink, "labels");
    if (!expect(mismatch_message && mismatch_message->find("source.traces (sca/traceset)") != std::string::npos,
            "caps mismatch message did not include source endpoint caps")) {
        return 1;
    }
    if (!expect(mismatch_message->find("sink.labels (sca/labels)") != std::string::npos,
            "caps mismatch message did not include sink endpoint caps")) {
        return 1;
    }

    leakflow::Pipeline param_match_pipeline;
    auto param_match_source = param_match_pipeline.add(make_linkable_element_with_caps(
        "source",
        "unused-in",
        leakflow::Caps("leakflow/torch-tensor"),
        "src",
        leakflow::Caps("leakflow/torch-tensor", {{"dtype", "float32"}, {"device", "cpu"}})));
    auto param_match_sink = param_match_pipeline.add(make_linkable_element_with_caps(
        "sink",
        "sink",
        leakflow::Caps("leakflow/torch-tensor", {{"dtype", "float32"}}),
        "unused-out",
        leakflow::Caps("leakflow/torch-tensor")));
    param_match_pipeline.link(param_match_source, "src", param_match_sink, "sink");
    if (!expect(param_match_pipeline.links().size() == 1, "matching caps parameters were not linked")) {
        return 1;
    }

    leakflow::Pipeline param_mismatch_pipeline;
    auto param_mismatch_source = param_mismatch_pipeline.add(make_linkable_element_with_caps(
        "source",
        "unused-in",
        leakflow::Caps("leakflow/torch-tensor"),
        "src",
        leakflow::Caps("leakflow/torch-tensor", {{"dtype", "float32"}})));
    auto param_mismatch_sink = param_mismatch_pipeline.add(make_linkable_element_with_caps(
        "sink",
        "sink",
        leakflow::Caps("leakflow/torch-tensor", {{"dtype", "float64"}}),
        "unused-out",
        leakflow::Caps("leakflow/torch-tensor")));
    if (!expect(throws_invalid_argument_for_link(
            param_mismatch_pipeline, param_mismatch_source, "src", param_mismatch_sink, "sink"),
            "caps parameter mismatch was not rejected")) {
        return 1;
    }
    auto param_mismatch_message = invalid_argument_message_for_link(
        param_mismatch_pipeline, param_mismatch_source, "src", param_mismatch_sink, "sink");
    if (!expect(param_mismatch_message
            && param_mismatch_message->find("source.src (leakflow/torch-tensor; dtype=float32)") != std::string::npos,
            "caps parameter mismatch message did not include source caps params")) {
        return 1;
    }
    if (!expect(param_mismatch_message->find("sink.sink (leakflow/torch-tensor; dtype=float64)") != std::string::npos,
            "caps parameter mismatch message did not include sink caps params")) {
        return 1;
    }

    leakflow::Pipeline generic_sink_pipeline;
    auto numpy_source = generic_sink_pipeline.add(
        make_linkable_element("numpy-source", "unused-in", "src", "leakflow/numpy-array"));
    auto generic_sink = generic_sink_pipeline.add(
        make_linkable_element("summary", "sink", "unused-out", "leakflow/buffer"));
    generic_sink_pipeline.link(numpy_source, "src", generic_sink, "sink");
    if (!expect(generic_sink_pipeline.links().size() == 1,
            "concrete buffer caps did not link to generic buffer sink")) {
        return 1;
    }

    leakflow::Pipeline generic_source_pipeline;
    auto generic_source = generic_source_pipeline.add(
        make_linkable_element("generic-source", "unused-in", "src", "leakflow/buffer"));
    auto specific_sink = generic_source_pipeline.add(
        make_linkable_element("numpy-sink", "sink", "unused-out", "leakflow/numpy-array"));
    if (!expect(throws_invalid_argument_for_link(generic_source_pipeline, generic_source, "src", specific_sink, "sink"),
            "generic buffer source unexpectedly linked to concrete sink")) {
        return 1;
    }
    auto generic_source_message = invalid_argument_message_for_link(
        generic_source_pipeline, generic_source, "src", specific_sink, "sink");
    if (!expect(generic_source_message
            && generic_source_message->find("generic-source.src (leakflow/buffer)") != std::string::npos,
            "generic source mismatch message did not include source endpoint")) {
        return 1;
    }
    if (!expect(generic_source_message->find("numpy-sink.sink (leakflow/numpy-array)") != std::string::npos,
            "generic source mismatch message did not include sink endpoint")) {
        return 1;
    }
    if (!expect(generic_source_message->find("source pad declares generic leakflow/buffer") != std::string::npos,
            "generic source mismatch message did not include generic-source hint")) {
        return 1;
    }

    leakflow::Pipeline negotiated_passthrough_pipeline;
    auto torch_source = negotiated_passthrough_pipeline.add(make_linkable_element_with_caps(
        "torch-source",
        "unused-in",
        leakflow::Caps(leakflow::generic_buffer_caps_type),
        "src",
        leakflow::Caps("leakflow/torch-tensor")));
    auto passthrough = negotiated_passthrough_pipeline.add(make_linkable_element(
        "summary", "sink", "src", leakflow::generic_buffer_caps_type));
    auto torch_sink = negotiated_passthrough_pipeline.add(make_linkable_element_with_caps(
        "torch-sink",
        "sink",
        leakflow::Caps("leakflow/torch-tensor"),
        "unused-out",
        leakflow::Caps(leakflow::generic_buffer_caps_type)));
    negotiated_passthrough_pipeline.link(torch_source, "src", passthrough, "sink");
    negotiated_passthrough_pipeline.link(passthrough, "src", torch_sink, "sink");
    if (!expect(negotiated_passthrough_pipeline.links().size() == 2,
            "generic forwarding source did not negotiate upstream concrete caps")) {
        return 1;
    }

    leakflow::Pipeline negotiated_mismatch_pipeline;
    auto mismatch_torch_source = negotiated_mismatch_pipeline.add(make_linkable_element_with_caps(
        "torch-source",
        "unused-in",
        leakflow::Caps(leakflow::generic_buffer_caps_type),
        "src",
        leakflow::Caps("leakflow/torch-tensor")));
    auto mismatch_passthrough = negotiated_mismatch_pipeline.add(make_linkable_element(
        "summary", "sink", "src", leakflow::generic_buffer_caps_type));
    auto numpy_only_sink = negotiated_mismatch_pipeline.add(make_linkable_element_with_caps(
        "numpy-sink",
        "sink",
        leakflow::Caps("leakflow/numpy-array"),
        "unused-out",
        leakflow::Caps(leakflow::generic_buffer_caps_type)));
    negotiated_mismatch_pipeline.link(mismatch_torch_source, "src", mismatch_passthrough, "sink");
    if (!expect(throws_invalid_argument_for_link(
            negotiated_mismatch_pipeline, mismatch_passthrough, "src", numpy_only_sink, "sink"),
            "negotiated concrete caps mismatch was not rejected")) {
        return 1;
    }
    auto negotiated_mismatch_message = invalid_argument_message_for_link(
        negotiated_mismatch_pipeline, mismatch_passthrough, "src", numpy_only_sink, "sink");
    if (!expect(negotiated_mismatch_message
            && negotiated_mismatch_message->find("summary.src (leakflow/torch-tensor)") != std::string::npos,
            "negotiated mismatch message did not include resolved source caps")) {
        return 1;
    }
    if (!expect(negotiated_mismatch_message->find("numpy-sink.sink (leakflow/numpy-array)") != std::string::npos,
            "negotiated mismatch message did not include concrete sink caps")) {
        return 1;
    }

    leakflow::Pipeline negotiated_fanout_pipeline;
    auto fanout_torch_source = negotiated_fanout_pipeline.add(make_linkable_element_with_caps(
        "torch-source",
        "unused-in",
        leakflow::Caps(leakflow::generic_buffer_caps_type),
        "src",
        leakflow::Caps("leakflow/torch-tensor")));
    auto generic_tee = std::make_shared<PassthroughElement>("tee");
    generic_tee->add_input_pad(
        leakflow::Pad("sink", leakflow::PadDirection::Input, leakflow::Caps(leakflow::generic_buffer_caps_type)));
    generic_tee->add_output_pad(
        leakflow::Pad("src_0", leakflow::PadDirection::Output, leakflow::Caps(leakflow::generic_buffer_caps_type)));
    generic_tee->add_output_pad(
        leakflow::Pad("src_1", leakflow::PadDirection::Output, leakflow::Caps(leakflow::generic_buffer_caps_type)));
    negotiated_fanout_pipeline.add(generic_tee);
    auto generic_branch_sink = negotiated_fanout_pipeline.add(
        make_linkable_element("summary", "sink", "unused-out", leakflow::generic_buffer_caps_type));
    auto concrete_branch_sink = negotiated_fanout_pipeline.add(make_linkable_element_with_caps(
        "torch-sink",
        "sink",
        leakflow::Caps("leakflow/torch-tensor"),
        "unused-out",
        leakflow::Caps(leakflow::generic_buffer_caps_type)));
    negotiated_fanout_pipeline.link(fanout_torch_source, "src", generic_tee, "sink");
    negotiated_fanout_pipeline.link(generic_tee, "src_0", generic_branch_sink, "sink");
    negotiated_fanout_pipeline.link(generic_tee, "src_1", concrete_branch_sink, "sink");
    if (!expect(negotiated_fanout_pipeline.links().size() == 3,
            "generic fan-out did not negotiate concrete caps on a branch")) {
        return 1;
    }

    leakflow::Pipeline request_pad_pipeline;
    auto request_source = request_pad_pipeline.add(make_linkable_element_with_caps(
        "torch-source",
        "unused-in",
        leakflow::Caps(leakflow::generic_buffer_caps_type),
        "src",
        leakflow::Caps("leakflow/torch-tensor")));
    auto request_tee = std::make_shared<PassthroughElement>("request-tee");
    request_tee->add_pad_template(leakflow::Pad(
        "src_%u",
        leakflow::PadDirection::Output,
        leakflow::Caps(leakflow::any_caps_type),
        leakflow::PadPresence::OnRequest));
    request_tee->add_input_pad(
        leakflow::Pad("sink", leakflow::PadDirection::Input, leakflow::Caps(leakflow::any_caps_type)));
    request_pad_pipeline.add(request_tee);
    auto request_sink = request_pad_pipeline.add(make_linkable_element_with_caps(
        "torch-sink",
        "sink",
        leakflow::Caps("leakflow/torch-tensor"),
        "unused-out",
        leakflow::Caps(leakflow::generic_buffer_caps_type)));
    request_pad_pipeline.link(request_source, "src", request_tee, "sink");
    request_pad_pipeline.link(request_tee, "src_0", request_sink, "sink");
    if (!expect(request_tee->output_pads().size() == 1,
            "request pad link did not create one output pad")) {
        return 1;
    }
    if (!expect(request_tee->output_pads()[0].name() == "src_0",
            "request pad link created the wrong output pad")) {
        return 1;
    }
    if (!expect(request_pad_pipeline.source_caps(*request_tee, "src_0").type() == "leakflow/torch-tensor",
            "request pad source caps did not resolve through upstream concrete caps")) {
        return 1;
    }

    if (!expect(throws_invalid_argument_for_link(valid_pipeline, source, "traces", normalize, "traces"),
            "exact duplicate link was not rejected")) {
        return 1;
    }

    leakflow::Pipeline fanout_pipeline;
    auto fanout_source = fanout_pipeline.add(
        make_linkable_element("source", "unused-in", "traces", "sca/traceset"));
    auto fanout_sink_a = fanout_pipeline.add(make_linkable_element("sink-a", "traces", "unused-out", "sca/traceset"));
    auto fanout_sink_b = fanout_pipeline.add(make_linkable_element("sink-b", "traces", "unused-out", "sca/traceset"));
    fanout_pipeline.link(fanout_source, "traces", fanout_sink_a, "traces");
    if (!expect(throws_invalid_argument_for_link(fanout_pipeline, fanout_source, "traces", fanout_sink_b, "traces"),
            "already-linked source pad was not rejected")) {
        return 1;
    }

    leakflow::Pipeline fanin_pipeline;
    auto fanin_source_a = fanin_pipeline.add(
        make_linkable_element("source-a", "unused-in", "traces-a", "sca/traceset"));
    auto fanin_source_b = fanin_pipeline.add(
        make_linkable_element("source-b", "unused-in", "traces-b", "sca/traceset"));
    auto fanin_sink = fanin_pipeline.add(make_linkable_element("sink", "traces", "unused-out", "sca/traceset"));
    fanin_pipeline.link(fanin_source_a, "traces-a", fanin_sink, "traces");
    if (!expect(throws_invalid_argument_for_link(fanin_pipeline, fanin_source_b, "traces-b", fanin_sink, "traces"),
            "already-linked sink pad was not rejected")) {
        return 1;
    }

    leakflow::Pipeline run_pipeline;
    auto run_source = run_pipeline.add(std::make_shared<SourceElement>());
    auto unlinked = std::make_shared<PassthroughElement>("unlinked");
    run_pipeline.add(unlinked);
    auto run_sink = std::make_shared<SinkElement>();
    run_pipeline.add(run_sink);
    run_pipeline.link(run_source, "out", run_sink, "in");

    auto run_output = run_pipeline.run();
    if (!expect(run_sink->received, "linked run did not deliver buffer to linked sink")) {
        return 1;
    }
    if (!expect(!unlinked->processed, "linked run processed an unlinked intermediate element")) {
        return 1;
    }
    if (!expect(!run_output.has_value(), "linked run returned unexpected output")) {
        return 1;
    }

    leakflow::Pipeline multi_input_pipeline;
    auto left_source = multi_input_pipeline.add(std::make_shared<ValueSourceElement>("left-source", "left-value"));
    auto right_source = multi_input_pipeline.add(std::make_shared<ValueSourceElement>("right-source", "right-value"));
    auto join = std::make_shared<MultiInputElement>();
    multi_input_pipeline.add(join);
    auto join_sink = std::make_shared<PassthroughElement>("join-sink");
    join_sink->add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset")));
    multi_input_pipeline.add(join_sink);

    multi_input_pipeline.link(left_source, "out", join, "left");
    multi_input_pipeline.link(right_source, "out", join, "right");
    multi_input_pipeline.link(join, "out", join_sink, "in");

    auto multi_input_output = multi_input_pipeline.run();
    if (!expect(join->processed, "multi-input element was not processed")) {
        return 1;
    }
    if (!expect(!join->saw_optional_traces, "unlinked optional pad was reported as connected")) {
        return 1;
    }
    if (!expect(multi_input_output.has_value(), "multi-input pipeline did not return terminal output")) {
        return 1;
    }
    if (!expect(multi_input_output->metadata("left") == "left-value",
            "multi-input element did not receive left input")) {
        return 1;
    }
    if (!expect(multi_input_output->metadata("right") == "right-value",
            "multi-input element did not receive right input")) {
        return 1;
    }
    if (!expect(multi_input_output->metadata("optional_traces") == "unconnected",
            "multi-input element did not tolerate unconnected optional input")) {
        return 1;
    }

    return 0;
}
