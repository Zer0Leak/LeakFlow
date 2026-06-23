#include "leakflow/core/pipeline.hpp"
#include "leakflow/plugins/core/core_elements.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

namespace {

class TestPayload final : public leakflow::Payload {
public:
    [[nodiscard]] std::string type_name() const override
    {
        return "test/payload";
    }
};

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

    leakflow::Buffer buffer(leakflow::Caps("sca/test"));
    buffer.set_metadata("source", "test");
    buffer.set_payload(std::make_shared<TestPayload>());

    core::Tee tee;
    if (!expect(tee.output_pads().empty(), "fresh Tee should not predeclare source pads")) {
        return 1;
    }
    if (!expect(tee.pad_templates().size() == 2, "Tee pad template count changed")) {
        return 1;
    }
    if (!expect(tee.pad_templates()[0].name() == "sink", "Tee sink template name changed")) {
        return 1;
    }
    if (!expect(tee.pad_templates()[0].caps().type() == leakflow::any_caps_type,
            "Tee sink template caps should be ANY")) {
        return 1;
    }
    if (!expect(tee.pad_templates()[1].name() == "src_%u", "Tee source template name changed")) {
        return 1;
    }
    if (!expect(tee.pad_templates()[1].presence() == leakflow::PadPresence::OnRequest,
            "Tee source template should be on request")) {
        return 1;
    }
    if (!expect(tee.pad_templates()[1].caps().type() == leakflow::any_caps_type,
            "Tee source template caps should be ANY")) {
        return 1;
    }
    if (!expect(tee.request_output_pad("src_0"), "Tee did not create requested src_0 pad")) {
        return 1;
    }
    if (!expect(tee.request_output_pad("src_1"), "Tee did not create requested src_1 pad")) {
        return 1;
    }
    const auto branches = tee.fork(buffer);
    if (!expect(branches.size() == 2, "Tee did not emit two branches")) {
        return 1;
    }
    if (!expect(branches[0].caps().type() == "sca/test", "Tee branch caps were not copied")) {
        return 1;
    }
    if (!expect(branches[0].metadata("source") == "test", "Tee branch metadata was not copied")) {
        return 1;
    }
    if (!expect(branches[0].payload() == branches[1].payload(), "Tee branches did not share payload pointer")) {
        return 1;
    }

    auto branch_a = branches[0];
    auto branch_b = branches[1];
    branch_a.set_metadata("branch", "a");
    if (!expect(!branch_b.has_metadata("branch"), "Tee branch metadata was not independent")) {
        return 1;
    }
    if (!expect(!branch_a.payload_is_unique(), "Tee branch payload unexpectedly reported unique")) {
        return 1;
    }

    auto source = std::make_shared<core::FakeSrc>("fake_src");
    auto linked_tee = std::make_shared<core::Tee>("tee");
    auto summary = std::make_shared<core::Summary>("summary");
    auto sink = std::make_shared<core::FakeSink>("fake_sink");

    leakflow::Pipeline pipeline;
    pipeline.add(source);
    pipeline.add(linked_tee);
    pipeline.add(summary);
    pipeline.add(sink);
    pipeline.link(source, "src", linked_tee, "sink");
    pipeline.link(linked_tee, "src_0", summary, "sink");
    pipeline.link(linked_tee, "src_1", sink, "sink");
    if (!expect(linked_tee->output_pads().size() == 2, "linked Tee did not create requested source pads")) {
        return 1;
    }

    (void)pipeline.run();
    if (!expect(!summary->last_summary().empty(), "linked Tee did not reach Summary branch")) {
        return 1;
    }
    if (!expect(sink->received(), "linked Tee did not reach FakeSink branch")) {
        return 1;
    }

    auto metadata_source = std::make_shared<core::FakeSrc>("metadata_fake_src");
    auto metadata_tee = std::make_shared<core::Tee>("metadata_tee");
    auto metadata_summary_0 = std::make_shared<core::Summary>("metadata_summary_0");
    auto metadata_summary_1 = std::make_shared<core::Summary>("metadata_summary_1");
    metadata_summary_0->set_property("level", std::int64_t{3});
    metadata_summary_1->set_property("level", std::int64_t{3});

    leakflow::Pipeline metadata_pipeline;
    metadata_pipeline.add(metadata_source);
    metadata_pipeline.add(metadata_tee);
    metadata_pipeline.add(metadata_summary_0);
    metadata_pipeline.add(metadata_summary_1);
    metadata_pipeline.link(metadata_source, "src", metadata_tee, "sink");
    metadata_pipeline.link(metadata_tee, "src_0", metadata_summary_0, "sink");
    metadata_pipeline.link(metadata_tee, "src_1", metadata_summary_1, "sink");
    metadata_pipeline.add_output_metadata_annotation(
        metadata_tee,
        {{"branch", "all"}, {"dataset", "smoke"}});
    metadata_pipeline.add_output_metadata_annotation_for_pad_template(
        metadata_tee,
        "src_%u",
        {{"branch", "template"}, {"branch.family", "tee"}});
    metadata_pipeline.add_output_metadata_annotation(
        metadata_tee,
        "src_0",
        {{"branch", "summary"}});

    (void)metadata_pipeline.run();
    if (!expect(metadata_summary_0->last_summary().find("branch=summary") != std::string::npos,
            "exact Tee metadata did not override template metadata")) {
        return 1;
    }
    if (!expect(metadata_summary_0->last_summary().find("dataset=smoke") != std::string::npos,
            "all-output Tee metadata was not stamped on src_0")) {
        return 1;
    }
    if (!expect(metadata_summary_1->last_summary().find("branch=template") != std::string::npos,
            "template Tee metadata was not stamped on src_1")) {
        return 1;
    }
    if (!expect(metadata_summary_1->last_summary().find("branch.family=tee") != std::string::npos,
            "template Tee metadata family was not stamped on src_1")) {
        return 1;
    }

    return 0;
}
