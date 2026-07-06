#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_segments.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

leakflow::Caps generic()
{
    return leakflow::Caps(leakflow::generic_buffer_caps_type);
}

// Minimal pad-carrying elements. decompose_into_segments() is a static topology
// pass, so process() is never called here -- the bodies are trivial.
class Src final : public leakflow::Element {
public:
    explicit Src(std::string name)
        : Element(std::move(name))
    {
        add_output_pad(leakflow::Pad("src", leakflow::PadDirection::Output, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
};

class TwoOutSrc final : public leakflow::Element {
public:
    explicit TwoOutSrc(std::string name)
        : Element(std::move(name))
    {
        add_output_pad(leakflow::Pad("src0", leakflow::PadDirection::Output, generic()));
        add_output_pad(leakflow::Pad("src1", leakflow::PadDirection::Output, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
};

class Mid final : public leakflow::Element {
public:
    explicit Mid(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }
};

class Join final : public leakflow::Element {
public:
    explicit Join(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in0", leakflow::PadDirection::Input, generic()));
        add_input_pad(leakflow::Pad("in1", leakflow::PadDirection::Input, generic()));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }
};

class Sink final : public leakflow::Element {
public:
    explicit Sink(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
};

// A thread-boundary element (a stand-in Queue) configured purely via the
// descriptor flag the decomposition reads -- no plugin dependency.
class TestQueue final : public leakflow::Element {
public:
    explicit TestQueue(std::string name)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "TestQueue";
        descriptor.klass = "PassThrough/Flow/Queue";
        descriptor.input_pads = {leakflow::Pad("sink", leakflow::PadDirection::Input, generic())};
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.thread_boundary = true;
        configure_from_descriptor(descriptor);
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }
};

// Index of the segment containing the element named `name`, or -1.
int segment_of(const std::vector<leakflow::PipelineSegment>& segments, const std::string& name)
{
    for (std::size_t s = 0; s < segments.size(); ++s) {
        for (const auto& element : segments[s].elements) {
            if (element->name() == name) {
                return static_cast<int>(s);
            }
        }
    }
    return -1;
}

bool segment_has_queue(const std::vector<std::shared_ptr<leakflow::Element>>& queues, const std::string& name)
{
    return std::any_of(queues.begin(), queues.end(),
                       [&](const auto& queue) { return queue->name() == name; });
}

} // namespace

int main()
{
    // Case 1: a Queue-free pipeline is exactly one segment with every element,
    // matching the offline single sweep (no thread boundaries).
    {
        leakflow::Pipeline pipeline;
        auto a = pipeline.add(std::make_shared<Src>("a"));
        auto b = pipeline.add(std::make_shared<Mid>("b"));
        auto c = pipeline.add(std::make_shared<Sink>("c"));
        pipeline.link(a, "src", b, "in");
        pipeline.link(b, "out", c, "in");

        const auto segments = decompose_into_segments(pipeline);
        if (!expect(segments.size() == 1, "queue-free pipeline must be one segment")) {
            return 1;
        }
        if (!expect(segments[0].elements.size() == 3, "the single segment must hold all 3 elements")) {
            return 1;
        }
        if (!expect(segments[0].input_queues.empty() && segments[0].output_queues.empty(),
                    "queue-free segment must have no boundary queues")) {
            return 1;
        }
        // Order preserved (a, b, c).
        if (!expect(segments[0].elements[0]->name() == "a" && segments[0].elements[2]->name() == "c",
                    "segment must keep pipeline order")) {
            return 1;
        }
    }

    // Case 2: one Queue cuts a linear pipeline into two segments; the producer
    // segment pushes into the Queue, the consumer segment pulls from it.
    {
        leakflow::Pipeline pipeline;
        auto a = pipeline.add(std::make_shared<Src>("a"));
        auto q = pipeline.add(std::make_shared<TestQueue>("q"));
        auto c = pipeline.add(std::make_shared<Sink>("c"));
        pipeline.link(a, "src", q, "sink");
        pipeline.link(q, "src", c, "in");

        const auto segments = decompose_into_segments(pipeline);
        if (!expect(segments.size() == 2, "one queue must yield two segments")) {
            return 1;
        }
        const int sa = segment_of(segments, "a");
        const int sc = segment_of(segments, "c");
        if (!expect(sa == 0 && sc == 1, "segments must be in pipeline order (a then c)")) {
            return 1;
        }
        if (!expect(segment_of(segments, "q") == -1, "a boundary Queue is not a segment member")) {
            return 1;
        }
        if (!expect(segment_has_queue(segments[0].output_queues, "q"), "producer segment must push into q")) {
            return 1;
        }
        if (!expect(segment_has_queue(segments[1].input_queues, "q"), "consumer segment must pull from q")) {
            return 1;
        }
        if (!expect(segments[0].input_queues.empty() && segments[1].output_queues.empty(),
                    "boundary attachment must be one-directional per segment here")) {
            return 1;
        }
    }

    // Case 3: two independent sources, each behind its own Queue, feeding ONE join.
    // "Two queues into the same join collapse into one downstream thread": the join
    // is a single segment pulling from both queues -> three segments total.
    {
        leakflow::Pipeline pipeline;
        auto a = pipeline.add(std::make_shared<Src>("a"));
        auto b = pipeline.add(std::make_shared<Src>("b"));
        auto qa = pipeline.add(std::make_shared<TestQueue>("qa"));
        auto qb = pipeline.add(std::make_shared<TestQueue>("qb"));
        auto d = pipeline.add(std::make_shared<Join>("d"));
        pipeline.link(a, "src", qa, "sink");
        pipeline.link(b, "src", qb, "sink");
        pipeline.link(qa, "src", d, "in0");
        pipeline.link(qb, "src", d, "in1");

        const auto segments = decompose_into_segments(pipeline);
        if (!expect(segments.size() == 3, "two queues into one join must give three segments")) {
            return 1;
        }
        const int sd = segment_of(segments, "d");
        if (!expect(sd >= 0 && segments[static_cast<std::size_t>(sd)].elements.size() == 1,
                    "the join is its own single downstream segment")) {
            return 1;
        }
        if (!expect(segment_has_queue(segments[static_cast<std::size_t>(sd)].input_queues, "qa") &&
                        segment_has_queue(segments[static_cast<std::size_t>(sd)].input_queues, "qb"),
                    "the join segment pulls from both queues")) {
            return 1;
        }
        if (!expect(segment_of(segments, "a") != segment_of(segments, "b"),
                    "independent sources are in different segments")) {
            return 1;
        }
    }

    // Case 4: a rejoining diamond with a Queue on each branch. The upstream tee
    // stays one segment (no Queue between A, B, C); D is the aggregator segment.
    // "Where the queues lead" -> both lead to D, so two segments total.
    {
        leakflow::Pipeline pipeline;
        auto a = pipeline.add(std::make_shared<TwoOutSrc>("a"));
        auto b = pipeline.add(std::make_shared<Mid>("b"));
        auto c = pipeline.add(std::make_shared<Mid>("c"));
        auto qb = pipeline.add(std::make_shared<TestQueue>("qb"));
        auto qc = pipeline.add(std::make_shared<TestQueue>("qc"));
        auto d = pipeline.add(std::make_shared<Join>("d"));
        pipeline.link(a, "src0", b, "in");
        pipeline.link(a, "src1", c, "in");
        pipeline.link(b, "out", qb, "sink");
        pipeline.link(c, "out", qc, "sink");
        pipeline.link(qb, "src", d, "in0");
        pipeline.link(qc, "src", d, "in1");

        const auto segments = decompose_into_segments(pipeline);
        if (!expect(segments.size() == 2, "diamond with queued branches must give two segments")) {
            return 1;
        }
        const int sa = segment_of(segments, "a");
        if (!expect(sa >= 0 && segment_of(segments, "b") == sa && segment_of(segments, "c") == sa,
                    "the upstream tee (a, b, c) stays one segment")) {
            return 1;
        }
        const int sd = segment_of(segments, "d");
        if (!expect(sd >= 0 && sd != sa, "the aggregator d is a separate segment")) {
            return 1;
        }
        if (!expect(segment_has_queue(segments[static_cast<std::size_t>(sa)].output_queues, "qb") &&
                        segment_has_queue(segments[static_cast<std::size_t>(sa)].output_queues, "qc"),
                    "the upstream segment pushes into both branch queues")) {
            return 1;
        }
        if (!expect(segment_has_queue(segments[static_cast<std::size_t>(sd)].input_queues, "qb") &&
                        segment_has_queue(segments[static_cast<std::size_t>(sd)].input_queues, "qc"),
                    "the aggregator pulls from both branch queues")) {
            return 1;
        }
    }

    return 0;
}
