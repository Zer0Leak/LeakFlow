#pragma once

#include <memory>
#include <vector>

namespace leakflow {

class Element;
class Pipeline;

// A maximal run of non-boundary elements with no Queue between them (live phase,
// S10). Each segment runs on one streaming thread. Boundary Queues are NOT segment
// members: a segment PULLS from its input_queues (at its head) and PUSHES to its
// output_queues (at its tail); the Queue is the thread-safe handoff between a
// producing and a consuming segment. See docs/design/dataflow_sync_model.md S10.
struct PipelineSegment {
    std::vector<std::shared_ptr<Element>> elements;      // in pipeline (topological) order
    std::vector<std::shared_ptr<Element>> input_queues;  // boundary Queues this segment pulls from
    std::vector<std::shared_ptr<Element>> output_queues; // boundary Queues this segment pushes into
};

// Decompose a pipeline into segments by cutting at every Queue (thread boundary,
// S10). The number of segments is the live runner's thread count: "two queues into
// the same join collapse into one downstream thread; two queues into different
// downstream elements give two threads." A Queue-free pipeline yields exactly one
// segment with all elements (identical to the offline single sweep). Segments are
// returned in pipeline order (by their first element).
[[nodiscard]] std::vector<PipelineSegment> decompose_into_segments(const Pipeline &pipeline);

} // namespace leakflow
