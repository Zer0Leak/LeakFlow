#include "leakflow/core/pipeline_segments.hpp"

#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline.hpp"

#include <cstddef>
#include <map>
#include <vector>

namespace leakflow {
namespace {

[[nodiscard]] bool is_boundary(const std::shared_ptr<Element> &element) {
    return element && element->is_thread_boundary();
}

} // namespace

std::vector<PipelineSegment> decompose_into_segments(const Pipeline &pipeline) {
    const auto &elements = pipeline.elements();
    const auto &links = pipeline.links();

    // Undirected adjacency among non-boundary elements. A link touching a Queue on
    // either end is a cut edge (the thread boundary), so it is skipped: that is how
    // "cutting at every Queue" splits the graph into segments (S10).
    std::map<const Element *, std::vector<const Element *>> adjacency;
    for (const auto &element : elements) {
        if (!is_boundary(element)) {
            adjacency.try_emplace(element.get());
        }
    }
    for (const auto &link : links) {
        if (is_boundary(link.source_element) || is_boundary(link.sink_element)) {
            continue;
        }
        adjacency[link.source_element.get()].push_back(link.sink_element.get());
        adjacency[link.sink_element.get()].push_back(link.source_element.get());
    }

    // Connected components (iterative DFS), visiting in pipeline order so segment
    // ids -- and thus the returned order -- follow pipeline order.
    std::map<const Element *, int> component;
    int component_count = 0;
    for (const auto &element : elements) {
        if (is_boundary(element) || component.contains(element.get())) {
            continue;
        }
        const int id = component_count++;
        std::vector<const Element *> stack{element.get()};
        component[element.get()] = id;
        while (!stack.empty()) {
            const auto *current = stack.back();
            stack.pop_back();
            for (const auto *neighbor : adjacency[current]) {
                if (component.try_emplace(neighbor, id).second) {
                    stack.push_back(neighbor);
                }
            }
        }
    }

    std::vector<PipelineSegment> segments(static_cast<std::size_t>(component_count));
    for (const auto &element : elements) {
        if (is_boundary(element)) {
            continue;
        }
        segments[static_cast<std::size_t>(component.at(element.get()))].elements.push_back(element);
    }

    // Attach each boundary Queue to its producing and consuming segments. The
    // Queue's incoming link names the producer (its segment pushes into the Queue);
    // its outgoing link names the consumer (its segment pulls from the Queue).
    for (const auto &element : elements) {
        if (!is_boundary(element)) {
            continue;
        }
        for (const auto &link : links) {
            if (link.sink_element.get() == element.get() && !is_boundary(link.source_element)) {
                if (const auto found = component.find(link.source_element.get()); found != component.end()) {
                    segments[static_cast<std::size_t>(found->second)].output_queues.push_back(element);
                }
            }
            if (link.source_element.get() == element.get() && !is_boundary(link.sink_element)) {
                if (const auto found = component.find(link.sink_element.get()); found != component.end()) {
                    segments[static_cast<std::size_t>(found->second)].input_queues.push_back(element);
                }
            }
        }
    }

    return segments;
}

} // namespace leakflow
