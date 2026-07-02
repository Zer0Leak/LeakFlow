#pragma once

#include "leakflow/core/runtime_context.hpp"

namespace leakflow {

// Optional capability for elements that own runtime tasks.
//
// Most SCA transforms should remain passive process_pads(inputs)->outputs
// elements. Sources, Queue-like boundaries, hardware capture, and UI bridges may
// implement ActiveElement once they need to drive dataflow from their own task.
class ActiveElement {
public:
    virtual ~ActiveElement() = default;

    virtual void start_active(RuntimeContext &context) = 0;
    // Wait for natural completion without requesting stop. The pipeline calls
    // this while peer segment threads/tasks are still alive, then calls
    // stop_active() during teardown or failure cleanup.
    virtual void wait_active() = 0;
    virtual void stop_active() noexcept = 0;
};

} // namespace leakflow
