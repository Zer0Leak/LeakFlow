#pragma once

#include <string_view>

namespace leakflow {

// Contract-only enum (Phase 25). Defines how a future threaded/live Queue should
// treat buffers from an older configuration epoch when a control change starts a
// new epoch. Phase 25 keeps Queue synchronous and does not enforce any of these;
// the enum exists so live mode (StreamingDrive) has a defined vocabulary and so
// "keep mixed" is never the silent default for reproducible SCA workflows. See
// docs/design/pipeline_controller.md.
enum class QueueEpochPolicy {
    Drain,      // process buffered old-epoch buffers, then switch to the new epoch
    Flush,      // discard buffered old-epoch buffers immediately
    KeepMixed,  // allow old and new epochs to coexist (rarely correct for SCA)
    Block,      // block new-epoch input until old-epoch buffers are consumed
    DropOldest, // drop the oldest buffered buffer when full
    DropNewest, // drop the newest buffered buffer when full
};

[[nodiscard]] std::string_view queue_epoch_policy_name(QueueEpochPolicy policy);

} // namespace leakflow
