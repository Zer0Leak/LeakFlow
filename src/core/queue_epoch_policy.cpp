#include "leakflow/core/queue_epoch_policy.hpp"

namespace leakflow {

std::string_view queue_epoch_policy_name(QueueEpochPolicy policy)
{
    switch (policy) {
    case QueueEpochPolicy::Drain:
        return "drain";
    case QueueEpochPolicy::Flush:
        return "flush";
    case QueueEpochPolicy::KeepMixed:
        return "keep-mixed";
    case QueueEpochPolicy::Block:
        return "block";
    case QueueEpochPolicy::DropOldest:
        return "drop-oldest";
    case QueueEpochPolicy::DropNewest:
        return "drop-newest";
    }

    return "unknown";
}

} // namespace leakflow
