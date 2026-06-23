#include "leakflow/core/buffer.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace {

class TestPayload final : public leakflow::Payload {
public:
    explicit TestPayload(std::string value)
        : value(std::move(value))
    {
    }

    [[nodiscard]] std::string type_name() const override
    {
        return "test/payload";
    }

    std::string value;
};

class OtherPayload final : public leakflow::Payload {
public:
    [[nodiscard]] std::string type_name() const override
    {
        return "test/other-payload";
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
    leakflow::Buffer empty(leakflow::Caps("sca/test"));
    if (!expect(!empty.payload_is_unique(), "buffer without payload reported unique payload")) {
        return 1;
    }
    if (!expect(empty.mutable_payload_if_unique<TestPayload>() == nullptr,
            "mutable payload returned non-null for missing payload")) {
        return 1;
    }

    leakflow::Buffer unique(leakflow::Caps("sca/test"));
    unique.set_payload(std::make_shared<TestPayload>("unique"));
    if (!expect(unique.payload_is_unique(), "single-owner payload did not report unique")) {
        return 1;
    }

    auto* mutable_payload = unique.mutable_payload_if_unique<TestPayload>();
    if (!expect(mutable_payload != nullptr, "unique payload did not return mutable pointer")) {
        return 1;
    }
    mutable_payload->value = "mutated";
    if (!expect(unique.mutable_payload_if_unique<TestPayload>()->value == "mutated",
            "mutable payload pointer did not allow mutation")) {
        return 1;
    }

    if (!expect(unique.mutable_payload_if_unique<OtherPayload>() == nullptr,
            "mutable payload returned non-null for wrong type")) {
        return 1;
    }

    leakflow::Buffer shared = unique;
    if (!expect(!unique.payload_is_unique(), "shared source buffer still reported unique payload")) {
        return 1;
    }
    if (!expect(!shared.payload_is_unique(), "shared branch buffer reported unique payload")) {
        return 1;
    }
    if (!expect(unique.mutable_payload_if_unique<TestPayload>() == nullptr,
            "shared payload returned mutable pointer")) {
        return 1;
    }

    return 0;
}
