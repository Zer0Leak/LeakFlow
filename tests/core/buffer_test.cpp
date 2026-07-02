#include "leakflow/core/buffer.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

class TestPayload final : public leakflow::Payload {
public:
    explicit TestPayload(std::string value)
        : value_(std::move(value))
    {
    }

    [[nodiscard]] const std::string& value() const
    {
        return value_;
    }

    [[nodiscard]] std::string type_name() const override
    {
        return "test/payload";
    }

private:
    std::string value_;
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
    leakflow::Buffer buffer(leakflow::Caps("sca/traceset", {
        {"dtype", "float32"},
    }));

    if (!expect(buffer.caps().type() == "sca/traceset", "caps type was not preserved")) {
        return 1;
    }
    if (!expect(buffer.caps().param("dtype") == "float32", "caps params were not preserved")) {
        return 1;
    }

    buffer.caps().set_param("layout", "trace,sample");
    if (!expect(buffer.caps().param("layout") == "trace,sample", "mutable caps access failed")) {
        return 1;
    }

    buffer.set_metadata("source", "synthetic");
    buffer.set_metadata("trace_count", "16");

    if (!expect(buffer.has_metadata("source"), "has_metadata failed for existing key")) {
        return 1;
    }
    if (!expect(!buffer.has_metadata("missing"), "has_metadata succeeded for missing key")) {
        return 1;
    }
    if (!expect(buffer.metadata("source") == "synthetic", "metadata returned wrong value")) {
        return 1;
    }
    if (!expect(buffer.metadata_or("missing", "fallback") == "fallback", "metadata_or returned wrong fallback")) {
        return 1;
    }
    if (!expect(buffer.metadata().size() == 2, "metadata map size was wrong")) {
        return 1;
    }

    bool missing_threw = false;
    try {
        (void)buffer.metadata("missing");
    } catch (const std::out_of_range&) {
        missing_threw = true;
    }
    if (!expect(missing_threw, "missing metadata did not throw")) {
        return 1;
    }

    if (!expect(!buffer.has_payload(), "new buffer unexpectedly had a payload")) {
        return 1;
    }
    if (!expect(buffer.payload() == nullptr, "new buffer returned a non-null payload")) {
        return 1;
    }
    if (!expect(buffer.payload_as<TestPayload>() == nullptr, "payload_as succeeded for missing payload")) {
        return 1;
    }

    auto payload = std::make_shared<TestPayload>("payload-value");
    buffer.set_payload(payload);

    if (!expect(buffer.has_payload(), "has_payload failed after setting payload")) {
        return 1;
    }
    if (!expect(buffer.payload() == payload, "payload did not return the stored pointer")) {
        return 1;
    }

    const auto typed_payload = buffer.payload_as<TestPayload>();
    if (!expect(typed_payload == payload, "payload_as returned wrong typed pointer")) {
        return 1;
    }
    if (!expect(typed_payload->value() == "payload-value", "payload_as returned payload with wrong value")) {
        return 1;
    }
    if (!expect(buffer.payload_as<OtherPayload>() == nullptr, "payload_as succeeded for wrong payload type")) {
        return 1;
    }

    buffer.set_payload(nullptr);
    if (!expect(!buffer.has_payload(), "set_payload(nullptr) did not clear has_payload")) {
        return 1;
    }
    if (!expect(buffer.payload() == nullptr, "set_payload(nullptr) did not clear payload")) {
        return 1;
    }

    return 0;
}
