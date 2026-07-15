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

    [[nodiscard]] std::string layout() const override
    {
        return "item";
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

    [[nodiscard]] std::string layout() const override
    {
        return "other_item";
    }
};

class EmptyLayoutPayload final : public leakflow::Payload {
public:
    [[nodiscard]] std::string type_name() const override
    {
        return "test/empty-layout-payload";
    }

    [[nodiscard]] std::string layout() const override
    {
        return {};
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

    buffer.set_metadata("payload.layout", "stale/layout");
    auto payload = std::make_shared<TestPayload>("payload-value");
    buffer.set_payload(payload);

    if (!expect(buffer.has_payload(), "has_payload failed after setting payload")) {
        return 1;
    }
    if (!expect(buffer.payload() == payload, "payload did not return the stored pointer")) {
        return 1;
    }
    if (!expect(buffer.metadata("payload.layout") == "item",
            "set_payload did not replace payload.layout metadata")) {
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

    bool empty_layout_threw = false;
    try {
        buffer.set_payload(std::make_shared<EmptyLayoutPayload>());
    } catch (const std::invalid_argument&) {
        empty_layout_threw = true;
    }
    if (!expect(empty_layout_threw, "set_payload accepted an empty payload layout")) {
        return 1;
    }
    if (!expect(buffer.payload() == payload,
            "rejected empty payload layout replaced the existing payload")) {
        return 1;
    }
    if (!expect(buffer.metadata("payload.layout") == "item",
            "rejected empty payload layout changed payload.layout metadata")) {
        return 1;
    }

    buffer.set_payload(nullptr);
    if (!expect(!buffer.has_payload(), "set_payload(nullptr) did not clear has_payload")) {
        return 1;
    }
    if (!expect(buffer.payload() == nullptr, "set_payload(nullptr) did not clear payload")) {
        return 1;
    }
    if (!expect(!buffer.has_metadata("payload.layout"),
            "set_payload(nullptr) did not erase payload.layout metadata")) {
        return 1;
    }

    // Axis labels: none by default, round-trip through the setter.
    if (!expect(buffer.units().empty(), "new buffer had axis labels")) {
        return 1;
    }
    buffer.set_units(leakflow::Units::of({3, 7, 9}));
    if (!expect(buffer.units() == leakflow::Units::of({3, 7, 9}), "axis labels did not round-trip")) {
        return 1;
    }

    // align_labels: shared labels + each input's positions, matched by value so the
    // two inputs may list the shared labels in different orders.
    {
        const auto alignment = leakflow::align_labels({1, 2, 3}, {3, 2, 5});
        if (!expect(alignment.shared == std::vector<std::int64_t>{2, 3}, "align_labels shared set wrong")) {
            return 1;
        }
        if (!expect(alignment.a_indices == std::vector<std::int64_t>{1, 2}
                    && alignment.b_indices == std::vector<std::int64_t>{1, 0},
                "align_labels positions wrong")) {
            return 1;
        }
        if (!expect(!alignment.identical, "align_labels wrongly reported identical")) {
            return 1;
        }
    }
    if (!expect(leakflow::align_labels({0, 1}, {0, 1}).identical, "align_labels missed an identical match")) {
        return 1;
    }
    if (!expect(leakflow::align_labels({1}, {0}).shared.empty(), "align_labels found a phantom shared label")) {
        return 1;
    }

    // describe(): leading-axis labels print in the Payload section as a bracketed
    // list named by the pluralised layout axis ("item" -> "items"; "unit" -> "units"
    // in the SCA layers).
    {
        leakflow::Buffer described(leakflow::Caps("sca/test"));
        described.set_payload(std::make_shared<TestPayload>("v"));
        described.set_units(leakflow::Units::of({5, 8}));
        const auto document = described.describe(2);
        bool printed = false;
        for (const auto& section : document.sections) {
            if (section.title != "Payload") {
                continue;
            }
            for (const auto& field : section.fields) {
                if (field.label == "items" && field.value.text == "[5,8]") {
                    printed = true;
                }
            }
        }
        if (!expect(printed, "describe did not print leading-axis labels in the Payload section")) {
            return 1;
        }
    }

    return 0;
}
