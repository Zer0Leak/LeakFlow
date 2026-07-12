#include "leakflow/core/payload.hpp"

#include <iostream>
#include <memory>
#include <string>

namespace {

class TestPayload final : public leakflow::Payload {
public:
    [[nodiscard]] std::string type_name() const override
    {
        return "test/payload";
    }

    [[nodiscard]] std::string layout() const override
    {
        return "trace/sample";
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
    auto payload = std::make_shared<TestPayload>();

    if (!expect(payload->type_name() == "test/payload", "payload type_name returned wrong value")) {
        return 1;
    }
    if (!expect(payload->layout() == "trace/sample", "payload layout returned wrong value")) {
        return 1;
    }

    std::shared_ptr<leakflow::Payload> base = payload;
    if (!expect(base->type_name() == "test/payload", "payload virtual dispatch failed")) {
        return 1;
    }
    if (!expect(base->layout() == "trace/sample", "payload layout virtual dispatch failed")) {
        return 1;
    }
    if (!expect(std::dynamic_pointer_cast<TestPayload>(base) == payload, "payload downcast failed")) {
        return 1;
    }

    return 0;
}
