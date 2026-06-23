#include "leakflow/core/pipeline.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

template <typename Function>
bool throws_invalid_argument(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

class SourceElement final : public leakflow::Element {
public:
    SourceElement()
        : Element("source")
    {
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            return input;
        }

        leakflow::Buffer buffer(leakflow::Caps("sca/traceset", {
            {"dtype", "float32"},
        }));
        buffer.set_metadata("source", "generated");
        return buffer;
    }
};

class TransformElement final : public leakflow::Element {
public:
    TransformElement()
        : Element("transform")
    {
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            input->set_metadata("transform", "visited");
        }

        return input;
    }
};

class SinkElement final : public leakflow::Element {
public:
    SinkElement()
        : Element("sink")
    {
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        received = input.has_value();
        return std::nullopt;
    }

    bool received = false;
};

class OrderElement final : public leakflow::Element {
public:
    OrderElement(std::string name, std::vector<std::string>* events)
        : Element(std::move(name))
        , events_(events)
    {
    }

    void start() override
    {
        events_->push_back("start:" + name());
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        events_->push_back("process:" + name());
        return input;
    }

    void stop() override
    {
        events_->push_back("stop:" + name());
    }

private:
    std::vector<std::string>* events_;
};

class TypedElement final : public leakflow::Element {
public:
    TypedElement(std::string name, std::string type_name)
        : Element(std::move(name))
    {
        set_element_identity(std::move(type_name), "Test");
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        return input;
    }
};

} // namespace

int main()
{
    leakflow::Pipeline empty_pipeline;
    if (!expect(empty_pipeline.size() == 0, "empty pipeline size was not zero")) {
        return 1;
    }
    if (!expect(!empty_pipeline.run().has_value(), "empty pipeline should return nullopt")) {
        return 1;
    }

    leakflow::Pipeline source_pipeline;
    auto source_element = std::make_shared<SourceElement>();
    if (!expect(!source_element->name_locked(), "new element name was unexpectedly locked before pipeline add")) {
        return 1;
    }
    source_pipeline.add(source_element);
    auto source_output = source_pipeline.run();
    if (!expect(source_pipeline.size() == 1, "source pipeline size was wrong")) {
        return 1;
    }
    if (!expect(source_element->name_locked(), "pipeline did not lock the added element name")) {
        return 1;
    }
    if (!expect(source_pipeline.find_element("source") == source_element,
            "pipeline find_element did not return the added source")) {
        return 1;
    }
    if (!expect(source_pipeline.element("source") == source_element,
            "pipeline element lookup did not return the added source")) {
        return 1;
    }
    if (!expect(source_pipeline.find_element("missing") == nullptr,
            "pipeline find_element unexpectedly returned a missing element")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&source_pipeline] {
            (void)source_pipeline.element("missing");
        }),
            "pipeline element lookup did not reject a missing element")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&source_element] {
            source_element->set_property("name", std::string("renamed"));
        }),
            "pipeline-added element name was not locked")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&source_pipeline] {
            source_pipeline.add(std::make_shared<SourceElement>());
        }),
            "pipeline accepted a duplicate element instance name")) {
        return 1;
    }
    if (!expect(source_output.has_value(), "source-only pipeline did not return buffer")) {
        return 1;
    }

    leakflow::Pipeline transform_pipeline;
    transform_pipeline.add(std::make_shared<SourceElement>());
    transform_pipeline.add(std::make_shared<TransformElement>());
    auto transform_output = transform_pipeline.run();
    if (!expect(transform_output.has_value(), "source-transform pipeline did not return buffer")) {
        return 1;
    }
    if (!expect(transform_output->caps().type() == "sca/traceset", "source-transform did not preserve caps type")) {
        return 1;
    }
    if (!expect(transform_output->caps().param("dtype") == "float32", "source-transform did not preserve caps params")) {
        return 1;
    }
    if (!expect(transform_output->metadata("transform") == "visited", "transform did not add metadata")) {
        return 1;
    }

    auto sink = std::make_shared<SinkElement>();
    leakflow::Pipeline sink_pipeline;
    sink_pipeline.add(std::make_shared<SourceElement>());
    sink_pipeline.add(std::make_shared<TransformElement>());
    sink_pipeline.add(sink);
    auto sink_output = sink_pipeline.run();
    if (!expect(sink->received, "sink did not receive buffer")) {
        return 1;
    }
    if (!expect(!sink_output.has_value(), "source-transform-sink should return nullopt")) {
        return 1;
    }

    std::vector<std::string> events;
    leakflow::Pipeline order_pipeline;
    order_pipeline.add(std::make_shared<OrderElement>("a", &events));
    order_pipeline.add(std::make_shared<OrderElement>("b", &events));
    order_pipeline.add(std::make_shared<OrderElement>("c", &events));
    (void)order_pipeline.run();

    const std::vector<std::string> expected_events = {
        "start:a",
        "start:b",
        "start:c",
        "process:a",
        "process:b",
        "process:c",
        "stop:c",
        "stop:b",
        "stop:a",
    };

    if (!expect(events == expected_events, "lifecycle order was wrong")) {
        return 1;
    }

    leakflow::Pipeline type_pipeline;
    auto tee0 = type_pipeline.add(std::make_shared<TypedElement>("tee0", "Tee"));
    auto tee1 = type_pipeline.add(std::make_shared<TypedElement>("tee1", "Tee"));
    auto summary0 = type_pipeline.add(std::make_shared<TypedElement>("summary0", "Summary"));
    const auto tee_elements = type_pipeline.elements_by_type("tee");
    if (!expect(tee_elements.size() == 2, "pipeline type lookup did not find all Tee elements")) {
        return 1;
    }
    if (!expect(tee_elements[0] == tee0 && tee_elements[1] == tee1,
            "pipeline type lookup did not preserve add order")) {
        return 1;
    }
    const auto summary_elements = type_pipeline.elements_by_type("Summary");
    if (!expect(summary_elements.size() == 1 && summary_elements[0] == summary0,
            "pipeline type lookup did not find Summary by canonical type")) {
        return 1;
    }

    return 0;
}
