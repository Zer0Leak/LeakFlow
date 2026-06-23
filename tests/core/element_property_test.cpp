#include "leakflow/core/element.hpp"

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

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

class ProbeElement final : public leakflow::Element {
public:
    explicit ProbeElement(std::string name)
        : Element(std::move(name))
    {
    }

    void start() override
    {
        started = true;
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        processed = true;
        return input;
    }

    void stop() override
    {
        stopped = true;
    }

    bool started = false;
    bool processed = false;
    bool stopped = false;
};

} // namespace

int main()
{
    ProbeElement element("probe");

    if (!expect(element.property_specs().size() == 1, "new element did not declare the common name property")) {
        return 1;
    }
    if (!expect(element.properties().size() == 1, "new element did not initialize the common name property")) {
        return 1;
    }
    if (!expect(element.has_property("name"), "new element did not report its name property")) {
        return 1;
    }
    if (!expect(element.property_as<std::string>("name") == std::optional<std::string>{"probe"},
            "new element name property default was wrong")) {
        return 1;
    }
    if (!expect(!element.has_property("poi_count"), "new element reported unknown property")) {
        return 1;
    }
    if (!expect(!element.property_as<std::int64_t>("poi_count").has_value(),
            "property_as returned a value for unknown property")) {
        return 1;
    }

    element.add_property(leakflow::PropertySpec(
        "poi_count",
        std::int64_t{20},
        "number of PoIs",
        "count",
        leakflow::IntRangeConstraint{1, 10000}));
    element.add_property(leakflow::PropertySpec("enabled", true));
    element.add_property(leakflow::PropertySpec("threshold", 0.25, "", "", leakflow::DoubleRangeConstraint{0.0, 1.0}));
    element.add_property(leakflow::PropertySpec(
        "method",
        std::string("pearson"),
        "",
        "",
        leakflow::StringEnumConstraint{{"pearson", "spearman"}}));
    element.add_property(leakflow::PropertySpec("sample_window", leakflow::IntInterval{1000, 2500}));
    element.add_property(leakflow::PropertySpec("time_window", leakflow::DoubleInterval{0.0, 2.5}));
    element.add_property(leakflow::PropertySpec("poi_indexes", leakflow::IntList{12, 40, 91}));
    element.add_property(leakflow::PropertySpec("thresholds", leakflow::DoubleList{0.1, 0.25, 0.5}));
    element.add_property(leakflow::PropertySpec("columns", leakflow::StringList{"traces", "plaintexts", "key"}));

    if (!expect(element.property_specs().size() == 10, "property specs were not stored")) {
        return 1;
    }
    if (!expect(element.properties().size() == 10, "property defaults were not initialized")) {
        return 1;
    }
    if (!expect(element.has_property("poi_count"), "has_property failed for declared property")) {
        return 1;
    }
    if (!expect(element.property_as<std::int64_t>("poi_count") == std::optional<std::int64_t>{20},
            "integer property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<bool>("enabled") == std::optional<bool>{true}, "bool property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<double>("threshold") == std::optional<double>{0.25},
            "double property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<std::string>("method") == std::optional<std::string>{"pearson"},
            "string property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<leakflow::IntInterval>("sample_window")
                == std::optional<leakflow::IntInterval>{leakflow::IntInterval{1000, 2500}},
            "integer interval property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<leakflow::DoubleInterval>("time_window")
                == std::optional<leakflow::DoubleInterval>{leakflow::DoubleInterval{0.0, 2.5}},
            "double interval property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<leakflow::IntList>("poi_indexes")
                == std::optional<leakflow::IntList>{leakflow::IntList{12, 40, 91}},
            "integer list property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<leakflow::DoubleList>("thresholds")
                == std::optional<leakflow::DoubleList>{leakflow::DoubleList{0.1, 0.25, 0.5}},
            "double list property default was wrong")) {
        return 1;
    }
    if (!expect(element.property_as<leakflow::StringList>("columns")
                == std::optional<leakflow::StringList>{leakflow::StringList{"traces", "plaintexts", "key"}},
            "string list property default was wrong")) {
        return 1;
    }

    element.set_property("poi_count", std::int64_t{50});
    element.set_property("enabled", false);
    element.set_property("threshold", 0.5);
    element.set_property("method", std::string("spearman"));
    element.set_property("sample_window", leakflow::IntInterval{2000, 3000});
    element.set_property("time_window", leakflow::DoubleInterval{1.0, 3.0});
    element.set_property("poi_indexes", leakflow::IntList{5, 8});
    element.set_property("thresholds", leakflow::DoubleList{0.2, 0.4});
    element.set_property("columns", leakflow::StringList{"traces"});
    element.set_property("name", std::string("renamed_probe"));

    if (!expect(element.property_as<std::int64_t>("poi_count") == std::optional<std::int64_t>{50},
            "integer property setting failed")) {
        return 1;
    }
    if (!expect(element.name() == "renamed_probe", "name property did not update the element instance name")) {
        return 1;
    }
    if (!expect(element.property_as<std::string>("name") == std::optional<std::string>{"renamed_probe"},
            "name property setting failed")) {
        return 1;
    }
    if (!expect(element.property_as<bool>("enabled") == std::optional<bool>{false}, "bool property setting failed")) {
        return 1;
    }
    if (!expect(element.property_as<double>("threshold") == std::optional<double>{0.5},
            "double property setting failed")) {
        return 1;
    }
    if (!expect(element.property_as<std::string>("method") == std::optional<std::string>{"spearman"},
            "string property setting failed")) {
        return 1;
    }
    if (!expect(element.property_as<std::string>("poi_count") == std::nullopt,
            "property_as returned value for wrong type")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([&element] {
            element.set_property("missing", std::int64_t{1});
        }),
            "unknown property setting was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.set_property("poi_count", std::string("50"));
        }),
            "property type mismatch was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.set_property("poi_count", std::int64_t{0});
        }),
            "integer range violation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.set_property("threshold", 2.0);
        }),
            "double range violation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.set_property("method", std::string("kendall"));
        }),
            "string enum violation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.add_property(leakflow::PropertySpec("poi_count", std::int64_t{20}));
        }),
            "duplicate property declaration was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.add_property(leakflow::PropertySpec("name", std::string("duplicate")));
        }),
            "duplicate common name property declaration was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.set_property("name", std::string());
        }),
            "empty element name was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&element] {
            element.add_property(
                leakflow::PropertySpec("bad_default", std::int64_t{0}, "", "", leakflow::IntRangeConstraint{1, 5}));
        }),
            "invalid default value was not rejected through element declaration")) {
        return 1;
    }

    element.add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps("sca/input")));
    element.add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/output")));
    if (!expect(element.input_pads().size() == 1, "existing input pad behavior changed")) {
        return 1;
    }
    if (!expect(element.output_pads().size() == 1, "existing output pad behavior changed")) {
        return 1;
    }

    element.start();
    (void)element.process(leakflow::Buffer(leakflow::Caps("sca/input")));
    element.stop();
    if (!expect(element.started && element.processed && element.stopped, "existing lifecycle behavior changed")) {
        return 1;
    }

    return 0;
}
