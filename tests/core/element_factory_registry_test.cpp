#include "leakflow/core/element_factory_registry.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

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
        configure_from_descriptor(descriptor());
    }

    [[nodiscard]] static leakflow::ElementDescriptor descriptor()
    {
        return {
            .type_name = "ProbeElement",
            .purpose = "factory registry probe",
        };
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        return input;
    }
};

[[nodiscard]] leakflow::PluginDescriptor probe_plugin_descriptor()
{
    return {
        .name = "leakflow_plugins_probe",
        .owner = "LeakFlow",
        .purpose = "probe plugin",
        .elements = {
            ProbeElement::descriptor(),
        },
    };
}

} // namespace

int main()
{
    leakflow::ElementFactoryRegistry registry;
    registry.register_plugin(
        probe_plugin_descriptor(),
        {
            leakflow::ElementFactoryRegistration{
                .type_name = "ProbeElement",
                .factory = [](std::string name) {
                    return std::make_shared<ProbeElement>(std::move(name));
                },
            },
        });

    if (!expect(registry.contains("probe-element"), "factory registry did not normalize element names")) {
        return 1;
    }
    if (!expect(registry.descriptors().find_element("probeelement").has_value(),
            "factory registry did not publish descriptors")) {
        return 1;
    }

    auto element = registry.create("probe_element", "probe0");
    if (!expect(element != nullptr, "factory registry did not create an element")) {
        return 1;
    }
    if (!expect(element->name() == "probe0", "factory-created element name was wrong")) {
        return 1;
    }
    if (!expect(element->element_type() == "ProbeElement", "factory-created element type was wrong")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([&registry] {
            (void)registry.create("MissingElement", "missing0");
        }),
            "unknown element factory was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&registry] {
            registry.register_plugin(probe_plugin_descriptor(), {});
        }),
            "plugin registration without matching factories was not rejected")) {
        return 1;
    }

    return 0;
}
