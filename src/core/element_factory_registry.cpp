#include "leakflow/core/element_factory_registry.hpp"

#include <set>
#include <stdexcept>
#include <utility>

namespace leakflow {

void ElementFactoryRegistry::register_plugin(
    PluginDescriptor plugin,
    std::vector<ElementFactoryRegistration> factories)
{
    std::set<std::string> plugin_types;
    for (const auto& element : plugin.elements) {
        plugin_types.insert(element_default_name_prefix(element.type_name));
    }

    std::set<std::string> factory_types;
    for (const auto& registration : factories) {
        if (registration.type_name.empty()) {
            throw std::invalid_argument("element factory type name cannot be empty");
        }
        if (!registration.factory) {
            throw std::invalid_argument("element factory cannot be empty");
        }

        const auto normalized = element_default_name_prefix(registration.type_name);
        if (!plugin_types.contains(normalized)) {
            throw std::invalid_argument("element factory has no matching plugin descriptor");
        }
        if (!factory_types.insert(normalized).second) {
            throw std::invalid_argument("duplicate element factory in plugin registration");
        }
        if (factories_.contains(normalized)) {
            throw std::invalid_argument("duplicate element factory");
        }
    }

    for (const auto& type : plugin_types) {
        if (!factory_types.contains(type)) {
            throw std::invalid_argument("plugin descriptor has no matching element factory");
        }
    }

    descriptors_.register_plugin(std::move(plugin));
    for (auto& registration : factories) {
        factories_.emplace(element_default_name_prefix(registration.type_name), std::move(registration.factory));
    }
}

const DescriptorRegistry& ElementFactoryRegistry::descriptors() const
{
    return descriptors_;
}

bool ElementFactoryRegistry::contains(std::string_view type_name) const
{
    return factories_.contains(element_default_name_prefix(type_name));
}

std::shared_ptr<Element> ElementFactoryRegistry::create(std::string_view type_name, std::string name) const
{
    const auto found = factories_.find(element_default_name_prefix(type_name));
    if (found == factories_.end()) {
        throw std::invalid_argument("unknown element type");
    }

    return found->second(std::move(name));
}

} // namespace leakflow
