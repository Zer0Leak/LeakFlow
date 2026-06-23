#include "leakflow/core/descriptor_registry.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow {
namespace {

[[nodiscard]] std::string normalized_name(std::string_view name)
{
    std::string normalized;
    normalized.reserve(name.size());

    for (const auto character : name) {
        if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
        }
    }

    return normalized;
}

} // namespace

void DescriptorRegistry::register_plugin(PluginDescriptor plugin)
{
    plugin = with_common_element_properties(std::move(plugin));

    if (plugin.name.empty()) {
        throw std::invalid_argument("plugin descriptor name cannot be empty");
    }
    if (find_plugin(plugin.name) != nullptr) {
        throw std::invalid_argument("duplicate plugin descriptor");
    }

    for (const auto& element : plugin.elements) {
        if (element.type_name.empty()) {
            throw std::invalid_argument("element descriptor type name cannot be empty");
        }
        if (find_element(element.type_name)) {
            throw std::invalid_argument("duplicate element descriptor");
        }
        const auto candidate = normalized_name(element.type_name);
        if (std::ranges::any_of(plugin.elements, [&element, &candidate](const ElementDescriptor& other) {
                return &other != &element && normalized_name(other.type_name) == candidate;
            })) {
            throw std::invalid_argument("duplicate normalized element descriptor");
        }
    }

    plugins_.push_back(std::move(plugin));
}

void DescriptorRegistry::register_plugins(std::vector<PluginDescriptor> plugins)
{
    for (auto& plugin : plugins) {
        register_plugin(std::move(plugin));
    }
}

const std::vector<PluginDescriptor>& DescriptorRegistry::plugins() const
{
    return plugins_;
}

const PluginDescriptor* DescriptorRegistry::find_plugin(std::string_view name) const
{
    const auto found = std::ranges::find_if(plugins_, [name](const PluginDescriptor& plugin) {
        return plugin.name == name;
    });

    if (found == plugins_.end()) {
        return nullptr;
    }

    return &*found;
}

std::optional<ElementDescriptorLookup> DescriptorRegistry::find_element(std::string_view name) const
{
    const auto normalized = normalized_name(name);
    for (const auto& plugin : plugins_) {
        for (const auto& element : plugin.elements) {
            if (element.type_name == name || normalized_name(element.type_name) == normalized) {
                return ElementDescriptorLookup{&plugin, &element};
            }
        }
    }

    return std::nullopt;
}

} // namespace leakflow
