#pragma once

#include "leakflow/core/descriptor.hpp"

#include <optional>
#include <string_view>
#include <vector>

namespace leakflow {

struct ElementDescriptorLookup {
    const PluginDescriptor* plugin = nullptr;
    const ElementDescriptor* element = nullptr;
};

class DescriptorRegistry {
public:
    void register_plugin(PluginDescriptor plugin);
    void register_plugins(std::vector<PluginDescriptor> plugins);

    [[nodiscard]] const std::vector<PluginDescriptor>& plugins() const;
    [[nodiscard]] const PluginDescriptor* find_plugin(std::string_view name) const;
    [[nodiscard]] std::optional<ElementDescriptorLookup> find_element(std::string_view name) const;

private:
    std::vector<PluginDescriptor> plugins_;
};

} // namespace leakflow
