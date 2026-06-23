#pragma once

#include "leakflow/core/descriptor_registry.hpp"
#include "leakflow/core/element.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

using ElementFactory = std::function<std::shared_ptr<Element>(std::string name)>;

struct ElementFactoryRegistration {
    std::string type_name;
    ElementFactory factory;
};

class ElementFactoryRegistry {
public:
    void register_plugin(PluginDescriptor plugin, std::vector<ElementFactoryRegistration> factories);

    [[nodiscard]] const DescriptorRegistry& descriptors() const;
    [[nodiscard]] bool contains(std::string_view type_name) const;
    [[nodiscard]] std::shared_ptr<Element> create(std::string_view type_name, std::string name) const;

private:
    DescriptorRegistry descriptors_;
    std::map<std::string, ElementFactory> factories_;
};

} // namespace leakflow
