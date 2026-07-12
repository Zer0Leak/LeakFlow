#pragma once

#include "leakflow/core/descriptor.hpp"
#include "leakflow/core/descriptor_registry.hpp"
#include "leakflow/core/element_factory_registry.hpp"
#include "leakflow/core/payload_codec.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace leakflow::plugins::extras {

inline constexpr auto plugin_name = "leakflow_plugins_extras";

[[nodiscard]] std::vector<PluginDescriptor> plugin_descriptors();
[[nodiscard]] const PluginDescriptor* find_plugin_descriptor(std::string_view name);
void register_plugin_descriptors(DescriptorRegistry& registry);
// `codecs` is injected into the BufferFileSink/BufferFileSrc factories (populated by
// the linking application with the base/crypto payload codecs). A null registry
// yields an empty one -- those two elements then build but reject unknown payloads.
void register_element_factories(ElementFactoryRegistry& registry,
                                std::shared_ptr<const PayloadCodecRegistry> codecs = {});

} // namespace leakflow::plugins::extras
