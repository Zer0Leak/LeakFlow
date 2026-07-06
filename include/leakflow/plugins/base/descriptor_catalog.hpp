#pragma once

#include "leakflow/core/descriptor.hpp"
#include "leakflow/core/descriptor_registry.hpp"
#include "leakflow/core/element_factory_registry.hpp"
#include "leakflow/core/payload_codec.hpp"

#include <string_view>
#include <vector>

namespace leakflow::plugins::base {

inline constexpr auto plugin_name = "leakflow_plugins_base";

[[nodiscard]] std::vector<PluginDescriptor> plugin_descriptors();
[[nodiscard]] const PluginDescriptor* find_plugin_descriptor(std::string_view name);
void register_plugin_descriptors(DescriptorRegistry& registry);
void register_element_factories(ElementFactoryRegistry& registry);
// Register the base payload codecs (TorchTensorPayload) for BufferFileSink/Src.
void register_payload_codecs(PayloadCodecRegistry& codecs);

} // namespace leakflow::plugins::base
