#include "leakflow/core/pipeline_observer.hpp"

#include <string>
#include <string_view>

namespace leakflow {

std::string pipeline_link_id(std::string_view source_element_name, std::string_view source_pad_name,
                             std::string_view sink_element_name, std::string_view sink_pad_name) {
    auto id = std::string(source_element_name);
    id += '.';
    id += source_pad_name;
    id += " -> ";
    id += sink_element_name;
    id += '.';
    id += sink_pad_name;
    return id;
}

} // namespace leakflow
