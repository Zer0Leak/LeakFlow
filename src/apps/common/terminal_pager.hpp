#pragma once

#include <string>
#include <string_view>

namespace leakflow::apps {

struct TerminalPagerOptions {
    bool enabled = true;
    std::string less_environment_variable = "LEAKFLOW_LS_LESS";
    std::string default_less_options = "RXF";
};

void present_with_optional_pager(std::string_view output, const TerminalPagerOptions& options = {});

} // namespace leakflow::apps
