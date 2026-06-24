#pragma once

#include "leakflow/core/caps.hpp"
#include "leakflow/core/element_factory_registry.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/property.hpp"
#include "leakflow/plot/plot_runtime.hpp"

#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace leakflow::cli {

struct PipelineExpressionBuildResult {
    Pipeline pipeline;
    std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime;
};

struct RunExpressionOptions {
    bool graph = false;
    bool auto_start = false; // --auto-start: begin the run on open (else opens Stopped)
};

[[nodiscard]] std::string normalized_identifier(std::string_view text);
[[nodiscard]] PropertyValue parse_property_value(const PropertySpec &spec, std::string_view text);
[[nodiscard]] Caps parse_caps_annotation(std::string_view text);
[[nodiscard]] std::map<std::string, std::string> parse_metadata_annotation(std::string_view text);

[[nodiscard]] ElementFactoryRegistry
builtin_element_factory_registry(std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime);
[[nodiscard]] Pipeline build_pipeline_from_expression(std::string_view expression,
                                                            ElementFactoryRegistry element_factories);
[[nodiscard]] PipelineExpressionBuildResult build_builtin_pipeline_from_expression(std::string_view expression);

int run_expression(std::string_view expression, std::ostream &output, RunExpressionOptions options = {});

} // namespace leakflow::cli
