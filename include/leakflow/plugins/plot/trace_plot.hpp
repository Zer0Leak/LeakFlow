#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/plot/plot_runtime.hpp"

#include <memory>
#include <optional>
#include <string>

namespace leakflow::plugins::plot {

class TracePlot final : public Element {
  public:
    explicit TracePlot(std::string name = "traceplot0");
    TracePlot(std::shared_ptr<leakflow::plot::PlotRuntime> runtime, std::string name = "traceplot0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;

    [[nodiscard]] std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime() const;
    void set_plot_runtime(std::shared_ptr<leakflow::plot::PlotRuntime> runtime);

  private:
    std::shared_ptr<leakflow::plot::PlotRuntime> runtime_;
};

} // namespace leakflow::plugins::plot
