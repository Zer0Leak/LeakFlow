#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/plot/table_view.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace leakflow::plugins::ml_plot {

// ML-to-plot bridge: translates a structured clustering-evaluation payload
// into generic TableView rows. It never evaluates metrics or alignments.
class ClusteringMetricsTablePlot final : public Element {
public:
  explicit ClusteringMetricsTablePlot(
      std::string name = "clusteringmetricstableplot0");
  ClusteringMetricsTablePlot(std::shared_ptr<leakflow::plot::TableView> view,
                             std::string name = "clusteringmetricstableplot0");

  [[nodiscard]] static ElementDescriptor descriptor();
  void start() override;
  [[nodiscard]] std::optional<Buffer>
  process(std::optional<Buffer> input) override;
  [[nodiscard]] std::optional<Buffer>
  process_inputs(ElementInputs inputs) override;

  [[nodiscard]] std::shared_ptr<leakflow::plot::TableView> table_view() const;
  void set_table_view(std::shared_ptr<leakflow::plot::TableView> view);

protected:
  void property_changed(std::string_view name) override;

private:
  std::shared_ptr<leakflow::plot::TableView> view_;
  std::int64_t run_sequence_ = 0;
};

} // namespace leakflow::plugins::ml_plot
