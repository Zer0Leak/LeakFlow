#pragma once

#include "leakflow/base/statistics.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto pearson_poi_method_id = "pearson-correlation";

class PearsonPoiFinder final : public Element {
public:
    explicit PearsonPoiFinder(std::string name = "pearsonpoifinder0");

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;
    [[nodiscard]] bool can_replay() const override;

private:
    void reset_incremental_correlation();
    void update_active_correlation_mode();
    void property_changed(std::string_view name) override;
    void live_driven_changed() override;

    std::optional<leakflow::base::InteractivePearsonCorrelation> incremental_correlation_;
    std::optional<leakflow::base::PearsonCorrelationOptions> incremental_options_;
};

} // namespace leakflow::plugins::crypto
