#pragma once

#include "leakflow/base/statistics.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto pearson_correlation_method_id = "pearson-correlation";

// Computes the Pearson correlation of every feature against every target and emits it
// as a CorrelationPayload (grouped [byte, channel, feature]). This is the *stateful*
// half of the old PearsonPoiFinder: in incremental mode it folds each buffer into
// running moments, so it is NOT replayable (can_replay() is false) and changing an
// accumulation property (correlation_mode / compute_dtype / epsilon) requires a
// restart. The (cheap, stateless) top-k selection lives in PoiSelect.
class PearsonCorrelator final : public Element {
public:
    explicit PearsonCorrelator(std::string name = "pearsoncorrelator0");

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
