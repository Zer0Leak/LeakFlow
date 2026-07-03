#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace leakflow::plugins::crypto {

inline constexpr auto dpa_attack_method_id = "dpa";

class DpaAttack final : public Element {
public:
    explicit DpaAttack(std::string name = "dpaattack0");
    ~DpaAttack() override;

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;
    [[nodiscard]] ElementOutputs process_pads(ElementInputs inputs) override;
    [[nodiscard]] bool can_replay() const override;

private:
    struct IncrementalState;

    void reset_incremental_state();
    void update_active_accumulation_mode();
    void property_changed(std::string_view name) override;
    void live_driven_changed() override;

    std::unique_ptr<IncrementalState> incremental_state_;
    std::optional<std::string> incremental_compute_dtype_;
};

} // namespace leakflow::plugins::crypto
