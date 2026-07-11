#pragma once

#include "leakflow/plot/plot_view.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace leakflow::plot {

// Generic two-row comparison table (domain-free): per (unit, channel), a header row of column
// labels plus a "reference" row and a "current" row. The element supplies both pre-formatted
// strings (for the cells, "-" where a side is absent) and the raw numeric values (NaN where
// absent) that the view uses to sort the columns, highlight the stronger row in each column,
// and draw a line under the table. Unit and channel sliders pick which table is shown; a metric toggle
// (value / |value|) and a sort selector (sample index / reference / current) live in the view.
// A PoiTablePlot element fills this (comparing a profiling PoI set against a re-scored one);
// PoiTableView renders it with ImGui/ImPlot, knowing nothing about correlations.
struct PoiTableGroup {
    std::vector<std::string> columns;   // header cells (e.g. sample indexes, or 1..N)
    std::vector<std::string> reference; // values or "-" (absent input)
    std::vector<std::string> current;   // values or "-"
    std::vector<double> sample;         // numeric column key (sample index, or 1..N) for sorting/x
    std::vector<double> reference_values; // numeric scores; NaN where the side is absent
    std::vector<double> current_values;   // numeric scores; NaN where the side is absent
    bool has_reference = false;         // reference input contributed this row
    bool has_current = false;           // current input contributed this row
};

struct PoiTableSnapshot {
    std::uint64_t id = 0;
    std::string element_name;
    std::string title;
    std::string reference_label = "reference";
    std::string current_label = "current";
    std::vector<std::int64_t> unit_ids;      // [U]
    std::vector<std::string> channel_labels; // [C]
    std::vector<PoiTableGroup> groups;       // [U * C], unit-major: groups[unit * C + channel]
};

class PoiTableView final : public PlotView {
public:
    PoiTableView() = default;

    // Worker-thread: replace the table owned by element_name (find-or-create the snapshot).
    void set_table(
        std::string element_name,
        std::string title,
        std::string reference_label,
        std::string current_label,
        std::vector<std::int64_t> unit_ids,
        std::vector<std::string> channel_labels,
        std::vector<PoiTableGroup> groups);

    [[nodiscard]] const std::vector<PoiTableSnapshot>& snapshots() const;

    // PlotView:
    void draw(const PlotDrawContext& context) override;
    void clear() override;
    [[nodiscard]] bool empty() const override;

private:
    mutable std::recursive_mutex mutex_;
    std::vector<PoiTableSnapshot> snapshots_;
    std::map<std::string, int> selected_unit_;    // per element, preserved across updates
    std::map<std::string, int> selected_channel_; // per element
    std::map<std::string, int> use_abs_;          // per element: 0 = signed value, 1 = |value|
    std::map<std::string, int> sort_mode_;        // per element: 0 = sample, 1 = reference, 2 = current
    std::uint64_t next_id_ = 1;
};

} // namespace leakflow::plot
