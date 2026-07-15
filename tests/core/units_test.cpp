#include "leakflow/core/units.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

void expect_format(const leakflow::Units& labels, const std::string& expected)
{
    expect(labels.format() == expected, "format was '" + labels.format() + "', expected '" + expected + "'");
}

} // namespace

int main()
{
    using leakflow::Units;

    // none
    expect(Units::none().empty(), "none was not empty");
    expect(Units::none().size() == 0, "none size was not 0");
    expect_format(Units::none(), "none");
    expect(Units::parse("none").empty() && Units::parse("").empty(), "none did not parse from none/empty");
    expect(Units::of({}).empty(), "of({}) was not none");
    expect(Units::range(0, 0).empty(), "range with count 0 was not none");

    // single
    expect_format(Units::of({3}), "[3]");
    expect(Units::of({3}).size() == 1 && Units::of({3}).at(0) == 3, "single unit wrong");
    expect_format(Units::range(3, 1), "[3]");

    // full contiguous range (the ex-"all" case), second number exclusive
    const auto full = Units::range(0, 16);
    expect_format(full, "[0:16]");
    expect(full.size() == 16 && full.at(0) == 0 && full.at(15) == 15, "range size/at wrong");
    expect(full.to_vector().size() == 16 && full.to_vector().back() == 15, "range to_vector wrong");

    // a plain contiguous list collapses to a range
    expect_format(Units::of({0, 1, 2}), "[0:3]");

    // gappy set with an internal run collapses per run, order preserved
    expect_format(Units::of({0, 1, 4, 5, 6}), "[0:2,4:7]");
    expect_format(Units::of({3, 5}), "[3,5]");

    // reordered set keeps its order (row i is values[i])
    const auto reordered = Units::of({5, 3});
    expect_format(reordered, "[5,3]");
    expect(reordered.at(0) == 5 && reordered.at(1) == 3, "reordered at() wrong");

    // parse round-trips through the shared grammar
    expect(Units::parse("[0:16]") == Units::range(0, 16), "parse [0:16] wrong");
    expect(Units::parse("[0]") == Units::range(0, 1), "parse [0] wrong");
    expect(Units::parse("[3,5]") == Units::of({3, 5}), "parse [3,5] wrong");
    expect(Units::parse("[5,3]") == Units::of({5, 3}), "parse [5,3] wrong");
    // items expand and re-canonicalise: {0,1,2,3,4} is contiguous -> a range
    expect(Units::parse("[0,1,2:5]") == Units::range(0, 5), "parse [0,1,2:5] wrong");
    expect_format(Units::parse("[0,1,2:5]"), "[0:5]");

    // equality distinguishes range from a same-valued explicit only when values differ
    expect(Units::range(0, 3) == Units::of({0, 1, 2}), "range vs contiguous of() should be equal");
    expect(Units::of({5, 3}) != Units::of({3, 5}), "different order should not be equal");

    if (failures == 0) {
        std::cout << "units tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
