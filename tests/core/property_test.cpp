#include "leakflow/core/property.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

template <typename Function>
bool throws_invalid_argument(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

} // namespace

int main()
{
    leakflow::PropertySpec enabled("enabled", true, "enable processing");
    if (!expect(std::get<bool>(enabled.default_value), "bool default was not stored")) {
        return 1;
    }
    leakflow::validate_property_value(enabled, false);

    leakflow::PropertySpec poi_count(
        "poi_count",
        std::int64_t{20},
        "number of PoIs",
        "count",
        leakflow::IntRangeConstraint{1, 10000});
    if (!expect(std::get<std::int64_t>(poi_count.default_value) == 20, "integer default was not stored")) {
        return 1;
    }
    leakflow::validate_property_value(poi_count, std::int64_t{100});
    if (!expect(throws_invalid_argument([&poi_count] {
            leakflow::validate_property_value(poi_count, std::int64_t{0});
        }),
            "integer range did not reject invalid value")) {
        return 1;
    }

    leakflow::PropertySpec threshold(
        "threshold",
        0.25,
        "score threshold",
        "score",
        leakflow::DoubleRangeConstraint{0.0, 1.0});
    if (!expect(std::get<double>(threshold.default_value) == 0.25, "double default was not stored")) {
        return 1;
    }
    leakflow::validate_property_value(threshold, 0.5);
    if (!expect(throws_invalid_argument([&threshold] {
            leakflow::validate_property_value(threshold, 1.5);
        }),
            "double range did not reject invalid value")) {
        return 1;
    }

    leakflow::PropertySpec method(
        "method",
        std::string("pearson"),
        "correlation method",
        "",
        leakflow::StringEnumConstraint{{"pearson", "spearman"}});
    if (!expect(std::get<std::string>(method.default_value) == "pearson", "string default was not stored")) {
        return 1;
    }
    leakflow::validate_property_value(method, std::string("spearman"));
    if (!expect(throws_invalid_argument([&method] {
            leakflow::validate_property_value(method, std::string("kendall"));
        }),
            "string enum did not reject invalid value")) {
        return 1;
    }
    if (!expect(property_value_type_name(method.default_value) == "string",
            "string enum became a separate value type")) {
        return 1;
    }

    leakflow::PropertySpec sample_window("sample_window", leakflow::IntInterval{1000, 2500});
    if (!expect(std::get<leakflow::IntInterval>(sample_window.default_value) == leakflow::IntInterval{1000, 2500},
            "integer interval value was not stored")) {
        return 1;
    }

    leakflow::PropertySpec time_window("time_window", leakflow::DoubleInterval{0.0, 2.5});
    if (!expect(std::get<leakflow::DoubleInterval>(time_window.default_value) == leakflow::DoubleInterval{0.0, 2.5},
            "double interval value was not stored")) {
        return 1;
    }

    leakflow::PropertySpec poi_indexes("poi_indexes", leakflow::IntList{12, 40, 91});
    if (!expect(std::get<leakflow::IntList>(poi_indexes.default_value) == leakflow::IntList{12, 40, 91},
            "integer list value was not stored")) {
        return 1;
    }

    leakflow::PropertySpec thresholds("thresholds", leakflow::DoubleList{0.1, 0.25, 0.5});
    if (!expect(std::get<leakflow::DoubleList>(thresholds.default_value) == leakflow::DoubleList{0.1, 0.25, 0.5},
            "double list value was not stored")) {
        return 1;
    }

    leakflow::PropertySpec columns("columns", leakflow::StringList{"traces", "plaintexts", "key"});
    if (!expect(std::get<leakflow::StringList>(columns.default_value)
                == leakflow::StringList{"traces", "plaintexts", "key"},
            "string list value was not stored")) {
        return 1;
    }

    if (!expect(property_value_to_string(leakflow::IntInterval{4, 8}) == "4..8",
            "integer interval string form was wrong")) {
        return 1;
    }
    if (!expect(leakflow::property_value_to_string(leakflow::IntList{1, 2, 3}) == "[1,2,3]",
            "integer list string form was wrong")) {
        return 1;
    }

    leakflow::PropertySpec channels(
        "channels",
        leakflow::StringList{"HW(y)"},
        "leakage channels",
        "",
        std::monostate{},
        "",
        leakflow::PropertyEffect{
            .kind = leakflow::PropertyEffectKind::PayloadOutput,
            .scope = leakflow::PropertyInvalidationScope::Downstream,
            .output_pads = {"leakage"},
        });
    if (!expect(channels.effect.kind == leakflow::PropertyEffectKind::PayloadOutput,
            "property effect kind was not stored")) {
        return 1;
    }
    if (!expect(channels.effect.scope == leakflow::PropertyInvalidationScope::Downstream,
            "property invalidation scope was not stored")) {
        return 1;
    }
    if (!expect(channels.effect.output_pads == std::vector<std::string>{"leakage"},
            "property effect output pads were not stored")) {
        return 1;
    }
    if (!expect(leakflow::property_effect_kind_name(channels.effect.kind) == "payload-output",
            "property effect kind name was wrong")) {
        return 1;
    }
    if (!expect(leakflow::property_invalidation_scope_name(channels.effect.scope) == "downstream",
            "property invalidation scope name was wrong")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            leakflow::PropertySpec("", std::int64_t{1});
        }),
            "empty property name was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            leakflow::PropertySpec("bad_range", std::int64_t{1}, "", "", leakflow::IntRangeConstraint{5, 1});
        }),
            "invalid integer range shape was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            leakflow::PropertySpec("bad_type", std::string("x"), "", "", leakflow::IntRangeConstraint{1, 5});
        }),
            "constraint/default type mismatch was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            leakflow::PropertySpec("bad_default", std::int64_t{0}, "", "", leakflow::IntRangeConstraint{1, 5});
        }),
            "invalid default value was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            leakflow::PropertySpec(
                "bad_effect",
                std::int64_t{1},
                "",
                "",
                std::monostate{},
                "",
                leakflow::PropertyEffect{
                    .kind = leakflow::PropertyEffectKind::PayloadOutput,
                    .scope = leakflow::PropertyInvalidationScope::None,
                });
        }),
            "output property effect without downstream invalidation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            leakflow::PropertySpec(
                "bad_pad",
                std::int64_t{1},
                "",
                "",
                std::monostate{},
                "",
                leakflow::PropertyEffect{
                    .kind = leakflow::PropertyEffectKind::PayloadOutput,
                    .scope = leakflow::PropertyInvalidationScope::Downstream,
                    .output_pads = {""},
                });
        }),
            "empty effect output pad name was not rejected")) {
        return 1;
    }

    return 0;
}
