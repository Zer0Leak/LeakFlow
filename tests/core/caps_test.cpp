#include "leakflow/core/caps.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

} // namespace

int main()
{
    leakflow::Caps empty_caps("sca/traceset");
    if (!expect(empty_caps.type() == "sca/traceset", "type construction failed")) {
        return 1;
    }
    if (!expect(empty_caps.params().empty(), "empty params expected")) {
        return 1;
    }

    leakflow::Caps caps("sca/traceset", {
        {"sample_count", "5000"},
        {"dtype", "float32"},
        {"layout", "trace,sample"},
    });

    if (!expect(caps.type() == "sca/traceset", "parameter construction changed type")) {
        return 1;
    }
    if (!expect(caps.has_param("dtype"), "has_param failed for existing key")) {
        return 1;
    }
    if (!expect(!caps.has_param("missing"), "has_param succeeded for missing key")) {
        return 1;
    }
    if (!expect(caps.param("dtype") == "float32", "param returned wrong value")) {
        return 1;
    }
    if (!expect(caps.param_or("missing", "fallback") == "fallback", "param_or returned wrong fallback")) {
        return 1;
    }

    caps.set_param("channel", "power");
    if (!expect(caps.param("channel") == "power", "set_param failed")) {
        return 1;
    }

    bool missing_threw = false;
    try {
        (void)caps.param("missing");
    } catch (const std::out_of_range&) {
        missing_threw = true;
    }
    if (!expect(missing_threw, "missing param did not throw")) {
        return 1;
    }

    const std::string expected = "sca/traceset; channel=power; dtype=float32; layout=trace,sample; sample_count=5000";
    if (!expect(caps.to_string() == expected, "to_string output was not deterministic")) {
        return 1;
    }

    leakflow::Caps float_source("leakflow/torch-tensor", {
        {"device", "cpu"},
        {"dtype", "float32"},
    });
    leakflow::Caps float_sink("leakflow/torch-tensor", {
        {"dtype", "float32"},
    });
    if (!expect(leakflow::caps_are_compatible(float_source, float_sink),
            "matching caps parameters should be compatible")) {
        return 1;
    }

    leakflow::Caps double_sink("leakflow/torch-tensor", {
        {"dtype", "float64"},
    });
    if (!expect(!leakflow::caps_are_compatible(float_source, double_sink),
            "conflicting caps parameters should be incompatible")) {
        return 1;
    }

    leakflow::Caps unspecified_source("leakflow/torch-tensor");
    if (!expect(leakflow::caps_are_compatible(unspecified_source, float_sink),
            "missing source caps parameter should remain unspecified, not incompatible")) {
        return 1;
    }

    leakflow::Caps generic_sink(leakflow::generic_buffer_caps_type, {
        {"dtype", "float32"},
    });
    if (!expect(leakflow::caps_are_compatible(float_source, generic_sink),
            "concrete caps should remain compatible with generic buffer sinks")) {
        return 1;
    }

    leakflow::Caps numpy_sink("leakflow/numpy-array");
    if (!expect(!leakflow::caps_are_compatible(float_source, numpy_sink),
            "different concrete caps types should be incompatible")) {
        return 1;
    }

    leakflow::Caps any_caps(leakflow::any_caps_type);
    if (!expect(leakflow::caps_are_any(any_caps), "ANY caps were not recognized")) {
        return 1;
    }
    if (!expect(leakflow::caps_are_compatible(any_caps, numpy_sink),
            "ANY source caps should be compatible with concrete sinks")) {
        return 1;
    }
    if (!expect(leakflow::caps_are_compatible(float_source, any_caps),
            "concrete source caps should be compatible with ANY sinks")) {
        return 1;
    }
    if (!expect(any_caps.to_string() == "ANY", "ANY caps string form changed")) {
        return 1;
    }

    return 0;
}
