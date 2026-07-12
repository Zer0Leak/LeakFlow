#include "leakflow/base/plot_annotation_payload.hpp"

#include "leakflow/core/buffer.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
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
    leakflow::base::PlotAnnotationPayload payload({
        leakflow::base::PlotAnnotation{
            .sample_index = 42,
            .value = 0.875,
            .fields = {{"key byte", "3"}, {"target", "HW(y)"}},
            .label = "byte_3.HW(y)",
            .text = "byte_3.HW(y): 0.875",
            .kind = "poi",
        },
        leakflow::base::PlotAnnotation{
            .sample_index = 42,
            .norm_value = -0.5,
            .fields = {{"key byte", "5"}, {"target", "HW(m)"}},
            .label = "byte_5.HW(m)",
            .kind = "poi",
            .target_index = 2,
        },
        leakflow::base::PlotAnnotation{
            .sample_index = 7,
            .fields = {{"target", "combined"}},
            .kind = "poi",
        },
    });

    if (!expect(payload.type_name() == leakflow::base::plot_annotation_caps_type,
            "PlotAnnotationPayload type name was wrong")) {
        return 1;
    }
    const auto annotation_layout =
        "annotation/[sample_index,value?,norm_value?,fields,label,text,kind,target_index?,marker]";
    if (!expect(payload.layout() == annotation_layout,
            "PlotAnnotationPayload layout was wrong")) {
        return 1;
    }
    if (!expect(payload.annotation_count() == 3, "PlotAnnotationPayload annotation count was wrong")) {
        return 1;
    }
    if (!expect(payload.annotation(0).sample_index == 42, "PlotAnnotationPayload sample_index was wrong")) {
        return 1;
    }
    if (!expect(payload.annotation(0).value && *payload.annotation(0).value == 0.875,
            "PlotAnnotationPayload value was wrong")) {
        return 1;
    }
    if (!expect(!payload.annotation(0).norm_value, "PlotAnnotationPayload exact-value annotation had norm_value")) {
        return 1;
    }
    if (!expect(payload.annotation(0).fields.size() == 2, "PlotAnnotationPayload field count was wrong")) {
        return 1;
    }
    if (!expect(payload.annotation(0).fields[0].first == "key byte"
                && payload.annotation(0).fields[0].second == "3",
            "PlotAnnotationPayload ordered first field was wrong")) {
        return 1;
    }
    if (!expect(payload.annotation(0).fields[1].first == "target"
                && payload.annotation(0).fields[1].second == "HW(y)",
            "PlotAnnotationPayload ordered second field was wrong")) {
        return 1;
    }
    if (!expect(payload.annotation(1).label == "byte_5.HW(m)", "PlotAnnotationPayload label was wrong")) {
        return 1;
    }
    if (!expect(payload.annotation(1).norm_value && *payload.annotation(1).norm_value == -0.5,
            "PlotAnnotationPayload norm_value was wrong")) {
        return 1;
    }
    if (!expect(!payload.annotation(2).value && !payload.annotation(2).norm_value,
            "PlotAnnotationPayload top annotation should not have y values")) {
        return 1;
    }
    if (!expect(payload.annotation(1).target_index && *payload.annotation(1).target_index == 2,
            "PlotAnnotationPayload target index was wrong")) {
        return 1;
    }
    if (!expect(payload.caps().type() == leakflow::base::plot_annotation_caps_type,
            "PlotAnnotationPayload caps were wrong")) {
        return 1;
    }

    // Level 3 lists every annotation; the fixture has three (payload + count + 3 annotations).
    leakflow::SummarySection full_summary("Payload");
    payload.describe(full_summary, 3);
    if (!expect(full_summary.fields.size() == 5,
            "PlotAnnotationPayload level-3 summary should list payload, count, and all annotations")) {
        return 1;
    }
    // Every PlotAnnotation field is printed unconditionally. For annotation 0 that is
    // sample_index, value, norm_value (none), two custom fields, label, text, kind, and
    // target_index (none) -> nine children.
    if (!expect(full_summary.fields[2].label == "annotation[0]" && full_summary.fields[2].children.size() == 9,
            "PlotAnnotationPayload should print every field of an annotation")) {
        return 1;
    }
    if (!expect(full_summary.fields[2].children[0].label == "sample_index",
            "PlotAnnotationPayload should label sample_index")) {
        return 1;
    }
    if (!expect(full_summary.fields[2].children[2].label == "norm_value"
                && full_summary.fields[2].children[2].value.text == "none",
            "PlotAnnotationPayload should render an absent optional field as none")) {
        return 1;
    }

    // Levels 1 and 2 show the count and only the first annotation.
    leakflow::SummarySection level1_summary("Payload");
    payload.describe(level1_summary, 1);
    leakflow::SummarySection level2_summary("Payload");
    payload.describe(level2_summary, 2);
    if (!expect(level1_summary.fields.size() == 3 && level2_summary.fields.size() == 3,
            "PlotAnnotationPayload levels 1-2 should show the count and only the first annotation")) {
        return 1;
    }
    if (!expect(level1_summary.fields[1].label == "annotations"
                && level1_summary.fields[2].label == "annotation[0]"
                && !level1_summary.fields[2].children.empty(),
            "PlotAnnotationPayload level-1 summary should report the count and first annotation")) {
        return 1;
    }
    if (!expect(level1_summary.fields.size() < full_summary.fields.size(),
            "PlotAnnotationPayload level-1 summary should be shorter than the level-3 summary")) {
        return 1;
    }

    // Level 0 shows only the count.
    leakflow::SummarySection identity_summary("Payload");
    payload.describe(identity_summary, 0);
    if (!expect(identity_summary.fields.size() == 2,
            "PlotAnnotationPayload level-0 summary should show only payload and count")) {
        return 1;
    }

    leakflow::Buffer buffer(payload.caps());
    buffer.set_payload(std::make_shared<leakflow::base::PlotAnnotationPayload>(payload));
    if (!expect(buffer.payload_as<leakflow::base::PlotAnnotationPayload>() != nullptr,
            "Buffer did not preserve PlotAnnotationPayload type")) {
        return 1;
    }
    if (!expect(buffer.metadata("payload.layout") == annotation_layout,
            "Buffer did not publish PlotAnnotationPayload layout")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            leakflow::base::PlotAnnotationPayload invalid({
                leakflow::base::PlotAnnotation{
                    .sample_index = -1,
                    .value = 1.0,
                    .kind = "poi",
                },
            });
        }),
            "PlotAnnotationPayload accepted negative sample_index")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            leakflow::base::PlotAnnotationPayload invalid({
                leakflow::base::PlotAnnotation{
                    .sample_index = 0,
                    .value = 1.0,
                },
            });
        }),
            "PlotAnnotationPayload accepted empty annotation kind")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            leakflow::base::PlotAnnotationPayload invalid({
                leakflow::base::PlotAnnotation{
                    .sample_index = 0,
                    .norm_value = 1.5,
                    .kind = "poi",
                },
            });
        }),
            "PlotAnnotationPayload accepted out-of-range norm_value")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            leakflow::base::PlotAnnotationPayload invalid({
                leakflow::base::PlotAnnotation{
                    .sample_index = 0,
                    .norm_value = -1.5,
                    .kind = "poi",
                },
            });
        }),
            "PlotAnnotationPayload accepted below-range norm_value")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            leakflow::base::PlotAnnotationPayload invalid({
                leakflow::base::PlotAnnotation{
                    .sample_index = 0,
                    .fields = {{"", "value"}},
                    .kind = "poi",
                },
            });
        }),
            "PlotAnnotationPayload accepted empty field key")) {
        return 1;
    }

    return 0;
}
