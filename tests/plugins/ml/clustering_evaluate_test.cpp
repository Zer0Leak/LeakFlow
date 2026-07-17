#include "leakflow/plugins/ml/clustering_evaluate.hpp"
#include "leakflow/plugins/ml/clustering_evaluation_payload.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/core/summary_document.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <variant>
#include <vector>

namespace {

bool expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

template <typename Function> bool throws_invalid_argument(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

leakflow::Buffer
tensor_buffer(torch::Tensor tensor,
              leakflow::Units units = leakflow::Units::none()) {
  auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
  leakflow::Buffer buffer{payload.caps()};
  buffer.set_units(std::move(units));
  buffer.set_payload(
      std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
  return buffer;
}

leakflow::ElementInputs inputs(leakflow::Buffer labels,
                               leakflow::Buffer truth) {
  leakflow::ElementInputs values;
  values.emplace("labels", std::move(labels));
  values.emplace("truth", std::move(truth));
  return values;
}

} // namespace

int main() {
  namespace ml = leakflow::ml;
  namespace plugin = leakflow::plugins::ml;

  // The new structured evaluator is a separate contract; every computational
  // property invalidates only its evaluation output downstream.
  {
    const auto descriptor = plugin::ClusteringEvaluate::descriptor();
    if (!expect(descriptor.type_name == "ClusteringEvaluate" &&
                    descriptor.input_pads.size() == 2 &&
                    descriptor.output_pads.size() == 1 &&
                    descriptor.output_pads.front().caps().type() ==
                        plugin::clustering_evaluation_caps_type,
                "descriptor pads/caps are wrong")) {
      return 1;
    }
    if (!expect(descriptor.property_specs.size() == 8,
                "descriptor should expose all evaluator properties")) {
      return 1;
    }
    const auto partition_quality = std::ranges::find_if(
        descriptor.property_specs, [](const auto &property) {
          return property.name == "semantic_partition_quality";
        });
    if (!expect(partition_quality != descriptor.property_specs.end() &&
                    std::get<bool>(partition_quality->default_value),
                "semantic partition quality must default on")) {
      return 1;
    }
    for (const auto &property : descriptor.property_specs) {
      if (!expect(property.effect.kind ==
                          leakflow::PropertyEffectKind::PayloadOutput &&
                      property.effect.scope ==
                          leakflow::PropertyInvalidationScope::Downstream &&
                      property.effect.output_pads ==
                          std::vector<std::string>{"evaluation"},
                  "an evaluator property has the wrong dataflow effect")) {
        return 1;
      }
    }
  }

  // Unbatched vector truth, arbitrary labels, producer parameters, and bounded
  // summary all cross the custom payload boundary.
  {
    auto labels = tensor_buffer(torch::tensor({-7, -7, 42, 42}, torch::kLong));
    labels.set_metadata("payload.layout", "unit/observation");
    labels.set_metadata("payload.cluster.method", "gaussian-mixture");
    labels.set_metadata("payload.cluster.covariance_type", "diagonal");
    labels.set_metadata("payload.cluster.n_features", "4");
    labels.set_metadata("payload.parameter.seed", "7");
    auto truth = tensor_buffer(torch::tensor(
        {{1.0, 2.0}, {1.0, 2.0}, {3.0, 4.0}, {3.0, 4.0}}, torch::kFloat64));
    truth.set_metadata("payload.cluster.method", "truth-producer");
    truth.set_metadata("payload.parameter.model", "known-truth");

    plugin::ClusteringEvaluate element;
    const auto output =
        element.process_inputs(inputs(std::move(labels), std::move(truth)));
    if (!expect(output.has_value() &&
                    output->caps().type() ==
                        plugin::clustering_evaluation_caps_type,
                "unbatched evaluation produced no structured output")) {
      return 1;
    }
    const auto payload =
        output->payload_as<plugin::ClusteringEvaluationPayload>();
    if (!expect(payload && payload->layout() == "unit/evaluation" &&
                    payload->result().schema_version == 6 &&
                    output->metadata_or("payload.layout", "") ==
                        "unit/evaluation",
                "structured payload identity/schema is wrong")) {
      return 1;
    }
    const auto &unit = payload->result().units.front();
    if (!expect(
            unit.exact.adjusted_rand_index.value == 1.0 &&
                !unit.semantic_partition_quality.has_value() &&
                payload->parameters().at("labels.cluster.method") ==
                    "gaussian-mixture" &&
                payload->parameters().at("labels.cluster.covariance_type") ==
                    "diagonal" &&
                payload->parameters().at("labels.cluster.n_features") == "4" &&
                payload->parameters().at("evaluation.semantic") == "off" &&
                !payload->parameters().contains("labels.parameter.seed") &&
                !payload->parameters().contains("truth.cluster.method") &&
                !payload->parameters().contains("truth.parameter.model"),
            "unbatched values/parameters are wrong")) {
      return 1;
    }
    leakflow::SummarySection summary("evaluation");
    payload->describe(summary, 3);
    if (!expect(summary.fields.size() < 32,
                "payload summary must remain bounded")) {
      return 1;
    }
  }

  // Evaluator-owned experiment parameters are bounded independently of the
  // numeric dimensionality, including default effective names and weights.
  {
    constexpr std::int64_t dimensions = 64;
    plugin::ClusteringEvaluate element;
    element.set_property("semantic", std::string("power"));
    element.set_property("semantic_ranges",
                         leakflow::DoubleList(dimensions, 1.0));
    const auto output = element.process_inputs(
        inputs(tensor_buffer(torch::zeros({2}, torch::kLong)),
               tensor_buffer(torch::zeros({2, dimensions}, torch::kFloat64))));
    const auto payload =
        output->payload_as<plugin::ClusteringEvaluationPayload>();
    if (!expect(
            payload->parameters().at("evaluation.dimension_names").size() <=
                    512 &&
                payload->parameters().at("evaluation.semantic_ranges").size() <=
                    512 &&
                payload->parameters()
                        .at("evaluation.semantic_weights")
                        .size() <= 512 &&
                payload->result()
                    .effective_options.semantic_partition_quality &&
                payload->result()
                    .units.front()
                    .semantic_partition_quality.has_value() &&
                payload->parameters().at(
                    "evaluation.semantic_partition_quality") == "true",
            "power-mode defaults or evaluator parameter bounds are wrong")) {
      return 1;
    }
  }

  // Even the most detailed summary level caps per-unit headline records.
  {
    constexpr std::int64_t units = 100;
    plugin::ClusteringEvaluate element;
    const auto output = element.process_inputs(
        inputs(tensor_buffer(torch::zeros({units, 2}, torch::kLong)),
               tensor_buffer(torch::zeros({units, 2, 1}, torch::kLong))));
    const auto payload =
        output->payload_as<plugin::ClusteringEvaluationPayload>();
    leakflow::SummarySection summary("evaluation");
    payload->describe(summary, 3);
    if (!expect(summary.fields.size() < 32,
                "detailed payload summary grew with every unit")) {
      return 1;
    }
  }

  // Full semantic evaluation maps every property into the numeric options,
  // retains floating vector truth, computes both alignments, and emits Q.
  {
    const auto labels =
        tensor_buffer(torch::tensor({0, 0, 0, 1, 1, 1}, torch::kLong));
    const auto truth_values = torch::tensor({{0.0, 0.0},
                                             {0.0, 0.0},
                                             {1.0, 1.0},
                                             {1.0, 1.0},
                                             {2.0, 2.0},
                                             {2.0, 2.0}},
                                            torch::kFloat64);
    const auto truth = tensor_buffer(truth_values);
    plugin::ClusteringEvaluate element;
    element.set_property("detail", std::string("full"));
    element.set_property("semantic", std::string("power"));
    element.set_property("dimension_names",
                         leakflow::StringList{"left", "right"});
    element.set_property("semantic_ranges", leakflow::DoubleList{2.0, 2.0});
    element.set_property("semantic_weights", leakflow::DoubleList{3.0, 1.0});
    element.set_property("power", std::int64_t{1});
    element.set_property("alignment", std::string("both"));
    const auto output = element.process_inputs(inputs(labels, truth));
    const auto payload =
        output->payload_as<plugin::ClusteringEvaluationPayload>();
    const auto &result = payload->result();
    const auto &unit = result.units.front();
    if (!expect(result.effective_options.detail ==
                        ml::ClusteringEvaluationDetail::Full &&
                    result.effective_options.semantic ==
                        ml::SemanticEvaluationMode::Power &&
                    result.effective_options.dimension_names ==
                        std::vector<std::string>({"left", "right"}) &&
                    result.effective_options.semantic_ranges ==
                        std::vector<double>({2.0, 2.0}) &&
                    result.effective_options.semantic_weights ==
                        std::vector<double>({3.0, 1.0}) &&
                    result.effective_options.power == 1 &&
                    result.effective_options.alignment ==
                        ml::AlignmentEvaluationMode::Both &&
                    unit.partition_detail.has_value() &&
                    unit.partition_detail->truth_vectors.scalar_type() ==
                        torch::kFloat64 &&
                    unit.exact_alignment.has_value() &&
                    unit.semantic_alignment.has_value() &&
                    unit.semantic_partition_quality.has_value() &&
                    unit.semantic_partition_quality->quality.defined(),
                "full semantic property/result mapping is wrong")) {
      return 1;
    }
  }

  // Semantic partition quality defaults on, but semantic-off evaluation
  // normalizes it inactive instead of emitting a meaningless composite.
  {
    plugin::ClusteringEvaluate element;
    element.set_property("semantic_partition_quality", true);
    const auto labels = tensor_buffer(torch::tensor({0, 0}, torch::kLong));
    const auto truth = tensor_buffer(torch::tensor({{0}, {0}}, torch::kLong));
    const auto output = element.process_inputs(inputs(labels, truth));
    const auto payload =
        output->payload_as<plugin::ClusteringEvaluationPayload>();
    if (!expect(
            !payload->result().effective_options.semantic_partition_quality &&
                !payload->result()
                     .units.front()
                     .semantic_partition_quality.has_value() &&
                payload->parameters().at(
                    "evaluation.semantic_partition_quality") == "false",
            "semantic-off partition quality was not normalized inactive")) {
      return 1;
    }
  }

  // Batched inputs align by typed unit identity, preserving labels order and
  // evaluating only the overlap.
  {
    const auto labels =
        tensor_buffer(torch::tensor({{0, 0, 1, 1}, {0, 1, 0, 1}}, torch::kLong),
                      leakflow::Units::of({2, 1}));
    const auto truth = tensor_buffer(
        torch::tensor({{{0}, {1}, {0}, {1}}, {{0}, {0}, {1}, {1}}},
                      torch::kLong),
        leakflow::Units::of({1, 2}));
    plugin::ClusteringEvaluate element;
    const auto output = element.process_inputs(inputs(labels, truth));
    const auto payload =
        output->payload_as<plugin::ClusteringEvaluationPayload>();
    if (!expect(
            output->units() == leakflow::Units::of({2, 1}) &&
                payload->result().units.size() == 2 &&
                payload->result().units[0].exact.adjusted_rand_index.value ==
                    1.0 &&
                payload->result().units[1].exact.adjusted_rand_index.value ==
                    1.0,
            "typed-unit reordering did not preserve the shared labels order")) {
      return 1;
    }

    const auto partial_labels =
        tensor_buffer(torch::tensor({{0, 0, 1, 1}, {0, 1, 0, 1}}, torch::kLong),
                      leakflow::Units::of({2, 3}));
    const auto partial_truth = tensor_buffer(
        torch::tensor({{{0}, {0}, {1}, {1}}, {{0}, {1}, {0}, {1}}},
                      torch::kLong),
        leakflow::Units::of({2, 1}));
    const auto partial =
        element.process_inputs(inputs(partial_labels, partial_truth));
    if (!expect(partial->units() == leakflow::Units::of({2}) &&
                    partial->payload_as<plugin::ClusteringEvaluationPayload>()
                            ->result()
                            .units.size() == 1,
                "partial typed-unit overlap was not restricted to the shared "
                "unit")) {
      return 1;
    }

    const auto unequal_labels =
        tensor_buffer(torch::tensor({{0, 0, 1, 1}, {0, 1, 0, 1}}, torch::kLong),
                      leakflow::Units::of({2, 3}));
    const auto unequal_truth = tensor_buffer(
        torch::tensor(
            {{{1}, {0}, {1}, {0}}, {{0}, {0}, {1}, {1}}, {{0}, {1}, {0}, {1}}},
            torch::kLong),
        leakflow::Units::of({4, 2, 1}));
    const auto unequal =
        element.process_inputs(inputs(unequal_labels, unequal_truth));
    const auto unequal_payload =
        unequal->payload_as<plugin::ClusteringEvaluationPayload>();
    if (!expect(unequal->units() == leakflow::Units::of({2}) &&
                    unequal_payload->result().units.size() == 1 &&
                    unequal_payload->result()
                            .units.front()
                            .exact.adjusted_rand_index.value == 1.0,
                "typed-unit partial overlap with unequal axis sizes was "
                "rejected or "
                "misaligned")) {
      return 1;
    }
  }

  // Disjoint, one-sided, and malformed unit identities must never silently
  // compare positional rows.
  {
    plugin::ClusteringEvaluate element;
    const auto values = torch::tensor({{0, 0, 1, 1}}, torch::kLong);
    const auto truth_values =
        torch::tensor({{{0}, {0}, {1}, {1}}}, torch::kLong);
    if (!expect(throws_invalid_argument([&] {
                  (void)element.process_inputs(inputs(
                      tensor_buffer(values, leakflow::Units::of({1})),
                      tensor_buffer(truth_values, leakflow::Units::of({2}))));
                }),
                "disjoint units must be rejected")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  (void)element.process_inputs(
                      inputs(tensor_buffer(values, leakflow::Units::of({1})),
                             tensor_buffer(truth_values)));
                }),
                "one-sided unit identity must be rejected")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  (void)element.process_inputs(inputs(
                      tensor_buffer(values, leakflow::Units::of({1, 2})),
                      tensor_buffer(truth_values, leakflow::Units::of({1}))));
                }),
                "malformed unit identity count must be rejected")) {
      return 1;
    }
  }

  // Required named inputs and payload types remain strict.
  {
    plugin::ClusteringEvaluate element;
    if (!expect(throws_invalid_argument(
                    [&] { (void)element.process(std::nullopt); }),
                "single-input entry point must reject use")) {
      return 1;
    }
    leakflow::ElementInputs missing;
    missing.emplace("labels",
                    tensor_buffer(torch::tensor({0, 1}, torch::kLong)));
    if (!expect(throws_invalid_argument(
                    [&] { (void)element.process_inputs(missing); }),
                "missing truth input must be rejected")) {
      return 1;
    }
  }

  return 0;
}
