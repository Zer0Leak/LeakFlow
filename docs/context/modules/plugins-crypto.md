# Crypto Plugins Context

Use this for pipeline elements that expose crypto/SCA leakage and PoI
capabilities (the AES validation path). For the underlying helper library, read
`docs/context/modules/crypto.md` first.

## Files

Public headers:

- `include/leakflow/plugins/crypto`

Sources:

- `src/plugins/crypto`

Tests:

- `tests/plugins/crypto`

CLI/inspect files if affected:

- `src/apps/leakflow/leakflow_cli.cpp`
- `src/apps/leakflow_ls/main.cpp`
- `src/apps/common`
- `tests/apps`

## Target

- `leakflow_plugins_crypto`

## Dependencies

`leakflow_plugins_crypto` depends on:

- `leakflow_core`,
- `leakflow_base` (Torch payloads + `pearson_correlation`),
- `leakflow_crypto` (Hamming/S-box/AES leakage helpers).

## Current Elements

- `AesLeakage` (`Analyze/SCA/Leakage/AES`): computes AES first-round S-box
  leakage targets. Required `traces` and `plaintexts` sink pads, an optional
  `keys` sink pad (required at runtime for `HW(m xor k)`/`HW(y)`/`y(n)`). Output
  is a Torch `uint8` `[B,N,C]` leakage tensor. Properties: `units` (default
  all 16), `channels` (subset/order of `HW(m)`, `HW(m_xor_k)`, `HW(y)`, and
  `y(0)` through `y(7)`; default `[HW(y)]`; `payload-output`, downstream
  invalidation on `leakage`). Labels the output's leading axis with those bytes as
  `Buffer::units()`, so a downstream per-unit fusion can verify it is scored against
  the same bytes.
- `AesLeakageHypothesis` (`Analyze/SCA/Hypothesis/AES`): computes AES
  first-round predicted leakage hypotheses for every selected byte and guess.
  Required `plaintexts` sink pad. Output is a Torch `uint8` `[U,G,N,L]`
  hypothesis tensor. Properties: `units` (default all 16), `channels`
  (subset/order of `HW(m)`, `HW(m_xor_k)`, `HW(y)`, and `y(0)` through `y(7)`;
  default `[HW(y)]`), `guess_values` (`[]` means full AES byte domain `0..255`;
  `payload-output`, downstream invalidation on `hypotheses`).
- `CpaAttack` (`Analyze/SCA/Attack/CPA`): generic Pearson CPA ranker. Required
  named inputs: `features` (`[N,S]` or `[U,N,S]` Torch tensor) and `hypotheses`
  (`[U,G,N,L]` Torch tensor). Emits `AttackScoresPayload` with `scores [U,G]`,
  `ranking [U,G]`, `best_guess [U]`, best score/channel/sample tensors, guess
  values, and optional `correlations [U,G,L,S]`. Properties include
  `score_method=max_abs|max_positive|max_negative`,
  `score_channels=guess_dependent|all`, `emit_correlations`, `top_k`,
  `correlation_mode=auto|recompute|incremental`, read-only
  `active_correlation_mode`, `compute_dtype=input|float32|float64`, and
  `epsilon`. In incremental active mode, `can_replay()` is false.
- `DpaAttack` (`Analyze/SCA/Attack/DPA`): generic difference-of-means DPA ranker.
  Required named inputs: `features` (`[N,S]` or `[U,N,S]` Torch tensor) and binary
  `hypotheses` (`[U,G,N,L]` Torch tensor, values `0/1`). Emits `scores`
  (`AttackScoresPayload`, same contract as `CpaAttack`) plus optional
  `best_difference` (CPU float32 Torch `[U,S]`, axes `attack_unit,sample`) for
  plotting the current best guess/channel difference trace per unit. `AttackStats`,
  `AttackStatsToPlotAnnotations`, `ScorePlot`, and `ScoreTablePlot` still connect
  to `scores`. The default score is the max absolute
  `mean(group1)-mean(group0)` over channel/sample. Properties include
  `score_method=max_abs|max_positive|max_negative`,
  `score_channels=guess_dependent|all`,
  `accumulation_mode=auto|recompute|incremental`, read-only
  `active_accumulation_mode`, `compute_dtype=input|float32|float64`, and `top_k`.
  In incremental active mode, `can_replay()` is false.
- `AttackStats` (`Analyze/SCA/Evaluation/AttackStats`): known-key diagnostics for an attack
  score result. Inputs: `scores` (`leakflow/attack-scores`) and an **optional**
  `truth` Torch tensor (`[U]`, `[1,U]`, `[16]`, or AES key-like `[N,16]`). Without
  `truth`, the GE/PGE-style fields (`true_rank`, `true_guess`, `true_score`,
  `success`) and the `attack.stats.rank_base`/`success_count` metadata are absent
  (`AttackStatsPayload::has_truth()` is false, `attack.stats.has_truth=false`),
  while every truth-independent statistic (top-K diagnostics, top1/top2 guesses,
  `score_gap`, margins, z-scores, separation, best channel/sample) is still
  produced. Emits `AttackStatsPayload` with optional `true_rank [U]`, `true_score [U]`,
  `top1_guess [U]`, `top2_guess [U]`, `score_gap [U]` (legacy alias for top-1
  relative margin), `success [U]`, and top-K diagnostic tensors such as
  `topk_guess [U,K]`, `topk_score [U,K]`, `topk_margin [U,K]`,
  `topk_relative_margin [U,K]`, `topk_z_score [U,K]`,
  `topk_robust_z_score [U,K]`, and `topk_separation [U,K]`. Properties:
  `top_k` (default 5) and `confidence_metrics` (default
  `[relative_margin,z_score,robust_z_score]`; allowed values are `margin`,
  `relative_margin`, `z_score`, `robust_z_score`, and `top_k_separation`).
- `AttackStatsToPlotAnnotations` (`Convert/PlotAnnotation/AttackStats`): converts
  `AttackStatsPayload` into generic `PlotAnnotationPayload` markers at each
  unit's best sample. Required input is `sink` (`leakflow/attack-stats`).
  Annotations include `success`, `true rank`, `true guess`, `true score`, top-1
  score, relative margin, and a `correct key` field for failed units. Annotation
  `norm_value` is positive for success and negative for failure, making recovered
  and failed units visually separate in `TracePlot`. When the upstream
  `AttackStatsPayload` has no truth (`has_truth()==false`), the truth/success
  fields and `correct key` are omitted, `norm_value` stays positive, and
  `payload.annotation.success_source=none`. Useful as
  `@attack ! @stats.scores; @key ! @stats.truth; @stats ! @ann ! @plot.annotations`.
- `PearsonCorrelator` (`Analyze/SCA/Score/Correlation`): joins the trace branch
  (`features`) and the leakage branch (`targets`) and emits the Pearson correlation of
  every sample against each target as a `CorrelationPayload` (`leakflow/correlation`).
  Properties: `correlation_mode` (auto/recompute/incremental), `compute_dtype`, `epsilon`.
  **Stateful** in incremental mode (`can_replay()==false`); accumulation-property changes
  need a restart. Re-owns the target model's `payload.leakage.*`/`payload.crypto.*` facts.
- `PoiSelect` (`Analyze/SCA/PoI/Select`): selects the top-k PoI sample indexes per
  (byte, channel) from a `CorrelationPayload` and emits a `CorrelationPoiPayload`.
  Properties: `top_k`, `rank_by`. **Stateless** (`can_replay()==true`), so changing
  `top_k`/`rank_by` in Idle re-selects from the cached correlation without re-streaming.
  Stamps `payload.poi.*` metadata. (`PearsonCorrelator` + `PoiSelect` are the split of
  the former `PearsonPoiFinder`.)
- `CorrelationPoiToPlotAnnotations` (`Convert/PlotAnnotation/PoI`): converts a
  `CorrelationPoiPayload` into a generic `PlotAnnotationPayload` for
  `TracePlot.annotations`. Property `precision` (0–12, default 3).

Payload types:

- `CorrelationPoiPayload` (`include/leakflow/plugins/crypto/correlation_poi_payload.hpp`).
- `AttackScoresPayload` and `AttackStatsPayload`
  (`include/leakflow/plugins/crypto/attack_payload.hpp`).

The descriptor catalog assembles the linked plugin descriptor and registers
matching element factories with `ElementFactoryRegistry`. Author/license metadata:

```text
Zer0Leak <edgard.lima@gmail.com>
Apache-2.0
```

## Contracts

`leakflow_core` must not depend on `leakflow_plugins_crypto`. AES knowledge stays
in `leakflow_crypto` (helpers) and this plugin (elements); it must not leak into
core, base, or the generic plugin families.

`PearsonCorrelator`/`PoiSelect` are generic SCA and are expected to move to a future
`leakflow_plugins_sca`; for now only its klass reflects that. See
`docs/design/metadata_klass_taxonomy.md`.

## Correctness

The current AES PoI pipeline (`Hdf5FileSrc` named pads → `Tee`/`AesLeakage` →
`PearsonCorrelator` → `PoiSelect` → `CorrelationPoiToPlotAnnotations` → `TracePlot`) is validated
numerically over the checked-in `key_01`/`key_02` fixtures by
`tests/plugins/crypto/aes_poi_correctness_test.cpp` (Phase 26), not just for
running. See `docs/guides/AES.md` for runnable examples.

## Common Tests

```bash
ctest --test-dir build -R 'leakflow_plugins_crypto|leakflow_cli|leakflow_ls'
```
