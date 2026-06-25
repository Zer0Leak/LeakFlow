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

- `AesLeakage` (`Analyze/SCA/Crypto/LeakageModel`): computes AES first-round S-box
  Hamming-weight leakage targets. Required `traces` and `plaintexts` sink pads, an
  optional `keys` sink pad (required at runtime for `HW(m xor k)`/`HW(y)`). Output
  is a Torch `uint8` `[B,N,C]` leakage tensor. Properties: `byte_indexes` (default
  all 16), `channels` (subset/order of `HW(m)`, `HW(m_xor_k)`, `HW(y)`; default
  `[HW(y)]`; `payload-output`, downstream invalidation on `leakage`).
- `PearsonPoiFinder` (`Analyze/SCA/Statistics/PoI`): joins the trace branch
  (`features`) and the leakage branch (`targets`) and selects per-target PoI sample
  indexes by Pearson correlation. Emits a `CorrelationPoiPayload`. Properties
  include `top_k` and `rank_by`. Re-owns the target model's
  `payload.leakage.*`/`payload.crypto.*` facts and stamps `payload.poi.*` and
  per-target `poi.target.N.*` metadata.
- `CorrelationPoiToPlotAnnotations` (`Convert/SCA/Plot/Annotations`): converts a
  `CorrelationPoiPayload` into a generic `PlotAnnotationPayload` for
  `TracePlot.annotations`. Property `precision` (0–12, default 3).

Payload type: `CorrelationPoiPayload` (`include/leakflow/plugins/crypto/correlation_poi_payload.hpp`).

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

`PearsonPoiFinder` is generic SCA and is expected to move to a future
`leakflow_plugins_sca`; for now only its klass reflects that. See
`docs/design/metadata_klass_taxonomy.md`.

## Correctness

The AES PoI pipeline (`TorchFileSrc` ×3 → `Tee` → `AesLeakage` →
`PearsonPoiFinder` → `CorrelationPoiToPlotAnnotations` → `TracePlot`) is validated
numerically over the checked-in `key_01`/`key_02` fixtures by
`tests/plugins/crypto/aes_poi_correctness_test.cpp` (Phase 26), not just for
running. See `docs/guides/AES.md` for runnable examples.

## Common Tests

```bash
ctest --test-dir build -R 'leakflow_plugins_crypto|leakflow_cli|leakflow_ls'
```
