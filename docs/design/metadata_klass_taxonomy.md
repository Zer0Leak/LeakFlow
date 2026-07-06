# Metadata, Caps, and Klass Taxonomy

This is the authoritative design document for how LeakFlow classifies metadata,
how metadata is forwarded between elements, how element `klass` strings are
structured, and how those map to a side-channel-attack (SCA) building-block
taxonomy.

The compact day-to-day version lives in
`docs/context/ARCHITECTURE_CONTRACTS.md`. This document is the source of truth.

## Motivation

LeakFlow is a Lego of SCA building blocks: techniques can be mixed, a block can
be swapped for another that fills the same slot, and the same block can be reused
with different parameters. For that to work:

- buffers must carry the right metadata to the right downstream blocks, and
- elements must not blindly propagate facts that stop being true once they
  transform or fuse data.

Before this design, a multi-input element such as `AesLeakage` simply copied
one arbitrarily-chosen input's entire metadata map. That preserved some facts
(for example `sample_rate_hz`, because it happened to be on the copied input)
while silently dropping others (for example `capture.source`, which was only on
the traces input), and it left misleading provenance on derived buffers (a
leakage buffer claiming `file.path=.../plain_texts.pt`).

## 1. Metadata taxonomy — four groups

Every metadata key belongs to exactly one group, chosen by the question:
*is this fact still true of the output buffer after the element acts?*

| Group | Meaning | Lifetime |
|---|---|---|
| **capture** | physical acquisition / dataset / countermeasure facts | true of anything derived from the experiment |
| **origin** | provenance of one specific input (file + role) | true only while the buffer *is* that input |
| **payload** | a producer's assertion about the current payload bytes | invalidated when the payload is replaced |
| **routing** | transient pipeline-topology / identity scratch | valid only on the immediate link |

`capture.countermeasure.*` (masking order, shuffling, desynchronization, ...) is
a first-class **capture** fact: nearly all SCA results are meaningless without
it, so it must survive transforms and fusion.

## 2. Group resolution

Every stamped metadata key carries its group as the **leading segment**, so
`leakflow::metadata_group(key)` is a direct lookup of that segment:

```text
leading capture.*    -> capture   (capture.source, capture.sample_rate_hz, capture.dataset.name, capture.countermeasure.*)
leading origin.*     -> origin    (origin.file.*, origin.role, and fused origin.<pad>.*)
leading routing.*    -> routing   (routing.element, routing.branch.*)
leading payload.*    -> payload   (payload.leakage.*, payload.crypto.*, payload.poi.*, payload.conversion.*, ...)
DEFAULT (unprefixed) -> payload
```

`leakflow_core` only needs the four group names; it never lists any domain
vocabulary. Unprefixed or unknown keys default to payload, the safe choice for
facts that should ride pass-through but not be forwarded onto derived buffers.

Every element stamps keys in the prefixed form (for example `TorchFileSrc` stamps
`origin.file.path` and `payload.leakage.inverted`). Hand-typed CLI metadata
annotations should follow the same convention; a bare annotation key resolves to
payload.

## 3. Klass scheme and forwarding profiles

Element `klass` is `<Profile>/<Family>[/<Role>[/<Variant>...]]`. The **leading
token is the forwarding profile** (a refinement of GStreamer's Source/Sink/Filter
split); the remaining tokens are the human-facing taxonomy used by `leakflow-ls`,
the graph, docs, and examples.

The rule of thumb is: profile says *how metadata flows*; family says *what slot
the element occupies*. For example, `PassThrough/Flow/Queue` and
`PassThrough/Flow/Sync` have the same forwarding profile but different flow
roles, while `Analyze/SCA/Score/Correlation` and `Analyze/SCA/PoI/Select` are
different SCA analysis slots that can be composed.

`leakflow::profile_for_klass(klass)` maps the leading token:

| Leading token | Profile | Meaning |
|---|---|---|
| `Source` | Source | produces buffers; owns its metadata |
| `Sink` | Sink | terminal; forwards nothing |
| `PassThrough` | PassThrough | forwards the buffer envelope unchanged |
| `Convert` | Reframe | single-input representation change |
| `Analyze` | Analyze | multi-input fusion or derivation of new knowledge |
| *(other, e.g. `Control`)* | PassThrough | conservative default |

## 4. Forwarding matrix

`leakflow::forward_metadata(inputs, profile, output)` applies, before the element
stamps its own keys:

| Profile | capture | origin | payload | routing |
|---|---|---|---|---|
| **Source** | own | own | own | — |
| **PassThrough** | copy | copy | copy | drop |
| **Reframe** | copy | copy | **drop** | drop |
| **Analyze** | **union** (conflict → error) | **`origin.<pad>.<key>`** | drop | drop |
| **Sink** | — | — | — | — |

Rules and rationale:

- **Routing is never forwarded** by the helper. `element` is re-stamped by each
  producing element; `branch.*` is stamped per-output at the link by `Tee` and
  CLI pad-metadata annotations. Pass-through elements that simply return the
  input buffer (Tee, Queue, Summary) keep the envelope intact and so carry
  routing to the immediate consumer; the next reframe/analyze drops it.
- **Analyze unions capture** across all connected inputs and **throws** on a
  conflicting value (for example two different `sample_rate_hz`). Capture values
  are non-secret, so the error names them.
- **Analyze relabels origin** as `origin.<pad>.<key>` so multiple inputs cannot
  collide and the output schema is stable. Example: the traces input's
  `file.path` becomes `origin.traces.file.path`. When an input's origin key is
  itself already forwarded (`origin.<prev>.<...>`, e.g. a prior Analyze output),
  the redundant leading `origin.` is stripped before relabeling, yielding a flat
  provenance path such as `origin.targets.keys.file.path` rather than a nested
  `origin.targets.origin.keys.file.path`.
- **Reframe copies origin as-is** (single input, no relabel) and copies capture,
  but **drops upstream payload** — the element re-owns only the payload facts it
  can vouch for (for example `conversion.*`).
- An Analyze element may additionally **re-own a curated subset** of payload
  facts it is asserting about its output. `PearsonCorrelator`/`PoiSelect` do this for
  `leakage.*`/`crypto.*`/`trace.*` that describe the target model, since the PoI
  results remain *about* those targets.

A future refinement, not yet implemented: a Reframe brick that legitimately
rewrites a capture key it owns (for example a `Resample` brick updating
`sample_rate_hz`). The mechanism for declaring such own-overrides will be added
when the signal-processing bricks are built.

## 5. Caps and payload model

Caps stay **general**; generality is set by the most generic *consumer*, not by
the producer. If a generic statistical routine consumes from two producers, both
producers emit the same general caps so the routine is a drop-in slot. Therefore
LeakFlow does **not** mint a distinct caps type per semantic role.

- Raw loaders (`TorchFileSrc`, `NumpySrc`) emit the generic transport caps
  (`leakflow/torch-tensor`, `leakflow/numpy-array`). Semantic meaning is layered
  downstream by a semantic-aware element, a CLI caps annotation, or the first
  SCA-aware element, using the existing generic↔concrete compatibility rule.
- Numeric semantics (traces, leakage, labels, scores, key-as-bytes,
  oracle-answers-as-bits) reuse `TorchTensorPayload`, distinguished by
  `dtype`/`rank`/`shape` and metadata `role`.
- A new payload type is introduced **only** when the structure is not a tensor.
  The single near-term case is `sca-model` (a trained template/NN). Oracle
  answers and recovered keys stay tensors until something proves otherwise.
- semantic role stays **metadata** (`origin.role`), not a blocking caps param, so
  genericity remains the default.

## 6. Library layering

```text
leakflow_core   framework kernel; no domain knowledge
  -> leakflow_base    minimal numeric transport + generic statistics
       -> leakflow_sca     (future) algorithm-agnostic SCA: shared caps/role
                            vocabulary + SCA math (SNR, TVLA, labeling, POI
                            selection, signal-processing primitives)
            -> leakflow_plugins_sca   (future) generic SCA elements
       -> leakflow_crypto  algorithm-specific helpers (Hamming, S-box, Kyber)
            -> leakflow_plugins_crypto  algorithm-specific elements
```

- `leakflow_base` stays minimal but is the right home for *generic* statistics
  (it already owns `pearson_correlation`).
- `leakflow_sca` is for side-channel work that still applies across algorithms
  (AES, Kyber, ...).
- `leakflow_crypto` is for truly algorithm-specific helpers.
- `PearsonCorrelator`/`PoiSelect` are generic SCA and will eventually move to
  `leakflow_plugins_sca`; for now only its klass was updated.

## 7. SCA building-block taxonomy (the Lego slots)

Derived from a 27-paper SLR corpus of SCA attacks on PQC (Kyber/ML-KEM,
Dilithium/ML-DSA, Falcon, and generalization targets). Each family is a swappable
**slot**; concrete elements are interchangeable **bricks**; properties give brick
variants.

| Slot (Klass family) | Profile | Representative bricks |
|---|---|---|
| `Source/File/*` | Source | file-backed buffers, tensors, models, datasets |
| `Source/Live/*` | Source | live or live-like trace/data sources |
| `Source/App/*` | Source | application-pushed frames |
| `Source/Test/*` | Source | fixtures and synthetic source blocks |
| `Source/AttackInput/*` | Source | chosen-ciphertext crafting for PC-oracle attacks |
| `PassThrough/Flow/*` | PassThrough | Tee, Queue, Sync, future flow-control bricks |
| `PassThrough/Inspect/*` | PassThrough | Summary and other buffer-inspection pass-throughs |
| `Convert/Tensor/*` | Reframe | dtype/device/backend tensor representation changes |
| `Convert/Signal/*` | Reframe | Align, Resample, Filter, Trim, Standardize, Segment, ShareCombine, Average (distinct small bricks) |
| `Convert/PlotAnnotation/*` | Reframe | convert analysis payloads into generic plot annotations |
| `Convert/Predict/*` | Reframe | apply trained model payloads to tensors, then feed analysis blocks |
| `Analyze/SCA/Leakage/*` | Analyze | leakage models (HW/HD/ID) per target op (S-box, NTT, message-decode, unpacking, modular reduction, CDT) |
| `Analyze/SCA/Hypothesis/*` | Analyze | guess-domain leakage hypotheses |
| `Analyze/SCA/Score/*` | Analyze | score/statistic curves such as correlation, SNR, NICV, t-test |
| `Analyze/SCA/PoI/*` | Analyze | PoI selection over score curves: fixed top-k, threshold, outlier, FDR, spacing |
| `Analyze/SCA/Label/*` | Analyze | labelers, cluster/MC labeling, blind labeling |
| `Analyze/SCA/Model/*` | Analyze | template/CNN/MLP/GNN/LR/RL/LLM distinguishers and trainers |
| `Analyze/SCA/Attack/*` | Analyze | attack/ranking blocks such as CPA, DPA, template attack |
| `Analyze/SCA/Evaluation/*` | Analyze | derived diagnostics such as known-key ranks, margins, confidence metrics |
| `Analyze/SCA/Oracle/*` | Analyze | PC (binary), MV-PC, decryption-failure, re-encryption oracles |
| `Analyze/SCA/Solver/*` | Analyze | linear-system, lattice reduction, belief propagation, ILP, key enumeration, residual-opt |
| `Sink/Evaluation/*` | Sink | terminal success rate, guessing entropy, key rank, accuracy/F1 reports |
| `Sink/Plot/*` | Sink | TracePlot, correlation plot, PoI overlay, GE curve |
| `Control/Fault/*` | (future) | voltage-glitch injector and other active perturbation |
| `*/Design/*` | (future) | pre-silicon RTL leakage localization (e.g. GNN on CDFG) |

Models exist as **both** a live `Analyze/SCA/Model/*` element emitting an `sca-model`
payload and a `Source/File/Model` (`ModelFileSrc`) / `Sink/File/Model` loader,
feeding a `Predict` brick.

## 8. Current element klasses

| Element | Klass | Slot |
|---|---|---|
| `FileSrc` / `FileSink` | `Source/File/Bytes`, `Sink/File/Bytes` | raw byte file I/O |
| `BufferFileSrc` / `BufferFileSink` | `Source/File/Buffer`, `Sink/File/Buffer` | persisted full-buffer I/O |
| `TorchFileSrc` / `TorchFileSink` | `Source/File/Torch`, `Sink/File/Torch` | Torch tensor file I/O |
| `NumpySrc` | `Source/File/Numpy` | NumPy array file source |
| `FakeSrc` / `FakeSink` | `Source/Test/Fake`, `Sink/Test/Fake` | test/smoke fixtures |
| `FakeLiveSrc` | `Source/Live/Torch` | deterministic live-like Torch source |
| `AppSrc` | `Source/App/Torch` | application-pushed Torch frames |
| `Tee` | `PassThrough/Flow/Tee` | branch/fan-out flow control |
| `Queue` | `PassThrough/Flow/Queue` | thread/rate-decoupling flow control |
| `Sync` | `PassThrough/Flow/Sync` | cross-source pairing / common-ancestor injection |
| `Summary` | `PassThrough/Inspect/Summary` | buffer inspection without changing data |
| `TorchConvert` | `Convert/Tensor/Torch` | Torch dtype/device reframe |
| `NumpyToTorch` | `Convert/Tensor/NumpyToTorch` | NumPy payload to Torch tensor reframe |
| `CorrelationPoiToPlotAnnotations` | `Convert/PlotAnnotation/PoI` | PoI-to-plot marker conversion |
| `AttackStatsToPlotAnnotations` | `Convert/PlotAnnotation/AttackStats` | attack-stats-to-plot marker conversion |
| `AesLeakage` | `Analyze/SCA/Leakage/AES` | AES leakage target generation |
| `AesLeakageHypothesis` | `Analyze/SCA/Hypothesis/AES` | AES guess-domain hypothesis generation |
| `PearsonCorrelator` | `Analyze/SCA/Score/Correlation` | correlation scoring |
| `PoiSelect` | `Analyze/SCA/PoI/Select` | PoI selection from scores |
| `CpaAttack` / `DpaAttack` | `Analyze/SCA/Attack/CPA`, `Analyze/SCA/Attack/DPA` | attack/ranking blocks |
| `AttackStats` | `Analyze/SCA/Evaluation/AttackStats` | known-key diagnostics and confidence metrics |
| `TracePlot` | `Sink/Plot/Trace` | trace visualization |
| `ScorePlot` | `Sink/Plot/AttackScore` | attack score/confidence visualization |
| `ScoreTablePlot` | `Sink/Plot/AttackScoreboard` | ranked attack scoreboard |

## 9. Deferred work

- **Oracle and Solver execution.** These families are taxonomized (klass + caps
  reserved) but their iterative query→measure→refine execution is deferred to a
  future multi-input / feedback executor. The current executor is synchronous and
  linear.
- **Active fault injection** (`Control/Fault/*`) and **pre-silicon RTL**
  (`*/Design/*`) are scoped as future families; LeakFlow may replicate those
  attacks later.
- **Reframe capture own-override** mechanism (e.g. `Resample` rewriting
  `sample_rate_hz`) is deferred to the signal-processing brick phase.
- **`leakflow_sca` / `leakflow_plugins_sca`** targets and the physical move of
  generic SCA elements are a later phase.
