# CPA Attack Design

## Status

Design note for the future AES CPA attack implementation.

This document records the agreed shape contracts, element split, metadata rules,
and implementation plan for a reusable correlation power analysis attack path.
The immediate validation target is AES. The design intentionally keeps the
statistical attack core algorithm-agnostic so a future Kyber / ML-KEM hypothesis
element can reuse the same `CpaAttack` element.

This is a design and planning document only. It does not imply the elements
described here already exist.

## Current State

LeakFlow already has the building blocks needed to start this work:

- `AesLeakage` computes known-key AES first-round leakage targets.
- `AesLeakage.channels` already uses the strings `HW(m)`, `HW(m_xor_k)`, and
  `HW(y)`.
- `AesLeakage` currently emits known-key leakage as a Torch `uint8` tensor with
  shape `[B,N,C]`, where `B` is selected AES byte indexes, `N` is trace count,
  and `C` is selected leakage channels.
- `PearsonPoiFinder` already computes Pearson correlation between trace samples
  and target leakage and has the recompute/incremental mode vocabulary that
  should be reused by CPA later.
- `CorrelationPoiPayload`, `CorrelationPoiToPlotAnnotations`, `TracePlot`, and
  `Summary` provide useful patterns for custom payloads, summaries, and plot
  conversion.

The missing pieces are:

- an AES key-guess leakage hypothesis element,
- a generic CPA ranking element,
- an attack-result payload,
- known-key attack statistics,
- attack-focused plots,
- and later incremental/live CPA state.

## Shape Vocabulary

Use these axis names in CPA code and metadata:

```text
U = attack units
    AES bytes, Kyber coefficients/groups, or any independent attack unit.

G = guesses / candidates
    256 for a full AES byte guess domain.

N = traces
    The trace/capture axis used for correlation.

L = leakage channels
    For example HW(y), HW(m_xor_k), or multiple model channels.

S = samples
    Time samples in each measured trace.
```

Avoid using `B` in generic CPA code. `B` is useful for AES byte-specific
documentation, but `U` keeps the attack core algorithm-agnostic.

## Element Split

The planned AES CPA path is:

```text
traces ─────────────────────────────────┐
                                        ▼
plaintexts -> AesLeakageHypothesis -> CpaAttack -> CpaAttackStats -> plots
```

Later, Kyber should replace only the hypothesis element:

```text
traces ──────────────────────────────────┐
                                         ▼
public data -> KyberLeakageHypothesis -> CpaAttack -> CpaAttackStats -> plots
```

`CpaAttack` must not know AES, S-boxes, plaintext byte positions, Kyber
coefficients, or key schedules. It only consumes measured trace features and
numeric hypothetical leakage.

## `AesLeakageHypothesis`

`AesLeakageHypothesis` is AES-specific and produces predicted leakage for every
selected byte and every key guess.

Suggested first properties:

```text
byte_indexes=[0..15]
channels=[HW(y)]
guess_values=[0..255]     // default domain; may remain implicit at first
```

The `channels` property should reuse the existing `AesLeakage` channel strings:

```text
HW(m)
HW(m_xor_k)
HW(y)
```

Input:

```text
plaintexts [N,16]
```

Output:

```text
hypotheses [U,G,N,L]
```

For AES first-round hypotheses:

```text
m       = plaintexts[n, byte]
m_xor_k = plaintexts[n, byte] XOR guess[g]
y       = SBox(m_xor_k)
```

Then:

```text
HW(m)       -> HW(m)
HW(m_xor_k) -> HW(m_xor_k)
HW(y)       -> HW(y)
```

For example:

```text
AesLeakageHypothesis(channels=[HW(y)],byte_indexes=[0,1,2])
```

with `N = 2000` emits:

```text
[3,256,2000,1]
```

The channel axis `L` follows the exact property order. For:

```text
channels=[HW(m),HW(m_xor_k),HW(y)]
```

the output axis is:

```text
L=0 -> HW(m)
L=1 -> HW(m_xor_k)
L=2 -> HW(y)
```

`HW(m)` does not depend on the key guess. It can be useful for diagnostics, but
it should not participate in key ranking by default because it gives every guess
the same evidence.

Suggested metadata:

```text
attack.hypothesis.algorithm = aes
attack.hypothesis.round = first
attack.unit.kind = byte
attack.unit.indexes = 0,1,2
attack.guess.kind = byte
attack.guess.count = 256
attack.guess.order = domain
attack.guess.values = 0..255
payload.leakage.channels = HW(y)
attack.channel.depends_on_guess = true
tensor.axes = attack_unit,guess,trace,leakage_channel
```

For multiple channels:

```text
payload.leakage.channels = HW(m),HW(m_xor_k),HW(y)
attack.channel.depends_on_guess = false,true,true
```

## Sharing Code With `AesLeakage`

`AesLeakage` and `AesLeakageHypothesis` should share AES math helpers, not
pipeline element logic.

`AesLeakage` does known-key leakage:

```text
plaintexts + real keys -> [U,N,L]
```

`AesLeakageHypothesis` does guess-domain leakage:

```text
plaintexts + guess values -> [U,G,N,L]
```

Both should call shared helper logic for:

- AES S-box table,
- byte selection,
- `m`, `m_xor_k`, and `y` formulas,
- Hamming weight conversion,
- channel parsing and validation,
- channel metadata naming.

The formulas differ mainly by the extra guess axis:

```text
AesLeakage HW(y):
  y[u,n] = SBox(plaintexts[n,byte] XOR key[n,byte])

AesLeakageHypothesis HW(y):
  y[u,g,n] = SBox(plaintexts[n,byte] XOR guess[g])
```

The first implementation should avoid copying the S-box/HW logic into the new
element. New AES channels should be added once in the shared helper layer and
then become available to both elements.

## `CpaAttack`

`CpaAttack` is the generic statistical attack element.

Inputs:

```text
features    [N,S] or [U,N,S]
hypotheses  [U,G,N,L]
```

Output correlations:

```text
correlations [U,G,L,S]
```

Time/sample remains the last axis. For one attack unit, guess, and channel:

```text
correlations[u,g,l] -> [S]
```

Scores preserve the original guess order:

```text
scores [U,G]
```

Meaning:

```text
scores[u,g] = score for guess index/value g
```

Scores are not sorted. Sorted guesses belong in a separate ranking tensor:

```text
ranking [U,G]
```

Meaning:

```text
ranking[u,0] = best guess index
ranking[u,1] = second-best guess index
```

Suggested first properties:

```text
score_method=max_abs
score_channels=guess_dependent
emit_correlations=false
top_k=5
```

Default scoring:

```text
score[u,g] = max(abs(correlation[u,g,l,s]))
```

reduced over selected `L` channels and all `S` samples. The default
`score_channels=guess_dependent` excludes channels like `HW(m)` from final key
ranking while still allowing them to exist for diagnostics.

Suggested metadata:

```text
attack.method = cpa
attack.correlation.method = pearson
attack.score.method = max_abs
attack.score.order = unsorted
attack.score.rank_order = descending
attack.score.reduced_axes = leakage_channel,sample
attack.score.channels = guess_dependent
attack.ranking.order = descending_score
attack.ranking.values = guess_index
```

## `CpaAttackPayload`

`CpaAttack` should emit a dedicated payload instead of forcing downstream
elements to infer meaning from one raw tensor.

Suggested tensors:

```text
scores        [U,G]
ranking       [U,G]
best_guess    [U]
best_score    [U]
best_channel  [U]
best_sample   [U]
correlations  [U,G,L,S] optional
```

`Summary` should be able to show a compact result such as:

```text
unit 0: best=0x2b score=0.82 sample=1842 channel=HW(y)
unit 1: best=0x7e score=0.76 sample=1901 channel=HW(y)
```

The output tensor values stay numeric. Interpretation, score method, channel
selection, and ranking meaning are metadata.

## `CpaAttackStats`

`CpaAttackStats` computes known-key diagnostics when the true key/secret is
available.

Inputs:

```text
attack_result
truth [U] or AES key [16]
```

Suggested outputs:

```text
true_rank [U]
true_score [U]
top1_guess [U]
top2_guess [U]
score_gap [U]
success [U]
```

Use precise terminology:

- `true_rank` or `PGE` for one attack run,
- `GE` only for an average over repeated independent runs.

## Plots

Plotting should stay downstream from the attack computation.

Useful first plot/converter elements:

```text
CpaAttackToPlotAnnotations
CpaScorePlot
CpaRankPlot
CpaCorrelationPlot
```

Useful visualizations:

- per-unit best guess table,
- true key rank per unit,
- top-1 vs top-2 score gap,
- score distribution over guesses for a selected unit,
- correlation curve for a selected guess/channel,
- correlation heatmap `[G,S]` for one unit/channel.

The first practical plot win is a score/rank view that makes it obvious whether
the correct key byte is separating from the alternatives.

## Recompute And Incremental Modes

The first CPA implementation should be recompute-only.

After the result format is stable, add the same mode vocabulary used by
`PearsonPoiFinder`:

```text
correlation_mode=auto|recompute|incremental
active_correlation_mode=recompute|incremental   // read-only
```

Suggested behavior:

```text
auto + live upstream     -> incremental
auto + non-live upstream -> recompute
```

Incremental CPA can store sufficient statistics instead of all prior traces:

```text
count
sum_x       [S]
sum_x2      [S]
sum_y       [U,G,L]
sum_y2      [U,G,L]
sum_xy      [U,G,L,S]
```

When active mode is incremental, replay/cached-buffer behavior must be protected
the same way as other stateful incremental elements. A property change upstream
must not silently mix new cached buffers with old accumulated state.

## Future Algorithm Hypothesis Contract

To add a new algorithm later, create a new hypothesis element that satisfies the
same generic contract:

```text
<Algorithm>LeakageHypothesis
```

Inputs are algorithm-specific:

```text
plaintexts, ciphertexts, public keys, messages, labels, segments, ...
```

Output is generic:

```text
hypotheses [U,G,N,L]
```

Required metadata:

```text
attack.hypothesis.algorithm = <algorithm>
attack.unit.kind = <unit-kind>
attack.unit.indexes = ...
attack.guess.kind = <guess-domain-kind>
attack.guess.count = <G>
attack.guess.order = domain
payload.leakage.channels = <ordered channel list>
attack.channel.depends_on_guess = <ordered bool list>
tensor.axes = attack_unit,guess,trace,leakage_channel
```

The hypothesis element owns:

- algorithm-specific intermediate formulas,
- candidate/guess domain generation,
- algorithm-specific leakage channel formulas,
- metadata describing units, guesses, and channels.

The hypothesis element must not own:

- Pearson correlation,
- ranking,
- plotting,
- known-key statistics,
- CPA state accumulation.

Those remain in `CpaAttack`, `CpaAttackStats`, and plot elements.

## Implementation Plan

### Phase 1: Shared AES hypothesis helpers

Refactor or extend the AES helper layer so `AesLeakage` and the future
`AesLeakageHypothesis` can share channel parsing, byte selection, S-box, and
Hamming-weight logic.

Keep `AesLeakage` behavior unchanged.

Validation:

- existing `AesLeakage` tests still pass,
- helper tests verify known-key mode and guess-domain mode agree when the guess
  domain contains the real key byte.

### Phase 2: `AesLeakageHypothesis`

Add the AES hypothesis plugin element.

Minimum support:

```text
plaintexts [N,16]
byte_indexes=[...]
channels=[HW(y)]
guess_values=0..255 default
output [U,256,N,L]
```

Validation:

- shape tests,
- metadata tests,
- channel order tests,
- numeric spot checks for `HW(y)`,
- multi-channel tests for `HW(m)`, `HW(m_xor_k)`, and `HW(y)`.

### Phase 3: `CpaAttack` recompute mode

Add the generic CPA plugin element with recompute-only Pearson correlation.

Minimum support:

```text
features [N,S]
hypotheses [U,G,N,L]
scores [U,G]
ranking [U,G]
best_guess [U]
```

Validation:

- synthetic data where the winning guess is known,
- AES fixture smoke pipeline:

```text
TorchFileSrc@traces -> CpaAttack.features
TorchFileSrc@plaintexts -> AesLeakageHypothesis -> CpaAttack.hypotheses
CpaAttack -> Summary
```

### Phase 4: `CpaAttackPayload` and summaries

Move attack outputs into a dedicated payload with named tensors and useful
summary rendering.

Validation:

- payload shape/type validation,
- summary output includes best guess, score, channel, and sample.

### Phase 5: `CpaAttackStats`

Add known-key diagnostics.

Validation:

- true-key rank is correct for synthetic fixtures,
- AES key fixture produces expected true-key extraction/ranking behavior.

### Phase 6: CPA plots

Add attack-focused plot converters/elements.

Validation:

- non-visual payload/snapshot tests,
- manual graph UI smoke pipeline.

### Phase 7: Incremental/live CPA

Add incremental CPA state and `auto` mode.

Validation:

- recompute and incremental agree on the same sequence,
- live pipeline updates scores over time,
- upstream cached-buffer/property-change behavior is rejected or safely resets.

### Phase 8: Kyber hypothesis element

Add a Kyber / ML-KEM hypothesis element only after the AES CPA path is stable.

`CpaAttack` should require no algorithm-specific changes in this phase.

## Example Target Pipeline

Non-live AES CPA:

```text
TorchFileSrc@traces_src(path=traces/aes/sync/aes_sync_attack/key_01/traces.pt);
TorchFileSrc@plain_src(path=traces/aes/sync/aes_sync_attack/key_01/plain_texts.pt);
AesLeakageHypothesis@hyp(channels=[HW(y)],byte_indexes=[0]);
CpaAttack@attack(score_method=max_abs,score_channels=guess_dependent);
Summary@summary(level=2);

@traces_src ! @attack.features;
@plain_src ! @hyp.plaintexts;
@hyp ! @attack.hypotheses;
@attack ! @summary
```

With known-key stats:

```text
TorchFileSrc@key_src(path=traces/aes/sync/aes_sync_attack/key_01/key.pt);
CpaAttackStats@stats;

@attack ! @stats.attack_result;
@key_src ! @stats.truth
```
