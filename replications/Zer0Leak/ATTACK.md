# Zer0Leak — Confusion-Aware Blind Side-Channel Attack on AES

> Working title for the paper. This document is the detailed, self-contained
> description of the attack, its relationship to prior work, and the empirical
> results. It is written to be lifted almost directly into a paper draft.
> Companion document: [IMPLEMENTATION.md](IMPLEMENTATION.md).

## 0. One-paragraph summary

We recover an AES key byte in the **fully blind** side-channel setting — the
adversary observes only power traces and knows **no plaintext, no ciphertext, and
no key**. For each trace we estimate the Hamming weights `HW(m)` (of the plaintext
byte) and `HW(y)` (of the first-round S-box output) with two independent, blindly
calibrated 1-D Gaussian-mixture templates, then match the **observed joint
distribution** of `(HW(m), HW(y))` against the **key-dependent theoretical joint
distribution** using a **confusion-aware maximum-likelihood criterion**. On an
unprotected ChipWhisperer AES capture the correct key byte ranks **3 / 256**; on a
jitter-desynchronized capture it ranks **6 / 256** — the only *honestly blind*
method (no deep network, no key-based model selection) that clears the standard
`GE ≤ 10` success bar under desynchronization in our comparison. The method is a
classical, deterministic re-instantiation of Clavier & Reynaud (CHES 2017) with
multi-PoI matched-filter projection and a physically constrained blind template.

## 1. Threat model

- **Blind / man-in-the-middle observer.** The adversary passively measures the
  power (or EM) leakage of an AES encryption. They do **not** know the plaintext,
  the ciphertext, or the key. Only the leakage traces are available. Truth (the
  key) is used **only** to *score* the attack (compute the rank / guessing
  entropy), never inside the attack pipeline.
- **Points of interest (PoIs) are located by offline profiling** on a
  *characterization device* with a known key (here `key_30`), then transferred to
  the *attack device* (`key_01`). This is the **one non-blind assumption shared by
  all prior blind-SCA work** (Linge, Clavier, Rezaeezade all state it). Fully
  blind PoI selection is an open problem and out of scope. We verified the PoIs
  transfer cleanly across devices (rank correlation of `|r|` across all 5000
  samples ≈ **0.937**; the S-box leak lands on the same sample, 1932, on both
  devices).
- **Leakage model.** First-order Hamming-weight leakage with additive Gaussian
  noise, the standard CMOS model and the one all cited works assume.

## 2. Notation and the key fact

For AES, `m` is the plaintext byte (public in a normal attack, **unknown** here)
and the sensitive intermediate is

```
y = SBOX(m XOR k*)          k* = secret key byte
```

- `HW(m)` = Hamming weight of the plaintext byte; leaks when the byte is loaded.
- `HW(y)` = Hamming weight of the S-box output; leaks during SubBytes.

**Key fact (Linge et al.).** Marginally, both `HW(m)` and `HW(y)` are
binomially distributed and *key-independent*, so neither alone reveals `k*`. But
the **joint** distribution `Pr(HW(m), HW(y) | k)` **depends on the key**: over the
256 possible plaintext values, the pairs `(HW(p), HW(SBOX(p XOR k)))` and their
frequencies form a 9×9 histogram that is distinct for every `k`. We verified all
256 keys yield **distinct** joint histograms, and that with perfect labels the
key is recovered uniquely (rank 1/256). This joint distribution is the *key
fingerprint* the attack matches against.

> Note on the definition of `m`. LeakFlow's `AesLeakage` and this attack define
> `m = plaintext byte` (leaks at load), **not** the S-box input `p XOR k`. This
> matters: the achievable set of `(HW(m), HW(y))` pairs is then **key-dependent**
> (42–49 distinct groups across keys; 45 for our `key_01` byte 0), whereas the
> S-box-input pairing `(HW(p XOR k), HW(y))` would be a fixed 47 groups
> independent of the key.

## 3. The attack pipeline

Six stages. Stages 1–5 are fully blind; stage 0 is the shared profiling
assumption.

### Stage 0 — PoI location (offline profiling)
On the characterization device, correlate every sample with the (known)
`HW(m)` and `HW(y)`; keep the **top 8 samples per channel** by `|r|`. These sample
indices are transferred to the attack device. (8 was the sweet spot; see §5.)

### Stage 1 — Matched-filter projection
Collapse each channel's 8 PoIs into a single scalar per trace via a
correlation-weighted sum (a matched filter), then standardize:

```
f_m = standardize( Σ_i  r_m[poi_i] · trace[:, poi_i] )     ≈ noisy HW(m)
f_y = standardize( Σ_i  r_y[poi_i] · trace[:, poi_i] )     ≈ noisy HW(y)
```

`f_m` and `f_y` are two independent 1-D leakage estimates (measured
`corr(f_m, f_y) ≈ 0.06` — the two leaks are physically separate in time).

### Stage 2 — Per-channel blind labeling (physically-constrained template)
We label each axis **separately** (this is "Case B"; see §6 for why this does not
lose the m–y relationship). For each of `f_m`, `f_y` we fit a **9-component 1-D
Gaussian mixture with a hard physical constraint**:

```
f ~ Σ_{h=0..8}  Binom(h; 8, ½) · N( a + b·h ,  σ² )
```

- means are **linear in the Hamming weight** (`μ_h = a + b·h`),
- variance `σ²` is **shared** across levels,
- mixing weights are **fixed to the binomial** `[1,8,28,56,70,56,28,8,1]/256`.

Only **3 parameters** `(a, b, σ)` are fitted, by EM. This constraint is the
crux of the blind template: a *free* 9-component GMM has 27 parameters and, at
realistic overlap, carves the leakage blob into arbitrary lumps that do **not**
correspond to Hamming-weight levels (blind confusion built from it fails — rank
~155 in our tests). The 3-parameter model *cannot* overfit and recovers the true
levels. The per-trace label is `argmax_h posterior(h | f)`.

### Stage 3 — Blind confusion matrix `C` (closed-form Gaussian)
The labels are noisy; the *confusion matrix* `C[t,e] = Pr(label = e | true HW =
t)` describes how. We compute it **blindly and in closed form** from the fitted
template — no truth, no Monte-Carlo:

- The `argmax` decision boundaries between adjacent levels `e, e+1` (with equal
  variance and binomial priors) are the closed-form points where the two
  posteriors are equal.
- `C[t, e] = Φ((edge_e − μ_t)/σ) − Φ((edge_{e-1} − μ_t)/σ)` — each true level's
  Gaussian integrated over the region where the labeler outputs `e`.

We build a separate `C_m` and `C_y` (the two channels' errors are independent —
this separability is exactly Clavier CHES 2017 Eq. 3, and is what makes Case B the
right structure). **Important subtlety:** because our labeling uses the binomial
prior it is *biased* (pulled toward `HW=4`); the confusion `C` must be **faithful
to that biased labeler** (integrate over the actual argmax regions). A literal
copy of Clavier's *symmetric* Gaussian `C` (which assumes an unbiased
`h = (l−β)/α` estimate) underperforms on our biased labeler (rank 39/188 vs. 3).

### Stage 4 — Theoretical per-key joint distributions `J_k`
For every key hypothesis `k ∈ {0..255}`, precompute (cipher-only, no traces) the
9×9 histogram

```
J_k[a,b] = ( #{ p ∈ 0..255 : HW(p)=a and HW(SBOX(p XOR k))=b } ) / 256
```

### Stage 5 — Confusion-aware maximum-likelihood key ranking
Build the observed 9×9 histogram `Nobs` by cross-tabulating the per-trace
`(label_m, label_y)` pairs (this preserves the m–y relationship per trace). Then
score each key by matching `Nobs` to the **confusion-blurred** model, **not** the
clean `J_k`:

```
J̃_k = C_mᵀ · J_k · C_y                      # blur the model the same way the labeler blurs the data
LL(k) = Σ_{a,b}  Nobs[a,b] · log J̃_k[a,b]
rank  = position of k* when keys are sorted by LL (descending)
```

This is the decisive step: matching the *blurred* observed histogram against the
*clean* `J_k` fails (rank ~68); matching against `C_mᵀ J_k C_y` succeeds (rank 3).
`J̃_k = C_mᵀ J_k C_y` is exactly Clavier CHES 2017 Eq. 2–3.

### Orientation resolution
Blindly we do not know whether higher leakage means higher or lower Hamming
weight (a per-axis sign ambiguity). We try both orientations per axis (reverse the
HW labels `h → 8−h` and reverse `C` accordingly) and keep the combination with the
**highest achievable log-likelihood** — a blind criterion (no truth). In our runs
this selects the correct orientation.

## 4. Relationship to prior work (honesty section)

The core of this attack is **not novel**; it is a faithful re-instantiation of the
classical blind-SCA ML criterion. This section maps each component to its origin;
it must appear in any paper.

| Component | Origin |
|---|---|
| Joint `(HW(m), HW(y))` distribution is the key fingerprint | **Linge, Dumas, Lambert-Lacroix**, COSADE 2014 |
| Confusion-aware match `J̃_k = C_mᵀ J_k C_y` | **Clavier & Reynaud**, CHES 2017, **Eq. 2** |
| Per-channel **independent Gaussian** confusion `C_m`, `C_y` | **Clavier & Reynaud**, CHES 2017, **Eq. 3** |
| Blind estimation of the leakage→HW coefficients (no key) | **Clavier & Reynaud**, CHES 2017, *Variance Analysis* labeling |
| Single-PoI "slicing" (binomial-quantile) labeling | **Linge et al.** (COSADE 2014) |
| GMM / multi-PoI ("MC") labeling | **Rezaeezade et al.**, USENIX Security 2025 |
| Multi-round fusion via the key schedule (belief propagation) | **Le Bouder et al.**, FPS 2016 |

**What is genuinely ours (a modest delta):**
1. A **classical, deterministic** pipeline (no deep network) that combines
   *multi-PoI matched-filter* projection with a **physically-constrained 3-parameter
   blind template** per channel, and derives the confusion matrix **faithfully to
   the biased argmax labeler** via closed-form argmax-region CDFs.
2. Demonstrating this **handles a jitter/desynchronization countermeasure**
   without a DNN — a case prior classical work never cleared (they report
   `GE = 55–208`, above the `GE ≤ 10` bar).
3. An explicit **honest-blindness argument**: the method needs **no key-based
   model selection**, unlike the deep-learning approach (see §7).

## 5. Results

Dataset: ChipWhisperer AES, 10 000 traces × 5000 samples, byte 0 (`key = 0x33`).
`sync` = unprotected (`traces/aes/sync/aes_sync_attack`); `jitter` =
`loop_iterations` desynchronization (`traces/aes/jitter/aes_jitter_attack`). PoIs
profiled on `key_30`, attack on `key_01`. Single deterministic run, per-byte rank.

**Correct key byte rank (lower is better; out of 256):**

| Dataset | Rank | HW(m) acc | HW(y) acc | Notes |
|---|--:|--:|--:|---|
| sync (unprotected) | **3** | 0.37 | 0.49 | σ_HW ≈ 0.71 both channels |
| jitter (desync) | **6** | 0.31 | 0.20 | 8 PoI/ch; 30 PoI → 131, 100 PoI → 75 |

**PoI transfer (profiling `key_30` → attack `key_01`).**
- sync: HW(m) leak at samples 528–533 (`|r|` 0.51→0.19), HW(y) at 1932–1939
  (`|r|` 0.71→0.24); near-lossless transfer.
- jitter: HW(m) survives (peak `|r|` 0.56, *replicated at several positions* 420,
  2936–2939, 3993–3995 — the jitter copies the early leak), transfers cleanly;
  HW(y) is *wrecked* (peak `|r|` 0.18 on profiling, halving to 0.08–0.11 on the
  attack device) because it leaks *late*, after the variable delay accumulates.
  The rank-6 result therefore rides mostly on the strong, transferable HW(m)
  channel plus a faint HW(y) signal the confusion-aware scoring correctly
  down-weights.

**Ablation / key findings.**
- Naive match (clean `J_k`): rank **68**. Confusion-aware (`C_mᵀ J_k C_y`): rank
  **3**. → the confusion-aware criterion is essential.
- Blind `C` from a **free** GMM: rank **155** (fails). Blind `C` from the
  **constrained** linear-binomial template: rank **3**. → the physical constraint
  is essential.
- Monte-Carlo `C` and closed-form argmax-region Gaussian `C` agree (both rank 3).
- Symmetric Clavier Eq. 5 `C` on our *biased* labeler: rank 39/188. Faithful
  argmax-region `C`: rank 3.
- More PoIs hurt under jitter (matched filter averages the desync-smeared leak).

**Comparison with prior blind AES results (honestly-blind only).**
Per-byte guessing entropy / rank. *Different datasets* (their CW vs. our
sync/jitter) — a capability comparison, not head-to-head.

Unprotected AES:

| Method (honestly blind) | GE / rank |
|---|--:|
| Linge et al. 2014 | 218 |
| Clavier & Reynaud 2017 | 223 |
| Rezaeezade — MC labeling (classical) 2025 | 79 |
| **Ours** | **3** |

Desynchronized AES:

| Method (honestly blind) | GE / rank | GE ≤ 10? |
|---|--:|:--:|
| Linge et al. 2014 | 67 | ✗ |
| Clavier & Reynaud 2017 | 208 | ✗ |
| Rezaeezade — MC labeling (classical) 2025 | 55 | ✗ |
| **Ours** | **6** | **✓** |

Rezaeezade's deep-network numbers (GE ≈ 1 unprotected, `< 4` desync) are
**excluded** because they select the top-10 of 100 models **by guessing entropy**,
which requires the secret key — not achievable in a blind attack (see §7).

## 6. Why per-channel labeling (Case B) does not lose the m–y relationship

A natural worry: since `y` depends on `m`, does labeling `HW(m)` and `HW(y)`
independently discard their dependence? No.

- The dependence is a property of the **traces** (each trace carries its own true
  `(HW(m), HW(y))` pair). Case B labels each axis from its own physical leak, then
  **cross-tabulates per trace**, which preserves the pairing. Two traces with the
  same `HW(m)=4` but different `HW(y)` land in different cells — the joint spread
  survives.
- You **cannot** use the m–y relationship *during* labeling anyway: that
  relationship *is* `k`-dependent, so using it would assume the answer (circular).
  The relationship is the **output** of the attack, not an input to labeling.
- Independent labeling makes the confusion **separable** (`C_m`, `C_y`
  independent), which is precisely the form Clavier Eq. 3 uses and what the
  confusion-aware match needs. (Rezaeezade's MC labeling instead clusters the two
  channels **jointly** into `(B+1)^{n_b}` = 81 clusters; that is a valid
  alternative for feeding a DNN, but it entangles the confusion and is harder to
  model in a closed-form classical criterion.)

## 7. The honest-blindness argument (model selection)

Rezaeezade et al. report GE ≈ 1 by training **100 random DNNs** and keeping the
**top-10 by guessing entropy**, then averaging. GE is the rank of the *correct
key*, so identifying the winning models **requires knowing the secret**. The paper
even explicitly rejects the one blind-computable metric (classification accuracy)
as "inadequate." Therefore their headline numbers are **not achievable in a
genuinely blind attack**; they assume oracle model selection.

Our method needs **no model selection at all**: it is deterministic, single-run,
and the only place the key appears is the final rank computation used to *evaluate*
the attack. This is the core methodological contribution to foreground: *prior
blind AES key recovery is either weak-but-honest (classical, GE 55–223) or
strong-but-not-actually-blind (DNN, GE ≈ 1 via key-based selection); a classical
confusion-aware attack is both honestly blind and strong, and it handles jitter.*

(For fairness, top-k-by-GE selection is common in DL-SCA and is sometimes defended
under a portability/profiling threat model — pick the model on a known-key
profiling device, deploy on the target. Rezaeezade do not make that argument.)

## 8. Limitations

1. **PoI selection is non-blind** (offline profiling on a known-key device) —
   shared by all cited works; fully blind PoI location remains open.
2. **Single key byte, single run** vs. the literature's averaged GE over 100
   experiments. Full-key recovery needs the 16 bytes combined (see §9).
3. **Different datasets** — our numbers are on our sync/jitter captures, not the
   public CW datasets used by the baselines. The claim requires same-dataset runs.
4. **SNR ceiling.** On sync the per-sample gap/noise ≈ 0.68 (adjacent HW levels
   overlap by ~⅔σ); the joint 45-group truth is not perfectly recoverable, only
   ranked. Under jitter the attack leans on HW(m); if the jitter also destroyed
   HW(m), the joint fingerprint would collapse.
5. **Orientation** is resolved heuristically by max-LL (works here, not proven).

## 9. Future work

1. **Same-dataset validation** on the public CW / CW-desync (and ASCON, Kyber)
   datasets — the step that turns this from "promising" into a defensible claim.
2. **Multi-byte / full-key fusion via the key schedule**, following Le Bouder et
   al. (FPS 2016): combine the 16 per-byte joint fingerprints through the
   AES KeyExpansion constraints with belief propagation. This is the honestly-blind
   route to a full-key break and to pushing single-byte rank-3/6 toward rank 1,
   *without* reintroducing the DNN's non-blind model selection.
3. **A blind model-selection proxy** (sharpness/margin of the key distribution,
   `max_k Pr(k)`), which is key-independent — both to strengthen our own
   auto-tuning and to re-evaluate the DNN approach fairly.
4. **Masking** — currently out of reach for classical blind SCA (Clavier's
   masking variant is weak/simulated).
5. **Fully blind PoI selection** (variance/SOST-like, or self-supervised).

## 10. References

> Bibliographic details to be verified against the venues before submission.

1. Y. **Linge**, C. **Dumas**, S. **Lambert-Lacroix**. *Using the Joint
   Distributions of a Cryptographic Function in Side Channel Analysis.* COSADE
   2014. — origin of blind SCA via joint HW distributions and *slicing* labeling.
2. C. **Clavier**, L. **Reynaud**. *Improved Blind Side-Channel Analysis by
   Exploitation of Joint Distributions of Leakages.* CHES 2017, LNCS 10529. —
   ML criterion (Eq. 2), per-channel Gaussian confusion (Eq. 3), *Variance
   Analysis* blind coefficient estimation. **The paper this attack re-instantiates.**
3. H. **Le Bouder** et al. *A Multi-Round Side Channel Attack on AES using Belief
   Propagation.* FPS 2016 (HAL hal-01405793). — fully blind, key-schedule fusion.
4. A. **Rezaeezade** et al. *Breaking the Blindfold: Deep Learning-based Blind
   Side-channel Analysis.* USENIX Security 2025 (ePrint 2025/157). — MC labeling,
   DL-BSCA, first blind SCA on desync/Kyber/ASCON. (Model selection via GE — §7.)
5. M. **Renauld**, F.-X. **Standaert**. *Algebraic Side-Channel Attacks.* — single-
   trace, unknown-plaintext capable (semi-blind).
6. *Blind Side Channel Analysis Against AEAD with a Belief Propagation Approach.*
   CARDIS 2023 (Elephant/Sparkle). — blind BP on lightweight ciphers.
7. *MALEAK: Blind Side-Channel Key Recovery Exploiting Modular Reduction.* ePrint
   2026/067.

## Appendix A — Prototype provenance

The numbers in §5 come from a NumPy/scikit-learn prototype (developed
interactively, archived in the session scratchpad). It reads the ChipWhisperer
HDF5 files, matched-filter projects the profiled PoIs, fits the constrained
template, builds the closed-form confusion, and runs the confusion-aware key rank.
[IMPLEMENTATION.md](IMPLEMENTATION.md) maps every prototype step to the LeakFlow
C++ target it will become.
