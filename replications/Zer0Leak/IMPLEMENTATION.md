# Zer0Leak — Implementation Guide

> Step-by-step plan to implement the confusion-aware blind AES attack
> ([ATTACK.md](ATTACK.md)) inside LeakFlow. Written to be followed closely, phase
> by phase. Every generic capability lands in framework code (`leakflow_ml`,
> `leakflow_crypto`, plugins); only attack-specific tuning lives in this
> replication app.

## 0. Design rule (where each piece goes)

LeakFlow is a **framework first**. Follow the framework-oriented rule:

- **Generic, domain-free numerics** → `leakflow_ml` (Torch-only, no core, no AES).
  - the constrained 1-D Hamming-weight template + labeling,
  - the closed-form Gaussian confusion matrix,
  - the confusion-aware maximum-likelihood ranking over *arbitrary* hypothesis
    distributions.
- **AES-specific but generic** → `leakflow_crypto`.
  - the per-key theoretical joint `(HW(m), HW(y))` distributions `J_k`.
- **Pipeline elements** → `leakflow_plugins_ml` (labeler) and
  `leakflow_plugins_crypto` (key ranker), wrapping the numerics as `Element`s.
- **Attack-specific tuning** → `replications/Zer0Leak/` (this app). Hard-codes:
  dataset paths, byte index, PoI count (8), matched-filter, orientation search,
  end-to-end wiring, and result reporting. May hard-code anything paper-specific.

Nothing about "8 PoIs", "byte 0", "ChipWhisperer", or "matched filter" belongs in
framework code — those are app choices. The framework only learns generic
capabilities (template labeling, confusion matrices, ML ranking).

## 1. Reference: prototype → C++ API map

The validated NumPy prototype has these functions (see ATTACK.md Appendix A).
Each maps to a LeakFlow API to build:

| Prototype (NumPy) | LeakFlow target | Layer |
|---|---|---|
| `corr_poi(T, model)` | `leakflow::crypto::pearson_correlation` (exists) / `PearsonCorrelator` | crypto |
| `matched_proj(T, weights, poi)` | app helper (matched filter) or `FeatureSelect`+reduce | app |
| `fit_linbin(f)` (3-param EM) | `leakflow::ml::HwTemplate::fit(...)` | ml |
| `label_gaussmath(f)` → labels + `C` | `HwTemplate::labels()`, `HwTemplate::confusion()` | ml |
| `Jk_all()` (AES 9×9 per key) | `leakflow::crypto::aes_joint_hw_distributions()` | crypto |
| `attack(fm, fy, ...)` (LL, orientation) | `leakflow::ml::confusion_aware_rank(...)` + app orientation loop | ml + app |

## 2. Data contract

Inputs are LeakFlow tensor-dataset HDF5 files (`Hdf5FileSrc` schema), same as the
Rezaeezade replication:
- `/traces` `float32 [N, S]` (here `[10000, 5000]`)
- `/plaintexts` `uint8 [N, 16]`
- `/keys` `uint8 [16]` (single key per file; used **only** to score)
- optional `/countermeasures/jitter/...` on jitter captures

Profiling and attack are two files:
- profiling (known key): `traces/aes/{sync,jitter}/aes_*_attack/key_30.h5`
- attack: `.../key_01.h5`

A precomputed correlation may be reused via `BufferFileSrc` on `out/aes_corr.h5`
(produced by the Rezaeezade PoI finder). The attack derives PoIs from that
correlation.

## 3. Phase plan

Implement in this order. Each phase is buildable, testable, and leaves the repo
green. Follow the incremental rule — one phase at a time.

### Phase 1 — Generic numeric core in `leakflow_ml`

**Files (new):**
- `include/leakflow/ml/hw_template.hpp`, `src/ml/hw_template.cpp`, `tests/ml/hw_template_test.cpp`
- `include/leakflow/ml/confusion_rank.hpp`, `src/ml/confusion_rank.cpp`, `tests/ml/confusion_rank_test.cpp`

**`HwTemplate` (constrained 1-D Hamming-weight template).**
```cpp
namespace leakflow::ml {

struct HwTemplateOptions {
    std::int64_t n_levels = 9;      // HW range 0..B  (B=8 -> 9 levels)
    std::int64_t max_iter = 200;
    // binomial class prior over levels; empty -> Binomial(B, 1/2)
    std::vector<double> class_prior;
};

struct HwTemplateFit {
    double offset = 0.0;   // a
    double slope  = 0.0;   // b   (mu_h = a + b*h)
    double sigma  = 0.0;   // shared std
    torch::Tensor labels;      // [N] int64, argmax posterior in 0..B
    torch::Tensor confusion;   // [B+1, B+1] float64, C[t,e]=Pr(label=e | true=t)
    double sigma_hw = 0.0;     // sigma / |slope|  (label-noise std in HW units)
    bool ascending = true;     // whether mu increases with h (orientation of fit)
};

// f: [N] float64 leakage projection for ONE channel. Fully blind (no truth).
[[nodiscard]] HwTemplateFit fit_hw_template(const torch::Tensor& f,
                                            const HwTemplateOptions& = {});
}
```
Algorithm (port of `fit_linbin` + `label_gaussmath`):
1. EM with fixed binomial weights, linear means, shared σ (3 params).
2. If `slope < 0`, negate `f` and refit so levels ascend with `f` (record
   `ascending`); the app resolves the physical orientation later.
3. Labels = argmax posterior.
4. Confusion `C[t,e]` via **closed-form argmax-region CDF**: boundaries where
   adjacent posteriors are equal, then `Φ((edge_e−μ_t)/σ) − Φ((edge_{e-1}−μ_t)/σ)`.
   (Use `torch::erf`/erfc for Φ. Guard the degenerate `slope≈0` case.)

**`confusion_aware_rank` (generic ML criterion — domain-free).**
```cpp
// hypotheses: [K, L, L] per-hypothesis joint distributions J_k (rows=axis0 HW,
//             cols=axis1 HW). Domain-agnostic (AES supplies them, but this fn
//             never mentions AES).
// nobs:       [L, L] observed joint histogram (counts).
// c0, c1:     [L, L] confusion matrices for axis0, axis1.
// returns:    [K] log-likelihood per hypothesis (higher = better).
[[nodiscard]] torch::Tensor confusion_aware_rank(
    const torch::Tensor& hypotheses,
    const torch::Tensor& nobs,
    const torch::Tensor& c0,
    const torch::Tensor& c1);
```
Computes `J̃_k = c0ᵀ · J_k · c1`, then `LL(k) = Σ nobs ⊙ log J̃_k`. This is the
generic engine; the app / crypto layer supplies `hypotheses`.

**Tests (`tests/ml`):**
- `fit_hw_template` on synthetic `N(a+b·h, σ)` mixtures recovers `(a,b,σ)` and a
  banded `C` summing to 1 per row; degenerate/flat inputs handled.
- `confusion_aware_rank`: with identity `C` reduces to matching clean `J_k`; with
  a known blur, the blurred-true hypothesis scores highest. Hand fixture with
  `K=3, L=3`.

CMake: add the two sources to `src/ml/CMakeLists.txt`; the two tests to
`tests/ml/CMakeLists.txt`.

### Phase 2 — AES joint HW distributions in `leakflow_crypto`

**Files:** extend `include/leakflow/crypto/aes.hpp` / `src/crypto/aes.cpp`;
add `tests/crypto/aes_joint_hw_test.cpp`.
```cpp
namespace leakflow::crypto {
// Returns [256, B+1, B+1] float64: for each key byte k, the normalized histogram
// of (HW(plaintext), HW(SBOX(plaintext XOR k))) over all 256 plaintext values.
// rows = HW(m), cols = HW(y). Depends only on the S-box — no traces, no key knowledge.
[[nodiscard]] torch::Tensor aes_joint_hw_distributions();
}
```
Uses the existing `sbox()` and Hamming-weight helpers. **Test:** all 256
histograms distinct; each sums to 1; byte-0 row/col marginals are binomial; and
for a fixed key the number of nonzero cells matches the known 42–49 range.

### Phase 3 — Pipeline elements

**`BlindHwLabel`** in `leakflow_plugins_ml` (klass `Analyze/Blind/HwLabel`):
- inputs: `features` (a `[U, N]` or `[N]` leakage projection per channel — one
  element instance per channel, or a `[N, C]` multi-channel input),
- output: an HW-label payload `[N]` in `0..B` plus the confusion matrix carried as
  a structured payload / metadata (bounded).
- wraps `fit_hw_template`. Property: `n_levels` (default 9), `class_prior`.

**`BlindKeyRank`** in `leakflow_plugins_crypto` (klass `Analyze/Blind/KeyRank`):
- inputs: two HW-label streams (`labels_m`, `labels_y`) plus their confusion
  matrices,
- builds `Nobs`, calls `crypto::aes_joint_hw_distributions()` and
  `ml::confusion_aware_rank(...)`,
- output: a key-ranking payload `[256]` (LL per key) + the recovered byte.
- Property: `byte_index`.

**Tests:** `tests/plugins/ml/blind_hw_label_test.cpp`,
`tests/plugins/crypto/blind_key_rank_test.cpp` — reproduce the rank on the tiny
checked-in `key_01` fixture (or a synthetic high-SNR fixture) so CI does not
depend on the local `traces/` tree.

### Phase 4 — The replication app (`replications/Zer0Leak/`)

**Files:** `CMakeLists.txt`, `src/blind_attack_main.cpp`, `README.md`. Register
in `replications/CMakeLists.txt` (`add_subdirectory(Zer0Leak)`), gated by
`LEAKFLOW_BUILD_REPLICATIONS`.

The app is the **tuned end-to-end attack** and hard-codes the paper's choices:
1. Load profiling + attack captures (`Hdf5FileSrc`) or reuse `out/aes_corr.h5`.
2. Compute/read the profiling correlation, pick **top-8 PoI per channel** for
   `HW(m)` and `HW(y)` (this is the app's tuning knob).
3. **Matched-filter** project the attack traces onto each channel's PoIs (app
   helper: correlation-weighted sum + standardize).
4. For each channel, `ml::fit_hw_template(...)` → labels + confusion.
5. **Orientation search**: 4 combinations (reverse labels `h→8−h` and confusion
   per axis); for each, build `Nobs`, call `crypto::aes_joint_hw_distributions()`
   + `ml::confusion_aware_rank(...)`; keep the max-LL combination.
6. Report the recovered byte and its rank; if the file's key is present, print the
   true rank (scoring only).

CLI shape (mirror the Rezaeezade app):
```bash
leakflow_zer0leak_blind_attack \
  --profiling traces/aes/jitter/aes_jitter_attack/key_30.h5 \
  --attack    traces/aes/jitter/aes_jitter_attack/key_01.h5 \
  --byte 0 --poi 8
# or reuse a saved correlation:
leakflow_zer0leak_blind_attack --correlation out/aes_corr.h5 \
  --attack traces/aes/jitter/aes_jitter_attack/key_01.h5 --byte 0 --poi 8
```

Optionally support `--graph` to visualize (reuse the plot runtime) and
`--all-bytes` to rank all 16 (groundwork for the Phase-5 key-schedule fusion).

**Validation targets (must reproduce the prototype):**
- sync `key_01` byte 0 → rank **3**
- jitter `key_01` byte 0, 8 PoI → rank **6** (and 30 PoI → 131, 100 PoI → 75)

### Phase 5 — (future) full-key fusion via the key schedule

Add a multi-byte fuser (belief propagation over the AES KeyExpansion, following
Le Bouder et al. FPS 2016) that combines the 16 per-byte LL vectors into a full
128-bit key ranking. Generic BP numerics in `leakflow_ml`; AES KeyExpansion
constraints in `leakflow_crypto`; wiring in this app. This is the honestly-blind
path to a full-key break and to pushing single-byte rank toward 1.

## 4. Build & test

Standard LeakFlow validation, plus the replication gate for Phase 4:
```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch" \
  -DLEAKFLOW_BUILD_REPLICATIONS=ON
cmake --build build -j
ctest --test-dir build -R 'leakflow_ml|leakflow_crypto|leakflow_plugins' --output-on-failure
cmake --build build --target leakflow_zer0leak_blind_attack -j
```

## 5. Design notes / gotchas (learned in the prototype)

- **Constrained template, not free GMM.** A free 9-component GMM's self-derived
  confusion fails (rank ~155). The 3-parameter linear-binomial model is what makes
  blind `C` correct. Do **not** substitute `leakflow_ml::GaussianMixture` here
  unfitted — it is the wrong tool for this step. (It remains useful for other
  clustering tasks.)
- **Confusion faithful to the labeler.** Because labeling uses the binomial prior
  (biased), the confusion must integrate over the **actual argmax regions**
  (closed-form CDF), not a symmetric Gaussian. Symmetric Eq. 5 requires the
  unbiased `h=(l−β)/α` label; if you switch to that labeling, switch `C` to match.
- **Separable `C_m`, `C_y`.** Keep the two channels' confusions independent
  (Case B). This is both correct here (independent leaks, `corr≈0.06`) and the
  form the ML criterion expects.
- **Orientation** is a per-axis sign ambiguity; resolve by max achievable LL. Keep
  it in the app (it is attack-level policy), not in the generic `HwTemplate`.
- **PoI count is a tuning knob**, not a framework constant. Fewer, stronger PoIs
  win under jitter (matched filter dilutes on desync-smeared leaks).
- **Everything blind except PoI location.** The truth (`/keys`) enters only the
  final scoring. Assert this in the app (no key bytes read before the rank step).
- **float64** for all template/rank math (matches the rest of `leakflow_ml`).

## 6. Status checklist

- [ ] Phase 1 — `HwTemplate`, `confusion_aware_rank` (+ tests) in `leakflow_ml`
- [ ] Phase 2 — `aes_joint_hw_distributions` (+ test) in `leakflow_crypto`
- [ ] Phase 3 — `BlindHwLabel`, `BlindKeyRank` elements (+ tests)
- [ ] Phase 4 — `replications/Zer0Leak` app; reproduce rank 3 (sync) / 6 (jitter)
- [ ] Phase 5 — key-schedule BP full-key fusion (future)
