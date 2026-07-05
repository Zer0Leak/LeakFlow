# Rezaeezade et al. (2025) — Breaking the Blindfold (PoI finder)

Replication of the **points-of-interest (PoI) finding** step from:

> Rezaeezade et al., *Breaking the Blindfold: Deep Learning-based Blind
> Side-channel Analysis*, 2025.
> PDF: [`../../papers/Rezaeezade et al. - 2025 - Breaking the Blindfold Deep Learning-based Blind Side-channel Analysis.pdf`](../../papers/)

Only the PoI finder is replicated here. Later steps (attack / DL model) are out of
scope for now.

## What it does

Aggregates the Pearson PoI finder across **many capture folders**. Each `key_*`
folder is one aligned `(traces, plaintexts, key)` capture. The app streams one
folder per step into an `AppSrc` (application-fed live source). Because `AppSrc`
declares itself live, `PearsonPoiFinder` auto-selects its **incremental** mode and
folds each folder into running correlation moments — it never resets between
folders, only at `start()`. The final emitted PoI is the aggregate over every
folder.

Pipeline (built once by the app, then fed frame-by-frame):

```
AppSrc@src
  src_0 (traces) ─► Tee@trace_tee ─┬─► @poi.features
  │                                └─► @leakage.traces
  src_1 (plain)  ────────────────────► @leakage.plaintexts
  src_2 (key)    ────────────────────► @leakage.keys
AesLeakage@leakage (channels=[HW(m),HW(y)]) ─► @poi.targets
PearsonPoiFinder@poi (top_k=[50], rank_by=[abs])   # auto → incremental (live)
```

One `AppSrc` step = one folder = one buffer per pad, all stamped with a single
shared vector clock, so `AesLeakage` and `PearsonPoiFinder` pair the three inputs
per folder with the default barrier (no `Sync` element needed).

## Data layout

Default root: `traces/aes/sync/aes_sync_poi/` (not in the repository). Each capture
folder must contain single Torch tensors:

```
traces/aes/sync/aes_sync_poi/
├── key_01/
│   ├── traces.pt        # float32 [n_traces, n_samples]
│   ├── plain_texts.pt   # uint8   [n_traces, 16]
│   └── key.pt           # uint8   [16]
├── key_02/ ...
└── key_NN/ ...
```

## Build

Gated behind `LEAKFLOW_BUILD_REPLICATIONS` (default OFF):

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch" \
  -DLEAKFLOW_BUILD_REPLICATIONS=ON
cmake --build build --target leakflow_rezaeezade_poi_finder -j
```

## Run

```bash
# headless: prints the aggregate PoI summary
./build/leakflow_rezaeezade_poi_finder traces/aes/sync/aes_sync_poi

# live pipeline graph window (watch folders stream through, PoI sharpen)
./build/leakflow_rezaeezade_poi_finder --graph traces/aes/sync/aes_sync_poi
```

`ROOT_DIR` defaults to `traces/aes/sync/aes_sync_poi`. Pass `--help` for usage.

## Notes / current limitations

- Folders are pulled **lazily**: `AppSrc` asks the app for the next folder when the
  pump needs it (one-ahead prefetch), so the `--graph` window opens immediately and
  only ~one folder is in memory at a time.
- `--graph` shows the pipeline topology + live buffer flow **and** a `TracePlot` of
  the current folder's traces with the accumulated PoIs overlaid as annotation
  markers (`CorrelationPoiToPlotAnnotations ! TracePlot`, `update_mode=replace`).
  The markers sharpen as more folders stream in. Headless mode ends at `@poi` and
  prints the aggregate PoI summary instead (no plot).
- The default analysis covers all 16 key bytes × 2 channels (`HW(m)`, `HW(y)`) ×
  `top_k=50`, so the plot shows many markers. Narrow `AesLeakage(byte_indexes=[0])`
  or lower `top_k` for a cleaner view.
- Stop → Start re-streams from the first folder: `AppSrc` owns the folder index and
  rewinds on `start()`.
- Saving the PoI result to disk (`PayloadFileSink`) is a separate, later step.
