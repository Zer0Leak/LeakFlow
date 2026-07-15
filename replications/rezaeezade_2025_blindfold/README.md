# Rezaeezade et al. (2025) — Breaking the Blindfold (PoI finder)

Replication of the points-of-interest (PoI) finding step from Rezaeezade et al.,
*Breaking the Blindfold: Deep Learning-based Blind Side-channel Analysis* (2025).
Later attack and deep-learning steps remain out of scope.

## Pipeline

The application reads one aligned capture file with `Hdf5FileSrc`. The file must
provide `/traces`, `/plaintexts`, and `/keys` using the LeakFlow tensor-dataset
HDF5 schema.

```text
Hdf5FileSrc@src
  traces     -> Tee@trace_tee -> PearsonCorrelator.features / AesLeakage.traces
  plaintexts -> AesLeakage.plaintexts
  keys       -> AesLeakage.keys
AesLeakage -> PearsonCorrelator.targets
PearsonCorrelator -> PoiSelect
```

The analysis covers all 16 AES bytes, the `HW(m)` and `HW(y)` channels, and
selects the top 50 absolute-correlation samples.

## Build

The target is gated by `LEAKFLOW_BUILD_REPLICATIONS` (default `OFF`):

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch" \
  -DLEAKFLOW_BUILD_REPLICATIONS=ON
cmake --build build --target leakflow_rezaeezade_poi_finder -j
```

## Run

```bash
# Headless; prints the PoI summary.
./build/leakflow_rezaeezade_poi_finder traces/aes/sync/aes_sync_poi/key_01.h5

# Pipeline graph and trace/PoI display.
./build/leakflow_rezaeezade_poi_finder --graph \
  traces/aes/sync/aes_sync_poi/key_01.h5

# Persist the correlation for later PoI selection.
./build/leakflow_rezaeezade_poi_finder \
  --save-correlation out/aes_corr.h5 \
  traces/aes/sync/aes_sync_poi/key_01.h5
```

The default input is `traces/aes/sync/aes_sync_poi/key_01.h5`. Use
`--auto-start` with `--graph` to begin immediately.

Reload a saved correlation without recomputing it:

```bash
leakflow run 'BufferFileSrc(path=out/aes_corr.h5) ! PoiSelect(top_k=[5],rank_by=[abs]) ! Summary ! FakeSink'
```
