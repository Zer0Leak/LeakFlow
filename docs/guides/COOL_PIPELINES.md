# Cool Pipelines

This guide collects demonstration-ready LeakFlow pipelines. Run commands from
the repository root.

## Build First

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
```

If `leakflow` is not installed or on your `PATH`, replace `leakflow` in the
commands below with `./build/leakflow`.

To build the C++ tutorial app that uses the same pipeline-expression API from a
linked application, enable examples:

```bash
CXX=clang++ cmake -S . -B build-examples -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLEAKFLOW_BUILD_EXAMPLES=ON -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build-examples -j
./build-examples/leakflow_pipeline_expression_plot_tutorial
```

The tutorial constructs the AES PoI TracePlot pipeline from a string, retrieves
`TracePlot@plot` by name, lists `Tee` elements by descriptor type, then runs the
pipeline and opens the plot runtime.

## Download Demo Traces

Download the AES synchronized trace data from
[the shared Google Drive folder](https://drive.google.com/drive/folders/1OnaoI5YBPkW_ancz3OfvsPiJQmHH1VV4?usp=sharing).

Place the downloaded `traces` content under the repository `traces/` directory.
For the AES synchronized data, the dataset should live under:

```text
traces/aes/sync/
```

The AES sync PoI demo below expects these files:

```text
traces/aes/sync/aes_sync_poi/key_01/traces.pt
traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt
traces/aes/sync/aes_sync_poi/key_01/key.pt
```

## AES Sync Trace Metadata Smoke

Use this small pipeline to check the trace file and see the source metadata in
the buffer-flow log.

```bash
leakflow --log-level info run \
  'TorchFileSrc@traces_src(path=traces/aes/sync/aes_sync_poi/key_01/traces.pt){capture.source=ChipWhisperer; payload.leakage.inverted=false; payload.leakage.range=[-0.5,0.5]; capture.sample_rate_hz=29454545.454545453} ! FakeSink'
```

`TorchFileSrc` also stamps file provenance metadata such as
`origin.file.format=torch-tensor`, `origin.file.path`, and `origin.file.size`.

Every metadata key carries its group as the leading segment. `capture.source`,
`capture.sample_rate_hz`, and `capture.dataset.name` are `capture`-group metadata
that survive through downstream analysis. `payload.leakage.inverted` (declared by
the user here) and `payload.leakage.range` describe the trace data itself, so they
are `payload`-group: preserved on pass-through hops but dropped when an analysis
element derives a new buffer. `TorchFileSrc` does not stamp `payload.leakage.inverted`
itself; its `invert` property only flips the tensor (×−1). See
`docs/design/metadata_klass_taxonomy.md`.

## Checked-In Torch Caps Smoke

Use this pipeline to see the three CLI annotation forms side by side on a small
fixture that is checked into the repository:

```bash
leakflow --log-level info run \
  'TorchFileSrc@traces \
      (path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) \
      {capture.dataset.name=aes_sync_fixture; origin.role=traces}; \
  Summary@summary(level=3); \
  @traces.src[caps=leakflow/torch-tensor; dtype=float32; device=cpu; rank=2] ! @summary.sink'
```

In this command:

- `(path=...)` sets the `TorchFileSrc` property.
- `{capture.dataset.name=...; origin.role=...}` stamps buffer metadata on
  `TorchFileSrc` output.
- `[caps=leakflow/torch-tensor; dtype=float32; device=cpu; rank=2]` annotates
  the `traces.src` pad caps.

Caps annotations are parsed and validated as CLI intent. The runtime buffer
caps still come from the element and payload until a later caps-negotiation
phase makes annotations mutable.

## AES Sync Pearson PoIs In TracePlot

This pipeline loads AES synchronized traces, computes first-round AES S-box
Hamming-weight leakage for byte `0`, finds the top absolute Pearson-correlation
PoIs, converts them to plot annotations, and opens TracePlot with the selected
PoIs overlaid on the trace browser.

# No metadata version

```bash
leakflow --log-level info run \
  'TorchFileSrc@traces_src(path=traces/aes/sync/aes_sync_poi/key_01/traces.pt); Tee@traces; TorchFileSrc@plain_src(path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt); TorchFileSrc@key_src(path=traces/aes/sync/aes_sync_poi/key_01/key.pt); AesLeakage@leakage(byte_indexes=[0]); PearsonPoiFinder@poi(top_k=[50],rank_by=[abs]); CorrelationPoiToPlotAnnotations@ann(precision=3); TracePlot@plot(title="AES bytes 3 and 5 PoIs",group=aes,label=traces,x_axis=sample); @traces_src ! @traces; @traces.src_0 ! @poi.features; @traces.src_1 ! @plot.sink; @traces.src_2 ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @poi.targets; @poi ! @ann ! @plot.annotations'
```
# Metadata version

```bash
leakflow --log-level info run --graph \
  'TorchFileSrc@traces_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/traces.pt) \
      {capture.source=ChipWhisperer; payload.leakage.inverted=false; payload.leakage.range=[-0.5,0.5]; capture.sample_rate_hz=29454545.454545453}; \
  TorchFileSrc@plain_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt); \
  TorchFileSrc@key_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/key.pt); \
  Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=trace-fanout}; \
      @trace_tee.src_0{routing.branch=plot-traces}; \
      @trace_tee.src_1{routing.branch=analysis-poi}; \
      @trace_tee.src_2{routing.branch=analysis-leakage}; \
  AesLeakage@leakage \
      (channels=[HW(y)],byte_indexes=[0]); \
  PearsonPoiFinder@poi \
      (top_k=[50],rank_by=[abs]); \
  CorrelationPoiToPlotAnnotations@ann \
      (precision=3); \
  TracePlot@plot \
      (title="AES byte 0 PoIs",group=aes,label=traces,x_axis=sample); \
  @traces_src ! @trace_tee; \
                    @trace_tee.src_0 ! @plot.sink; \
                    @trace_tee.src_1 ! @poi.features; \
                    @trace_tee.src_2 ! @leakage.traces; \
  @plain_src ! @leakage.plaintexts; \
                                              @leakage ! @poi.targets; \
  @key_src   ! @leakage.keys; \
  @poi ! @ann ! @plot.annotations'
```

## Live Synchronized Version

This version streams traces and plaintexts independently through queues, then
uses `Sync(policy=zip)` to pair trace row `N` with plaintext row `N`. The fixed
AES key is loaded once by `TorchFileSrc` and held by `AesLeakage` across live
updates.

```bash
leakflow --log-level info run --graph \
  'FakeLiveSrc@traces_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/traces.pt,trace_rate=1.0) \
      {payload.leakage.inverted=false;payload.leakage.range=[-0.5,0.5];capture.sample_rate_hz=29454545.454545453}; \
   Queue@traces_queue(max_size=8,drop_oldest=false); \
   FakeLiveSrc@plain_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt,trace_rate=1.0); \
   Queue@plain_queue(max_size=8,drop_oldest=false); \
   TorchFileSrc@key_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/key.pt); \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=trace-fanout}; \
      @trace_tee.src_0{routing.branch=plot-traces}; \
      @trace_tee.src_1{routing.branch=analysis-poi}; \
      @trace_tee.src_2{routing.branch=analysis-leakage}; \
   Sync@leakage_sync(policy=zip); \
   AesLeakage@leakage(channels=[HW(y)],byte_indexes=[0]); \
   PearsonPoiFinder@poi(top_k=[50],rank_by=[abs]); \
   CorrelationPoiToPlotAnnotations@ann(precision=3); \
   TracePlot@plot \
      (title="AES byte 0 PoIs",group=aes,label=traces,x_axis=sample); \
   @traces_src ! @traces_queue ! @trace_tee; \
      @trace_tee.src_0 ! @plot.sink; \
      @trace_tee.src_1 ! @poi.features; \
      @trace_tee.src_2 ! @leakage_sync.in_0; \
   @plain_src ! @plain_queue; \
      @plain_queue.src ! @leakage_sync.in_1; \
   @leakage_sync.out_0 ! @leakage.traces; \
   @leakage_sync.out_1 ! @leakage.plaintexts; \
   @key_src ! @leakage.keys; \
   @leakage ! @poi.targets; \
   @poi ! @ann ! @plot.annotations'
```

Notes:

- `Tee@trace_tee` fans the trace tensor to three branches: `TracePlot.sink`
  (plot), `PearsonPoiFinder.features` (correlation), and `AesLeakage.traces`
  (leakage alignment check).
- `FakeLiveSrc@traces_src{...}` carries explicit user `capture` facts
  (`capture.sample_rate_hz`) that are unioned and preserved through
  `AesLeakage`, `PearsonPoiFinder`, and `CorrelationPoiToPlotAnnotations` all
  the way to the plot. The trace-describing `payload.leakage.inverted`/
  `payload.leakage.range` are `payload`-group: they ride the pass-through
  traces→plot branch but are dropped when `AesLeakage`/`PearsonPoiFinder`
  derive new buffers.
- The `@trace_tee.src_%u{routing.branch.family=...}` and exact
  `@trace_tee.src_N{routing.branch=...}` annotations are `routing`-group metadata.
  They ride each buffer to its immediate consumer (and appear in `--graph`/logs),
  but are not forwarded onto derived buffers such as the leakage tensor or PoI
  results.
- File provenance (`origin.file.path`, `origin.file.size`, `origin.file.format`)
  is `origin`-group. Pass-through hops keep it as-is, but the multi-input
  `AesLeakage` and `PearsonPoiFinder` relabel each input's origin as
  `origin.<pad>.<key>` (for example `origin.plaintexts.file.path`,
  `origin.keys.file.path`, `origin.features.file.path`) so the fused buffer keeps
  unambiguous provenance.
- `AesLeakage(channels=[HW(m)],byte_indexes=[0])` creates the `HW(m)` target
  for AES byte `0` and stamps its own `payload.leakage.*`/`payload.crypto.*`/
  `payload.trace.*` metadata.
- `PearsonPoiFinder(top_k=[50],rank_by=[abs])` selects the strongest absolute
  correlations per target, re-owns the target model's
  `payload.leakage.*`/`payload.crypto.*` facts, and adds `payload.poi.*` results.
- `CorrelationPoiToPlotAnnotations(precision=3)` turns those PoIs into
  `TracePlot.annotations` markers with rounded display text.
- Change `byte_indexes=[0]` and the `TracePlot` title together when preparing a
  demo for other AES state bytes.
- Metadata grouping and forwarding rules are documented in
  `docs/design/metadata_klass_taxonomy.md`.
