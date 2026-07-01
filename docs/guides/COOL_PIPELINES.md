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

## AES Sync CPA Best-Correlation Sample In TracePlot

This pipeline runs a full Pearson CPA attack on the checked-in AES fixtures (no
downloaded traces needed) and marks each recovered byte's best-correlation sample
on the trace browser. It loads traces, plaintexts, and the known key; builds
first-round `HW(y)` leakage hypotheses for every byte and every key guess;
ranks the guesses with `CpaAttack`; derives known-key diagnostics with
`AttackStats`; converts those into plot annotations; and prints a structured
`Summary` of the stats.

```bash
leakflow --log-level info run --graph \
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt){capture.source=ChipWhisperer; capture.sample_rate_hz=29454545.454545453; payload.leakage.range=[-0.5,0.5]}; \
   TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); \
   TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); \
   Tee@trace_tee; \
   AesLeakageHypothesis@hyp(channels=[HW(y)],byte_indexes=[],guess_values=[]); \
   CpaAttack@attack(compute_dtype=float64,correlation_mode=recompute); \
   AttackStats@stats(top_k=2); \
   Tee@stats_tee; \
   AttackStatsToPlotAnnotations@ann(precision=3); \
   TracePlot@plot(title="AES CPA byte 0 - best correlation sample",group=cpa,label=traces,x_axis=sample); \
   Summary@summary(level=3); \
   @traces_src ! @trace_tee; \
      @trace_tee.src_0 ! @plot.sink; \
      @trace_tee.src_1 ! @attack.features; \
   @plain_src ! @hyp.plaintexts; \
   @hyp ! @attack.hypotheses; \
   @attack ! @stats.scores; \
   @key_src ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @plot.annotations; \
      @stats_tee.src_1 ! @summary'
```

Notes:

- `AesLeakageHypothesis(byte_indexes=[],guess_values=[])` builds the full
  `[U,G,N,L]` hypothesis tensor over all 16 bytes and the full `0..255` guess
  domain; `CpaAttack` ranks guesses, and `AttackStats(top_k=2)` adds known-key
  diagnostics (true rank, top-1/top-2 margins) using `@key_src ! @stats.truth`.
- `AttackStatsToPlotAnnotations(precision=3)` marks each unit's best sample on the
  trace; `@trace_tee.src_0` carries the raw traces to `TracePlot.sink` while
  `@trace_tee.src_1` feeds `CpaAttack.features`.
- `Tee@stats_tee` fans the stats to both the annotation converter and a `Summary`,
  so the diagnostics print while the markers render.

## AES Sync CPA Live (Synchronized Streams)

The live counterpart of the CPA pipeline above. Traces and plaintexts stream
independently through `Queue`s; `Sync(policy=zip)` pairs trace row `N` with
plaintext row `N` before the attack, and the fixed key is loaded once by
`TorchFileSrc`. `CpaAttack(correlation_mode=auto)` resolves to incremental under a
live source, so the scores and the marked best-correlation sample refine as rows
arrive. Uses the checked-in fixtures (no downloaded traces needed).

```bash
leakflow --log-level error run --graph \
  'FakeLiveSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt,trace_rate=0){capture.source=ChipWhisperer; capture.sample_rate_hz=29454545.454545453; payload.leakage.range=[-0.5,0.5]}; \
   Queue@traces_queue(max_size=8,drop_oldest=false); \
   FakeLiveSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt,trace_rate=0); \
   Queue@plain_queue(max_size=8,drop_oldest=false); \
   TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); \
   Tee@trace_tee; \
   AesLeakageHypothesis@hyp(channels=[HW(y)],byte_indexes=[],guess_values=[]); \
   Sync@attack_sync(policy=zip); \
   CpaAttack@attack(compute_dtype=float32,correlation_mode=auto); \
   AttackStats@stats(top_k=3); \
   Tee@stats_tee; \
   AttackStatsToPlotAnnotations@ann(precision=3); \
   TracePlot@plot(title="AES CPA byte 0 - best correlation sample",group=cpa,label=traces,x_axis=sample); \
   Summary@summary(level=3); \
   @traces_src ! @traces_queue ! @trace_tee; \
      @trace_tee.src_0 ! @plot.sink; \
      @trace_tee.src_1 ! @attack_sync.in_0; \
   @plain_src ! @plain_queue ! @hyp.plaintexts; \
   @hyp ! @attack_sync.in_1; \
   @attack_sync.out_0 ! @attack.features; \
   @attack_sync.out_1 ! @attack.hypotheses; \
   @attack ! @stats.scores; \
   @key_src ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @plot.annotations; \
      @stats_tee.src_1 ! @summary'
```

Notes:

- `Sync@attack_sync(policy=zip)` is the front door for pairing the two independent
  live streams: it injects a common ancestor so `CpaAttack`'s join uses the default
  barrier. `@attack_sync.out_0`/`out_1` carry the aligned trace/hypothesis pair.
- `CpaAttack(correlation_mode=auto)` resolves to `incremental` because the inputs
  are live-driven, so the attack accumulates statistics across streamed rows; set
  `trace_rate` above `0` on the `FakeLiveSrc` sources to watch the convergence pace.
- `TracePlot` is live-driven here, so its `update_mode=auto` resolves to
  `accumulate`: the trace slider builds a scrubbable history of streamed rows.
- `capture.source`/`capture.sample_rate_hz` are user-provided here: `FakeLiveSrc`
  does not assert a `capture.source` (it is suggested metadata, like `TorchFileSrc`),
  so the `{capture.source=ChipWhisperer; capture.sample_rate_hz=...}` annotation
  supplies them. These `capture`-group facts are preserved through
  `AesLeakageHypothesis`/`CpaAttack`/`AttackStats` to the plot (e.g.
  `capture.sample_rate_hz` drives `TracePlot(x_axis=time_us)`).

## AES Sync CPA Score Convergence (ScorePlot)

`ScorePlot` (in `leakflow_plugins_crypto_plot`) consumes `AttackStats` results
directly and accumulates them into a stacked score plot: one panel per metric
(`score` plus the configured confidence metrics), one line per attack byte, a
point per streamed buffer at x = observation count. Success is drawn with the
marker shape (square = success, x = failure, circle = unknown). Under a live
source, `CpaAttack(correlation_mode=auto)` runs incrementally, so the score and
relative margin refine as rows arrive — the classic convergence view.

```bash
leakflow --log-level error run --graph \
  'FakeLiveSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt,trace_rate=0); \
   Queue@traces_queue(max_size=8,drop_oldest=false); \
   FakeLiveSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt,trace_rate=0); \
   Queue@plain_queue(max_size=8,drop_oldest=false); \
   TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); \
   AesLeakageHypothesis@hyp(channels=[HW(y)],byte_indexes=[],guess_values=[]); \
   Sync@attack_sync(policy=zip); \
   CpaAttack@attack(compute_dtype=float32,correlation_mode=auto); \
   AttackStats@stats(top_k=3); \
   ScorePlot@scores(group=cpa,title="CPA score convergence",metrics=[score,relative_margin]); \
   @traces_src ! @traces_queue ! @attack_sync.in_0; \
   @plain_src ! @plain_queue ! @hyp.plaintexts; \
   @hyp ! @attack_sync.in_1; \
   @attack_sync.out_0 ! @attack.features; \
   @attack_sync.out_1 ! @attack.hypotheses; \
   @attack ! @stats.scores; \
   @key_src ! @stats.truth; \
   @stats ! @scores'
```

Notes:

- `ScorePlot` reads the crypto `AttackStatsPayload` directly (no generic converter)
  but registers a generic score snapshot into the shared plot runtime, so it reuses
  the `group` windows and the circle/square/x markers while `leakflow_plot` stays
  domain-free.
- `metrics=[...]` selects the stacked panels: `score`, `relative_margin`, `margin`,
  `z_score`, `robust_z_score`, `top_k_separation`, and `true_rank` (truth only).
- Increase `trace_rate` on the `FakeLiveSrc` sources to watch the convergence pace;
  drop `@key_src ! @stats.truth` to run without truth (markers become circles).

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
