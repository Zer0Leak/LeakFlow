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

The AES sync demos below expect these files:

```text
traces/aes/sync/aes_sync_poi/key_01/traces.pt
traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt
traces/aes/sync/aes_sync_poi/key_01/key.pt
```

For a quick smoke run without the downloaded dataset, replace those paths with
the checked-in fixtures:

```text
tests/fixtures/aes/sync/key_01/traces_first_50.pt
tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt
tests/fixtures/aes/sync/key_01/key_first_50.pt
```

## Plot View Set

The attack pipelines below keep one focused offline and one focused live example
for CPA, then the same pair for DPA.

CPA examples open all three attack views:

- `TracePlot`: raw traces with best-sample annotations from `AttackStats`.
- `ScorePlot`: score and confidence curves.
- `ScoreTablePlot`: the scoreboard, with ranked guesses per attack unit.

DPA examples open the same three views plus a second `TracePlot` for
`DpaAttack.best_difference`, the generated difference-of-means trace for the
current best guess/channel per unit.

All examples use `--graph` so the pipeline graph, property controls, vector-clock
state, and plot windows are visible together.

To profile any of the four pipelines, add the profiling run options after `run`,
for example:

```text
run --print-profile --profile-file build/leakflow-profile.json --graph
```

`--print-profile` prints the per-element timing table at exit, while
`--profile-file` writes Chrome Trace Event JSON for `chrome://tracing` or
Perfetto.

## AES Sync CPA Offline - Trace, Score, Scoreboard

This one-shot CPA run loads full tensors with `TorchFileSrc`, so no `Queue` or
`Sync` is needed. It is a good final-result view: the trace plot shows the raw
traces with best-sample markers, `ScorePlot` shows the final score/confidence
state, and `ScoreTablePlot` shows the recovered-key scoreboard.

```bash
leakflow --log-level warning run --graph \
  'TorchFileSrc@traces_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/traces.pt) \
      {capture.source=ChipWhisperer; capture.dataset.name=aes_sync_poi; capture.sample_rate_hz=29454545.454545453; origin.role=traces; payload.leakage.inverted=false; payload.leakage.range=[-0.5,0.5]}; \
   TorchFileSrc@plain_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt) \
      {capture.dataset.name=aes_sync_poi; origin.role=plaintexts}; \
   TorchFileSrc@key_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/key.pt) \
      {capture.dataset.name=aes_sync_poi; origin.role=key; payload.crypto.algorithm=AES-128}; \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=cpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[HW(y)],byte_indexes=[],guess_values=[]); \
   CpaAttack@attack \
      (score_method=max_abs,score_channels=guess_dependent,compute_dtype=float64,correlation_mode=recompute,emit_correlations=false,top_k=8); \
   AttackStats@stats \
      (top_k=8,confidence_metrics=[relative_margin,z_score,robust_z_score]); \
   Tee@stats_tee; \
      @stats_tee.src_%u{routing.branch.family=attack-stats}; \
      @stats_tee.src_0{routing.branch=trace-annotations}; \
      @stats_tee.src_1{routing.branch=score-plot}; \
      @stats_tee.src_2{routing.branch=scoreboard}; \
   AttackStatsToPlotAnnotations@ann(precision=3); \
   TracePlot@trace_plot \
      (title="AES CPA - raw traces with best samples",group=cpa,label=raw-traces,x_axis=time_us,update_mode=replace,annotation_update_mode=replace,center0=true); \
   ScorePlot@score_plot \
      (group=cpa,title="CPA score and confidence",metrics=[score,relative_margin,true_rank],show_second_score=true,max_units=16); \
   ScoreTablePlot@scoreboard \
      (group=cpa,title="CPA scoreboard",sort=score,rows=8,max_history=1); \
   @traces_src ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack.features; \
   @plain_src ! @hyp.plaintexts; \
   @hyp ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @key_src ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @trace_plot.annotations; \
      @stats_tee.src_1 ! @score_plot; \
      @stats_tee.src_2 ! @scoreboard'
```

Notes:

- `byte_indexes=[]` and `guess_values=[]` mean all 16 AES bytes and the full
  `0..255` guess domain.
- `ScorePlot(show_second_score=true)` needs `AttackStats(top_k>=2)`; this command
  uses `top_k=8`, which also controls the scoreboard row budget.
- `TracePlot(x_axis=time_us)` reads `capture.sample_rate_hz` from the trace
  metadata. If you remove that metadata, it falls back to sample indexes.

## AES Sync CPA Live - Trace, Score, Scoreboard

This live CPA run streams traces and plaintexts independently, crosses a real
`Queue` on each stream, and uses `Sync(policy=zip)` to pair trace row `N` with
plaintext row `N`. `CpaAttack(correlation_mode=auto)` resolves to incremental
under live inputs, so the score plot and scoreboard update as observations arrive.

```bash
leakflow --log-level warning run --graph \
  'FakeLiveSrc@traces_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/traces.pt,trace_rate=0) \
      {capture.source=ChipWhisperer; capture.dataset.name=aes_sync_poi; capture.sample_rate_hz=29454545.454545453; origin.role=traces; payload.leakage.range=[-0.5,0.5]}; \
   Queue@traces_queue(max_size=8,drop_oldest=false); \
   FakeLiveSrc@plain_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt,trace_rate=0) \
      {capture.dataset.name=aes_sync_poi; origin.role=plaintexts}; \
   Queue@plain_queue(max_size=8,drop_oldest=false); \
   TorchFileSrc@key_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/key.pt) \
      {capture.dataset.name=aes_sync_poi; origin.role=key; payload.crypto.algorithm=AES-128}; \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=live-trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=cpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[HW(y)],byte_indexes=[],guess_values=[]); \
   Sync@attack_sync(policy=zip); \
   CpaAttack@attack \
      (score_method=max_abs,score_channels=guess_dependent,compute_dtype=float32,correlation_mode=auto,top_k=8); \
   AttackStats@stats \
      (top_k=8,confidence_metrics=[relative_margin,z_score,robust_z_score]); \
   Tee@stats_tee; \
      @stats_tee.src_%u{routing.branch.family=live-attack-stats}; \
      @stats_tee.src_0{routing.branch=trace-annotations}; \
      @stats_tee.src_1{routing.branch=score-plot}; \
      @stats_tee.src_2{routing.branch=scoreboard}; \
   AttackStatsToPlotAnnotations@ann(precision=3); \
   TracePlot@trace_plot \
      (title="Live AES CPA - traces and current best samples",group=cpa-live,label=streamed-traces,x_axis=time_us,update_mode=accumulate,annotation_update_mode=replace,center0=true); \
   ScorePlot@score_plot \
      (group=cpa-live,title="Live CPA score convergence",metrics=[score,relative_margin,true_rank],show_second_score=true,max_units=16); \
   ScoreTablePlot@scoreboard \
      (group=cpa-live,title="Live CPA scoreboard",sort=score,rows=8,max_history=50); \
   @traces_src ! @traces_queue ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack_sync.in_0; \
   @plain_src ! @plain_queue ! @hyp.plaintexts; \
   @hyp ! @attack_sync.in_1; \
   @attack_sync.out_0 ! @attack.features; \
   @attack_sync.out_1 ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @key_src ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @trace_plot.annotations; \
      @stats_tee.src_1 ! @score_plot; \
      @stats_tee.src_2 ! @scoreboard'
```

Notes:

- `trace_rate=0` streams as fast as possible. Use `trace_rate=1.0` to watch the
  score separation and scoreboard evolve row by row.
- `TracePlot(update_mode=accumulate)` keeps the streamed trace history; the
  annotation mode is `replace`, so the markers show the current attack state.
- Drop `@key_src ! @stats.truth` to run without known-key truth. The scoreboard
  still ranks guesses, while truth/rank/success fields disappear.

## AES Sync DPA Offline - Trace, Difference Trace, Score, Scoreboard

This one-shot DPA run uses binary AES S-box bit hypotheses and
`DpaAttack.best_difference` to show the generated difference-of-means traces.
The raw trace plot still receives `AttackStats` annotations, while the second
`TracePlot` shows the current best difference trace for each attack unit.

```bash
leakflow --log-level warning run --graph \
  'TorchFileSrc@traces_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/traces.pt) \
      {capture.source=ChipWhisperer; capture.dataset.name=aes_sync_poi; capture.sample_rate_hz=29454545.454545453; origin.role=traces; payload.leakage.inverted=false; payload.leakage.range=[-0.5,0.5]}; \
   TorchFileSrc@plain_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt) \
      {capture.dataset.name=aes_sync_poi; origin.role=plaintexts}; \
   TorchFileSrc@key_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/key.pt) \
      {capture.dataset.name=aes_sync_poi; origin.role=key; payload.crypto.algorithm=AES-128}; \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=dpa-trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=dpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[y(0),y(3),y(7)],byte_indexes=[],guess_values=[]); \
   DpaAttack@attack \
      (score_method=max_abs,score_channels=guess_dependent,compute_dtype=float32,accumulation_mode=recompute,top_k=8); \
   AttackStats@stats \
      (top_k=8,confidence_metrics=[relative_margin,z_score,robust_z_score]); \
   Tee@stats_tee; \
      @stats_tee.src_%u{routing.branch.family=dpa-attack-stats}; \
      @stats_tee.src_0{routing.branch=trace-annotations}; \
      @stats_tee.src_1{routing.branch=score-plot}; \
      @stats_tee.src_2{routing.branch=scoreboard}; \
   AttackStatsToPlotAnnotations@ann(precision=3); \
   TracePlot@trace_plot \
      (title="AES DPA - raw traces with best samples",group=dpa,label=raw-traces,x_axis=time_us,update_mode=replace,annotation_update_mode=replace,center0=true); \
   TracePlot@diff_plot \
      (title="AES DPA - best difference traces",group=dpa,label=best-difference,layout=stacked,x_axis=sample,update_mode=replace,trace_context_label=unit,center0=true); \
   ScorePlot@score_plot \
      (group=dpa,title="DPA score and confidence",metrics=[score,relative_margin,true_rank],show_second_score=true,max_units=16); \
   ScoreTablePlot@scoreboard \
      (group=dpa,title="DPA scoreboard",sort=score,rows=8,max_history=1); \
   @traces_src ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack.features; \
   @plain_src ! @hyp.plaintexts; \
   @hyp ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @attack.best_difference ! @diff_plot.sink; \
   @key_src ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @trace_plot.annotations; \
      @stats_tee.src_1 ! @score_plot; \
      @stats_tee.src_2 ! @scoreboard'
```

Notes:

- `y(0)`, `y(3)`, and `y(7)` are binary S-box output-bit hypotheses. Narrow to
  `channels=[y(0)],byte_indexes=[0]` for a faster interactive demo.
- `DpaAttack.best_difference` emits a CPU float32 `[attack_unit,sample]` tensor,
  so `TracePlot(trace_context_label=unit)` gives the slider unit semantics.
- The raw trace and difference trace share `group=dpa`; `layout=stacked` puts the
  generated difference traces under the raw trace panel.

## AES Sync DPA Live - Trace, Difference Trace, Score, Scoreboard

This live DPA version mirrors the live CPA shape. Traces and plaintexts stream
through queues, `Sync(policy=zip)` aligns them, and
`DpaAttack(accumulation_mode=auto)` resolves to incremental. The extra
`TracePlot@diff_plot` replaces its snapshot each update so it always shows the
latest best-difference state.

```bash
leakflow --log-level warning run --graph \
  'FakeLiveSrc@traces_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/traces.pt,trace_rate=0) \
      {capture.source=ChipWhisperer; capture.dataset.name=aes_sync_poi; capture.sample_rate_hz=29454545.454545453; origin.role=traces; payload.leakage.range=[-0.5,0.5]}; \
   Queue@traces_queue(max_size=8,drop_oldest=false); \
   FakeLiveSrc@plain_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt,trace_rate=0) \
      {capture.dataset.name=aes_sync_poi; origin.role=plaintexts}; \
   Queue@plain_queue(max_size=8,drop_oldest=false); \
   TorchFileSrc@key_src \
      (path=traces/aes/sync/aes_sync_poi/key_01/key.pt) \
      {capture.dataset.name=aes_sync_poi; origin.role=key; payload.crypto.algorithm=AES-128}; \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=live-dpa-trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=dpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[y(0),y(3),y(7)],byte_indexes=[],guess_values=[]); \
   Sync@attack_sync(policy=zip); \
   DpaAttack@attack \
      (score_method=max_abs,score_channels=guess_dependent,compute_dtype=float32,accumulation_mode=auto,top_k=8); \
   AttackStats@stats \
      (top_k=8,confidence_metrics=[relative_margin,z_score,robust_z_score]); \
   Tee@stats_tee; \
      @stats_tee.src_%u{routing.branch.family=live-dpa-attack-stats}; \
      @stats_tee.src_0{routing.branch=trace-annotations}; \
      @stats_tee.src_1{routing.branch=score-plot}; \
      @stats_tee.src_2{routing.branch=scoreboard}; \
   AttackStatsToPlotAnnotations@ann(precision=3); \
   TracePlot@trace_plot \
      (title="Live AES DPA - traces and current best samples",group=dpa-live,label=streamed-traces,x_axis=time_us,update_mode=accumulate,annotation_update_mode=replace,center0=true); \
   TracePlot@diff_plot \
      (title="Live AES DPA - current best difference traces",group=dpa-live,label=best-difference,layout=stacked,x_axis=sample,update_mode=replace,trace_context_label=unit,center0=true); \
   ScorePlot@score_plot \
      (group=dpa-live,title="Live DPA score convergence",metrics=[score,relative_margin,true_rank],show_second_score=true,max_units=16); \
   ScoreTablePlot@scoreboard \
      (group=dpa-live,title="Live DPA scoreboard",sort=score,rows=8,max_history=50); \
   @traces_src ! @traces_queue ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack_sync.in_0; \
   @plain_src ! @plain_queue ! @hyp.plaintexts; \
   @hyp ! @attack_sync.in_1; \
   @attack_sync.out_0 ! @attack.features; \
   @attack_sync.out_1 ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @attack.best_difference ! @diff_plot.sink; \
   @key_src ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @trace_plot.annotations; \
      @stats_tee.src_1 ! @score_plot; \
      @stats_tee.src_2 ! @scoreboard'
```

Notes:

- `DpaAttack` incremental mode keeps group sums and counts, not raw traces. The
  best-difference plot is generated output, not a copy of the source traces.
- `ScoreTablePlot(max_history=50)` keeps a scrub history of the live scoreboard.
  Set it to `1` for current-only replacement.
- The DPA examples intentionally use bit hypotheses. `DpaAttack` rejects
  non-binary hypotheses, while CPA accepts numeric leakage hypotheses.

## Save An Expensive Correlation, Re-select PoIs Offline

`BufferFileSink` persists a whole buffer (caps + metadata + payload) to a `.lfbuf`
directory; `BufferFileSrc` reloads it. Because the expensive step is the
correlation and `PoiSelect` is cheap, you can compute the correlation once, save it,
then reload and try any `top_k`/`rank_by` **offline, with no re-streaming**.

Compute and save the correlation:

```bash
leakflow --log-level warning run \
  'TorchFileSrc@t(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); \
   TorchFileSrc@p(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); \
   TorchFileSrc@k(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); \
   Tee@tee; AesLeakage@lk(channels=[HW(m),HW(y)],byte_indexes=[]); \
   PearsonCorrelator@corr; BufferFileSink@save(path=out/aes_corr.lfbuf); \
   @t ! @tee; @tee.src_0 ! @corr.features; @tee.src_1 ! @lk.traces; \
   @p ! @lk.plaintexts; @k ! @lk.keys; @lk ! @corr.targets; @corr ! @save'
```

`out/aes_corr.lfbuf/manifest.txt` is human-readable (`caps.type=leakflow/correlation`,
the leakage metadata, `payload.correlation.observation_count=50`, ...) and
`payload.pt` holds the correlation tensor.

### Streaming a whole profiling set (a folder of keys)

The save command above reads a single `.pt` file. Real PoI profiling uses **many
traces across several keys** -- e.g. `traces/aes/sync/aes_sync_poi` holds 50 `key_*`
folders (each with `traces.pt` / `plain_texts.pt` / `key.pt`), 100 000 traces in all.
Loading all of that into one buffer is wasteful, so the replication app streams it:

```bash
leakflow_rezaeezade_poi_finder --save-correlation out/aes_corr.lfbuf --graph \
  traces/aes/sync/aes_sync_poi
```

It walks each `key_*` folder in turn, feeding that folder's traces/plaintexts/keys
through the same `AesLeakage -> PearsonCorrelator` chain and **accumulating** the
Pearson correlation over every key and trace -- an `AppSrc` pulls one folder at a time,
so only ~one folder is ever resident in memory. `--graph` opens the live pipeline graph
(same controls as `leakflow run --graph`) so you watch the aggregate correlation and its
PoIs build up as folders stream in; drop `--graph` for a headless run, add `--auto-start`
to begin on open. `--save-correlation PATH` writes the **same** `out/aes_corr.lfbuf` the
reload-and-plot command below consumes -- so a folder-scale, multi-key correlation drops
straight into the offline PoI-selection workflow.

You can get an identical result from the plain `leakflow run` save pipeline above by
first **merging** the folder into single `traces.pt` / `plain_texts.pt` / `key.pt`
tensors -- concatenate every `key_*` folder, expanding each folder's `[16]` key to one
row per trace so the keys stay per-trace -- and pointing the three `TorchFileSrc`s at
them. That swaps the app's streaming for a one-shot load of the whole set (fine for a
subset, heavy for all 100 000 traces), which is exactly why the folder-walking app
exists.

Reload the correlation and plot the PoIs. The expensive correlation is **not**
recomputed (`@corr_src ! @poi` re-selects from the cached correlation); the traces are
reloaded only as the cheap backdrop the PoI markers are drawn on:

```bash
leakflow run --graph \
  'TorchFileSrc@traces_src \
      (path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) \
      {capture.sample_rate_hz=29454545.454545453; origin.role=traces}; \
   BufferFileSrc@corr_src(path=out/aes_corr.lfbuf); \
   PoiSelect@poi(top_k=[3],rank_by=[abs]); \
   Tee@poi_tee; \
   CorrelationPoiToPlotAnnotations@ann(precision=3); \
   TracePlot@plot(title="Cached Pearson PoIs",group=poi,label=traces,x_axis=time_us,annotation_update_mode=replace); \
   Summary@summary(level=2); \
   FakeSink@sink; \
   @traces_src ! @plot.sink; \
   @corr_src ! @poi ! @poi_tee; \
      @poi_tee.src_0 ! @ann ! @plot.annotations; \
      @poi_tee.src_1 ! @summary ! @sink'
```

Change `top_k`/`rank_by` and re-run as often as you like — the correlation is not
recomputed. This is the cross-session complement to editing `PoiSelect` properties
live in `--graph` (which re-selects from the cached correlation in Idle).

## How Good Are The Profiling PoIs On Attack Traces?

PoIs are leakage *locations* — the sample offsets where the implementation leaks —
so they are selected once during profiling (the correlation saved to
`out/aes_corr.lfbuf` above) and then reused against a fresh capture. This pipeline
asks whether that reuse is justified: it re-runs the Pearson correlation **only at
the saved PoI columns** on new traces (`PoiCorrelation`), then lays the attack
scores next to the profiling scores in a comparison table (`PoiTablePlot`) with a
line under it per phase. Strong transfer keeps the `attack` row tracking the
`profiling` row; desync, jitter, or a different device shows up as the attack line
collapsing.

```bash
leakflow run --graph \
  'BufferFileSrc@corr_src(path=out/aes_corr.lfbuf); \
   PoiSelect@poi(top_k=[20],rank_by=[abs]); \
   Tee@poi_tee; \
   TorchFileSrc@attack_traces(path=traces/aes/sync/aes_sync_poi/key_01/traces.pt); \
   TorchFileSrc@attack_plain(path=traces/aes/sync/aes_sync_poi/key_01/plain_texts.pt); \
   TorchFileSrc@attack_key(path=traces/aes/sync/aes_sync_poi/key_01/key.pt); \
   Tee@trace_tee; \
   AesLeakage@leakage(channels=[HW(m),HW(y)],byte_indexes=[]); \
   PoiCorrelation@poicorr; \
   PoiTablePlot@tbl \
      (title="Profiling vs attack PoIs",reference_label=profiling,current_label=attack,precision=3); \
   @corr_src ! @poi ! @poi_tee; \
      @poi_tee.src_0 ! @tbl.reference; \
      @poi_tee.src_1 ! @poicorr.poi; \
   @attack_traces ! @trace_tee; \
      @trace_tee.src_0 ! @poicorr.traces; \
      @trace_tee.src_1 ! @leakage.traces; \
   @attack_plain ! @leakage.plaintexts; \
   @attack_key ! @leakage.keys; \
   @leakage ! @poicorr.targets; \
   @poicorr ! @tbl.current'
```

Notes:

- The comparison window (titled *Profiling vs attack PoIs*) has `unit` and
  `channel` sliders, a `metric` toggle (`value` vs `|value|`), and a `sort`
  selector (`sample` index — the default — or by `profiling` / `attack` score).
  Each column highlights the phase with the larger score (`profiling` blue,
  `attack` orange), and the line under the table redraws to match the metric and
  sort.
- To actually measure *transfer*, point the three `@attack_*` sources at a capture
  **other** than the one the correlation was profiled on — a different `key_*`
  folder under `aes_sync_poi`, a resynchronized set, or another device. Pointing
  them at the profiling key (as above) is the sanity baseline where both rows
  should agree.
- `PoiCorrelation` recomputes correlation over every unit at each PoI column, so it
  scales with trace count. `key_01` (2 000 traces) is interactive; for a larger
  attack set, re-score a subset first, or narrow with `AesLeakage(byte_indexes=[0])`
  and a smaller `top_k`.
- Both table inputs are optional — feed only `@tbl.reference` (or only
  `@tbl.current`) to inspect one PoI set on its own; the missing row shows `-`.
