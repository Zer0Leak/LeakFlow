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

The AES sync demos below expect one HDF5 file per key:

```text
traces/aes/sync/aes_sync_poi/key_01.h5
```

For a quick smoke run without the downloaded dataset, use the checked-in
fixture:

```text
tests/fixtures/aes/sync/key_01.h5
```

Each file contains `/traces`, `/plaintexts`, `/keys`, and derived
`/ciphertexts`, plus acquisition metadata. Jitter captures additionally contain
`/countermeasures/jitter/parameters/loop_iterations`; synchronized captures do
not have a `/countermeasures` group.

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

This one-shot CPA run loads the linked tensors from one `Hdf5FileSrc`, so no
`Queue` or `Sync` is needed. It is a good final-result view: the trace plot shows the raw
traces with best-sample markers, `ScorePlot` shows the final score/confidence
state, and `ScoreTablePlot` shows the recovered-key scoreboard.

```bash
leakflow --log-level warning run --graph \
  'Hdf5FileSrc@data \
      (path=traces/aes/sync/aes_sync_poi/key_01.h5); \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=cpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[HW(y)],units=[],guess_values=[]); \
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
   @data.traces ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack.features; \
   @data.plaintexts ! @hyp.plaintexts; \
   @hyp ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @data.keys ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @trace_plot.annotations; \
      @stats_tee.src_1 ! @score_plot; \
      @stats_tee.src_2 ! @scoreboard'
```

Notes:

- `units=[]` and `guess_values=[]` mean all 16 AES bytes and the full
  `0..255` guess domain.
- `ScorePlot(show_second_score=true)` needs `AttackStats(top_k>=2)`; this command
  uses `top_k=8`, which also controls the scoreboard row budget.
- `TracePlot(x_axis=time_us)` reads `capture.sample_rate_hz` from the trace
  metadata. If you remove that metadata, it falls back to sample indexes.

## AES Sync CPA Live - Trace, Score, Scoreboard

This live CPA run replays aligned batches from one `FakeLiveHdf5Src` and crosses
a real `Queue` on each active stream. All pads from a batch share source
provenance, so the default barrier pairs trace batch `N` with plaintext batch
`N`; no explicit `Sync` is needed. `CpaAttack(correlation_mode=auto)` resolves
to incremental under live inputs, so the score plot and scoreboard update as
observations arrive.

```bash
leakflow --log-level warning run --graph \
  'FakeLiveHdf5Src@data \
      (path=traces/aes/sync/aes_sync_poi/key_01.h5,batch_size=1,trace_rate=0); \
   Queue@traces_queue(max_size=8,drop_oldest=false); \
   Queue@plain_queue(max_size=8,drop_oldest=false); \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=live-trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=cpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[HW(y)],units=[],guess_values=[]); \
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
   @data.traces ! @traces_queue ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack.features; \
   @data.plaintexts ! @plain_queue ! @hyp.plaintexts; \
   @hyp ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @data.keys ! @stats.truth; \
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
- Drop `@data.keys ! @stats.truth` to run without known-key truth. The scoreboard
  still ranks guesses, while truth/rank/success fields disappear.

## AES Sync DPA Offline - Trace, Difference Trace, Score, Scoreboard

This one-shot DPA run uses binary AES S-box bit hypotheses and
`DpaAttack.best_difference` to show the generated difference-of-means traces.
The raw trace plot still receives `AttackStats` annotations, while the second
`TracePlot` shows the current best difference trace for each attack unit.

```bash
leakflow --log-level warning run --graph \
  'Hdf5FileSrc@data \
      (path=traces/aes/sync/aes_sync_poi/key_01.h5); \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=dpa-trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=dpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[y(0),y(3),y(7)],units=[],guess_values=[]); \
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
   @data.traces ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack.features; \
   @data.plaintexts ! @hyp.plaintexts; \
   @hyp ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @attack.best_difference ! @diff_plot.sink; \
   @data.keys ! @stats.truth; \
   @stats ! @stats_tee; \
      @stats_tee.src_0 ! @ann ! @trace_plot.annotations; \
      @stats_tee.src_1 ! @score_plot; \
      @stats_tee.src_2 ! @scoreboard'
```

Notes:

- `y(0)`, `y(3)`, and `y(7)` are binary S-box output-bit hypotheses. Narrow to
  `channels=[y(0)],units=[0]` for a faster interactive demo.
- `DpaAttack.best_difference` emits a CPU float32 `[attack_unit,sample]` tensor,
  so `TracePlot(trace_context_label=unit)` gives the slider unit semantics.
- The raw trace and difference trace share `group=dpa`; `layout=stacked` puts the
  generated difference traces under the raw trace panel.

## AES Sync DPA Live - Trace, Difference Trace, Score, Scoreboard

This live DPA version mirrors the live CPA shape. Aligned HDF5 trace/plaintext
batches stream through queues and match by their shared source provenance;
`DpaAttack(accumulation_mode=auto)` resolves to incremental. The extra
`TracePlot@diff_plot` replaces its snapshot each update so it always shows the
latest best-difference state.

```bash
leakflow --log-level warning run --graph \
  'FakeLiveHdf5Src@data \
      (path=traces/aes/sync/aes_sync_poi/key_01.h5,batch_size=1,trace_rate=0); \
   Queue@traces_queue(max_size=8,drop_oldest=false); \
   Queue@plain_queue(max_size=8,drop_oldest=false); \
   Tee@trace_tee; \
      @trace_tee.src_%u{routing.branch.family=live-dpa-trace-fanout}; \
      @trace_tee.src_0{routing.branch=raw-trace-plot}; \
      @trace_tee.src_1{routing.branch=dpa-features}; \
   AesLeakageHypothesis@hyp \
      (channels=[y(0),y(3),y(7)],units=[],guess_values=[]); \
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
   @data.traces ! @traces_queue ! @trace_tee; \
      @trace_tee.src_0 ! @trace_plot.sink; \
      @trace_tee.src_1 ! @attack.features; \
   @data.plaintexts ! @plain_queue ! @hyp.plaintexts; \
   @hyp ! @attack.hypotheses; \
   @attack.scores ! @stats.scores; \
   @attack.best_difference ! @diff_plot.sink; \
   @data.keys ! @stats.truth; \
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

`BufferFileSink` persists a whole buffer (caps + metadata + payload) to a single
HDF5 file (the `leakflow.buffer` schema); `BufferFileSrc` reloads it. Because the
expensive step is the correlation and `PoiSelect` is cheap, you can compute the
correlation once, save it, then reload and try any `top_k`/`rank_by` **offline, with
no re-streaming**.

Compute and save the correlation:

```bash
leakflow --log-level warning run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); \
   Tee@tee; AesLeakage@lk(channels=[HW(m),HW(y)],units=[]); \
   PearsonCorrelator@corr; BufferFileSink@save(path=out/aes_corr.h5); \
   @data.traces ! @tee; @tee.src_0 ! @corr.features; @tee.src_1 ! @lk.traces; \
   @data.plaintexts ! @lk.plaintexts; @data.keys ! @lk.keys; @lk ! @corr.targets; @corr ! @save'
```

`out/aes_corr.h5` is a self-describing HDF5 file: `h5dump -H` shows the envelope
attributes (`caps.type=leakflow/correlation`, the leakage metadata,
`payload.correlation.observation_count=50`, ...) and the correlation tensor as native
datasets under `/payload`.

### Streaming a whole profiling set (a folder of keys)

The save command above reads a single `.h5` file. Real PoI profiling uses **many
traces across several keys** -- e.g. `traces/aes/sync/aes_sync_poi` holds 50
HDF5 files named `key_NN.h5`, 100 000 traces in all. Loading all of that into one buffer is
wasteful, so the replication app streams the retained original `key_*` folders:

```bash
leakflow_rezaeezade_poi_finder --save-correlation out/aes_corr.h5 --graph \
  traces/aes/sync/aes_sync_poi/key_50.h5
```

It walks each retained `key_*` folder in turn, feeding that folder's
traces/plaintexts/keys through the same `AesLeakage -> PearsonCorrelator` chain
and **accumulating** the
Pearson correlation over every key and trace -- an `AppSrc` pulls one folder at a time,
so only ~one folder is ever resident in memory. `--graph` opens the live pipeline graph
(same controls as `leakflow run --graph`) so you watch the aggregate correlation and its
PoIs build up as folders stream in; drop `--graph` for a headless run, add `--auto-start`
to begin on open. `--save-correlation PATH` writes the **same** `out/aes_corr.h5` the
reload-and-plot command below consumes -- so a folder-scale, multi-key correlation drops
straight into the offline PoI-selection workflow.

The original folders are retained temporarily so the later HDF5/Zarr benchmark
can convert both formats from identical inputs. Current single-key CLI examples
use `key_NN.h5`; the replication app remains the multi-key streaming path until
it receives its own format-neutral dataset-reader migration.

Reload the correlation and plot the PoIs. The expensive correlation is **not**
recomputed (`@corr_src ! @poi` re-selects from the cached correlation); the traces are
reloaded only as the cheap backdrop the PoI markers are drawn on:

```bash
leakflow run --graph \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); \
   BufferFileSrc@corr_src(path=out/aes_corr.h5); \
   PoiSelect@poi(top_k=[3],rank_by=[abs]); \
   Tee@poi_tee; \
   CorrelationPoiToPlotAnnotations@ann(precision=3); \
   TracePlot@plot(title="Cached Pearson PoIs",group=poi,label=traces,x_axis=time_us,annotation_update_mode=replace); \
   Summary@summary(level=2); \
   FakeSink@sink; \
   @data.traces ! @plot.sink; \
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
`out/aes_corr.h5` above) and then reused against a fresh capture. This pipeline
asks whether that reuse is justified: it re-runs the Pearson correlation **only at
the saved PoI columns** on new traces (`PoiCorrelation`), then lays the attack
scores next to the profiling scores in a comparison table (`PoiTablePlot`) with a
line under it per phase. Strong transfer keeps the `attack` row tracking the
`profiling` row; desync, jitter, or a different device shows up as the attack line
collapsing.

```bash
leakflow run --graph \
  'BufferFileSrc@corr_src(path=out/aes_corr.h5); \
   PoiSelect@poi(top_k=[20],rank_by=[abs]); \
   Tee@poi_tee; \
   Hdf5FileSrc@attack(path=traces/aes/sync/aes_sync_poi/key_05.h5); \
   Tee@trace_tee; \
   AesLeakage@leakage(channels=[HW(m),HW(y)],units=[]); \
   PoiCorrelation@poicorr; \
   PoiTablePlot@tbl \
      (title="Profiling vs attack PoIs",reference_label=profiling,current_label=attack,precision=3); \
   @corr_src ! @poi ! @poi_tee; \
      @poi_tee.src_0 ! @tbl.reference; \
      @poi_tee.src_1 ! @poicorr.poi; \
   @attack.traces ! @trace_tee; \
      @trace_tee.src_0 ! @poicorr.traces; \
      @trace_tee.src_1 ! @leakage.traces; \
   @attack.plaintexts ! @leakage.plaintexts; \
   @attack.keys ! @leakage.keys; \
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
- To actually measure *transfer*, point `Hdf5FileSrc@attack` at a capture
  **other** than the one the correlation was profiled on — a different
  `key_NN.h5` under `aes_sync_poi`, a resynchronized set, or another device.
  Pointing it at the profiling key (as above) is the sanity baseline where both
  rows should agree.
- `PoiCorrelation` recomputes correlation over every unit at each PoI column, so it
  scales with trace count. `key_01` (2 000 traces) is interactive; for a larger
  attack set, re-score a subset first, or narrow with `AesLeakage(units=[0])`
  and a smaller `top_k`.
- Both table inputs are optional — feed only `@tbl.reference` (or only
  `@tbl.current`) to inspect one PoI set on its own; the missing row shows `-`.

### Compare PoI Transfer And GMM Clusters

This extended version selects the profiling PoIs, extracts those columns from
the attack traces, clusters them with a 49-component GMM, and evaluates the
clusters directly against the vector truth `(HW(m), HW(y))`. The structured
metrics, effective evaluator options, captured GMM parameters, and bounded
stored-contingency heatmap are shown in one tabbed comparison window. The same
selected PoIs are also re-correlated on the attack traces and shown beside the profiling scores. One
`Hdf5FileSrc` supplies aligned traces, plaintexts, and the fixed key.

See [Interpreting Clustering Evaluation Metrics](CLUSTERING_EVALUATION.md) for
the meaning, direction, and limitations of every displayed metric and heatmap
value. This graph opts into `semantic_partition_quality`, the preferred
separation/pair-recall composite for comparing cluster counts. The older
`combined_quality` is deprecated because it can assign a misleadingly high
value to a one-cluster collapse.

```bash
A=traces/aes/sync/aes_sync_attack/key_01.h5  # or an HDF5 subset such as A=out/key05_sub.h5 for a fast interactive run
leakflow --log-level warning run --graph \
  "BufferFileSrc@corr_src(path=out/aes_corr.h5); \
   PoiSelect@poi(top_k=[50],rank_by=[abs]); \
   Tee@poi_tee; \
   CorrelationPoiToIndexes@poi2idx(units=[0]); \
   Hdf5FileSrc@data(path=$A); \
   Tee@trace_tee; \
   FeatureSelect@featsel; \
   AesLeakage@leakage(channels=[HW(m),HW(y)],units=[0]); \
   Tee@leak_tee; \
   PoiCorrelation@poicorr; \
   PoiTablePlot@tbl(title=\"Profiling vs attack PoIs\",reference_label=profiling,current_label=attack,precision=3); \
   GaussianMixture@gmm(n_components=49,covariance_type=diagonal,n_init=1,max_iter=100,seed=0); \
   ClusteringEvaluate@eval(semantic=power,semantic_ranges=[8,8],dimension_names=[hm,hy],detail=full,alignment=both,semantic_partition_quality=true); \
   ClusteringMetricsTablePlot@metrics(title=\"GMM clustering evaluation\",update_mode=accumulate); \
   @corr_src ! @poi ! @poi_tee; \
   @poi_tee.src_0 ! @tbl.reference; \
   @poi_tee.src_1 ! @poi2idx ! @featsel.indexes; \
   @poi_tee.src_2 ! @poicorr.poi; \
   @data.traces ! @trace_tee; \
   @trace_tee.src_0 ! @featsel.features; \
   @trace_tee.src_1 ! @leakage.traces; \
   @trace_tee.src_2 ! @poicorr.traces; \
   @data.plaintexts ! @leakage.plaintexts; \
   @data.keys ! @leakage.keys; \
   @leakage ! @leak_tee; \
   @leak_tee.src_0 ! @eval.truth; \
   @leak_tee.src_1 ! @poicorr.targets; \
   @featsel ! @gmm ! @eval.labels; \
   @poicorr ! @tbl.current; \
   @eval.evaluation{payload.parameter.dataset=key_01} ! @metrics.sink"
```

The GMM `labels` (via `FeatureSelect` → `GaussianMixture`) and the vector truth
(via `AesLeakage`) must describe the **same units**. `ClusteringEvaluate` aligns
them by unit id, so `CorrelationPoiToIndexes(units=[...])` and
`AesLeakage(units=[...])` need to name the same bytes: disjoint units are an
error (*"labels and truth share no units"*), and a partial overlap warns and scores
only the shared units. Match them — both `[0]`, or both left at the full range — for
a meaningful comparison. The units each buffer carries show as `units=[…]` in a
`Summary` and in the `--graph` buffer inspector.

The metrics window has eight tabs:

- **Overview** is the place to start. It shows one row per run and unit, with
  explicit `Observations (N)` and `Features (S)` columns, group/cluster counts,
  headline metrics including semantic partition separation and the optional
  semantic partition quality, and the core GMM and experiment parameters
  needed to compare configurations. GMM supplies the feature value from its
  fitted axis; a producer without `payload.cluster.n_features` displays `N/A`.
- **Exact**, **Semantic**, **Fragmentation**, **Combined**, and **Alignment**
  contain the complete stored metric detail without duplicating a metric across
  tabs. `↑` after a metric means higher is better; `↓` means lower is better.
  These direction markers are part of the metric label; the arrow drawn on a
  column header shows the current table sort direction.
- **Parameters** shows effective evaluator settings, captured GMM context, and
  explicit experiment metadata once per run instead of repeating them on every
  metric row. It retains `labels.cluster.n_features` even though Overview also
  promotes that value to `Features (S)`.
- **Heatmap** row-normalizes the stored Full-detail sparse contingency. It uses
  the stored exact-overlap column permutation when available (raw column order
  otherwise), labels rows with the canonical truth vectors and dimension names,
  and labels columns with the actual predicted IDs. It remains rectangular and
  supports a different shape for every unit. Hover a cell to see the full true
  group, actual predicted cluster ID, observation count, percentage within the
  true-group row, percentage within the predicted-cluster column, and percentage
  of all observations. The tooltip stays numeric and does not include tutorial
  text. Global detail shows
  `requires ClusteringEvaluate(detail=full)`; when all unit pages in one run
  exceed 1,000,000 dense cells in total, they show a display-limit message
  instead of being allocated. This tab has
  fixed row normalization and exact alignment when available; selectable
  raw/semantic alignment and `none|row|col` normalization remain deferred to a
  standalone matrix plot.

`detail=full` and `alignment=both` are intentionally kept in this example: the
family tabs make the resulting per-dimension, per-cluster, per-group, and both
alignment records manageable. `update_mode=accumulate` keeps each
re-evaluation for comparison, which is useful when changing `GaussianMixture`
parameters such as `covariance_type` or `n_components` while Idle. Use
`update_mode=replace` to keep only the newest evaluation, or the default
`update_mode=auto` to accumulate for live-driven input and replace otherwise.
`active_update_mode` reports that resolved choice. Heatmap accumulation appends
one independent matrix frame per evaluation run; it never sums contingency
counts, while replace keeps only the latest frame. When the evaluation contains
multiple units, a horizontal **Unit** slider on Overview, the metric-family tabs,
and Heatmap selects one typed unit; the selection follows you across tabs, while
Parameters remains run-wide. To exercise it with this example,
change both `CorrelationPoiToIndexes(units=[0])` and
`AesLeakage(units=[0])` to the same multi-unit set, such as `[0,1]`. Click a
header within the current tab to sort ascending or descending. The table's
**Clear** button clears that tab;
**Clear all** removes every table in this comparison group from the tabbed
metrics window without clearing other plot views.
These display actions use stored results and do not refit the GMM or recompute
clustering evaluation.

Effective evaluator options come from the structured payload. The GMM
parameters explicitly captured from the labels buffer's `payload.cluster.*`
metadata include `method`, `n_components`, `covariance_type`, `n_features`, and
`converged`;
explicit experiment parameters can be stamped on the evaluation buffer as
`payload.parameter.*` metadata. Stamp an experiment parameter on
`@eval.evaluation` (or its outgoing link), as the graph does above; do not expect
`payload.parameter.*` on labels/truth to pass through the evaluator's
Analyze-boundary metadata forwarding. If `A` selects another dataset, update the
example `payload.parameter.dataset=key_01` stamp too.

`out/key05_sub.h5` is only an example subset path; point `A` at an existing
HDF5 file. Alternatively, keep the selected full key file and add
`row_count=1000` to `Hdf5FileSrc@data` for a quicker interactive run without
creating another file.
