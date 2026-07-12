# Plot Module Context

Use this for plotting or graph-inspection work involving `leakflow_plot`,
`leakflow_plugins_plot`, ImGui, ImPlot, `TracePlot`, pipeline graph rendering,
or plot runtime behavior.

## Files

Public headers:

- `include/leakflow/plot`
- `include/leakflow/plugins/plot`

Sources:

- `src/plot`
- `src/plugins/plot`

Tests:

- No GUI/window CTest for Phase 22.
- Keep existing CTest passing.
- Add non-visual tests only when they do not require an interactive display.

## Targets

- `leakflow_plot`: ImGui/ImPlot runtime and backend support; owns `PlotView` and
  the registry (`ScoreView` and `HeatmapView` live here — display data only,
  domain-free).
- `leakflow_plugins_plot`: plot plugin family with `TracePlot` and `HeatmapPlot`
  (`Plot/Heatmap`): renders any `[R,C]`/`[U,R,C]` matrix as an ImPlot `PlotHeatmap`
  (Viridis colormap + colorbar); props `normalize` (none/row/col), `unit`
  (-1 = sum over units), `title`/`row_label`/`col_label`. A sink; fills a
  `HeatmapView` via the shared-view factory pattern. Motivating use: a
  `ClusteringStats` reordered-confusion matrix, row-normalised into a per-class
  recall heatmap.
- `leakflow_plugins_crypto_plot`: crypto→plot bridge; `ScorePlot` fills a
  `ScoreView` and `ScoreTablePlot` fills a `TableView`, both from
  `AttackStatsPayload`. Depends on `leakflow_plugins_crypto` + `leakflow_plot`.

## Dependency Boundary

`leakflow_core` must not depend on plotting.

`leakflow_plot` may depend on:

- `leakflow_log`,
- `leakflow_core` for copied pipeline observation snapshots,
- ImGui,
- ImPlot,
- selected window/render backends such as GLFW/OpenGL3/Vulkan.

`leakflow_plugins_plot` may depend on:

- `leakflow_core`,
- `leakflow_base`,
- `leakflow_plot`.

Plotting must not pull AES, Kyber, YAML, dynamic plugin loading, or GUI
framework abstractions into core.

## Backend Vocabulary

```text
auto
opengl3
vulkan
```

Auto should prefer Vulkan only when compiled and available at runtime, then
fall back to OpenGL3.

Do not select Vulkan based only on NVIDIA visibility.

## TracePlot Contract

Input:

- `TorchTensorPayload`,
- rank 1 or rank 2,
- CPU,
- float32 initially.

Rank 1 means one trace.

Rank 2 means `[trace, sample]`, with axis 0 controlled by a slider.

`TracePlot` takes a plot-owned snapshot because the GUI loop outlives pipeline
processing.

`TracePlot` is a sink element. Its direct `process(...)` return value preserves
the input buffer for API-level inspection, but the element descriptor does not
declare a source pad.

`PlotRuntime` owns no plot data itself: it is a registry of `PlotView`s (see
**Plot Views** below). The trace snapshots and their UI state live in the
built-in `TraceView`; every other plot type is another registered view. It is
usable from both the CLI and application code.

`leakflow_plugins_plot` factory registration requires a shared `PlotRuntime`;
the `TracePlot` factory captures it and fills its built-in `trace_view()`, so the
generic `ElementFactoryRegistry` does not depend on plotting.

`leakflow::plot::run_until_closed(runtime)` owns a standalone GLFW/OpenGL3
window loop.

`leakflow::plot::draw_plot_runtime(runtime)` draws into the current ImGui frame
for applications that already own the UI loop.

## Plot Views (PlotView registry)

`PlotRuntime` is a registry of plot types, analogous to a `Pipeline` being a
registry of `Element`s. It holds `views_`, a list of `PlotView`
(`include/leakflow/plot/plot_view.hpp`), and owns no plot data itself:

```text
PlotView: draw(ctx)  clear()  empty()   // polymorphic; the runtime never type-switches
```

- Add a plot type = a new `PlotView` subclass (its own copied data, UI state,
  and ImGui/ImPlot rendering in its own `.cpp`) + the element that fills it. No
  edit to `plot_runtime.cpp` or `imgui_plot_loop.cpp`.
- `PlotRuntime::add_view(shared_ptr<PlotView>)` registers a view. The built-in
  `TraceView` is created and registered at construction and reachable via
  `trace_view()`. `draw_plot_runtime` builds a `PlotDrawContext` (streaming flag +
  control runtime) and loops `views()` calling `draw(ctx)`. `has_sessions()` is
  true for any non-empty view. `clear()` (Stop/Start recycle) cascades to each
  `view->clear()`.
- A view owns **its own mutex**; it is filled from the worker thread and drawn
  from the UI thread, and both serialize on that lock. A view therefore has no
  lifetime dependency on `PlotRuntime` (lock order is always runtime → view).
- Shared, domain-free drawing primitives (colour, marker shapes, number label,
  pixel distance) live in `src/plot/plot_render_util.{hpp,cpp}` for reuse across
  views. They stay generic — colours/markers/pixels, not traces/scores.

`TraceView` (`include/leakflow/plot/trace_view.hpp`, `src/plot/trace_view.cpp`)
is the trace plot as a view: it owns the trace snapshots, all trace UI state, the
two-way trace-index / x-axis listeners, its own mutex, and the trace rendering.
`TracePlot` elements share the runtime's built-in `TraceView`; the graph installs
the slider/x-axis listeners on it (`plot_runtime.trace_view()->set_*_listener`).

`ScoreView` (`include/leakflow/plot/score_view.hpp`, `src/plot/score_view.cpp`)
is the first non-trace view: stacked metric panels, one line series per attack
unit, a marker shape per point (square = success, x = failure, circle = truth
unknown), per-unit "latest wrong key" vertical lines, draggable panel heights.
Its display data is generic (panels/series/points), so `leakflow_plot` stays
domain-free of CPA.

`TableView` (`include/leakflow/plot/table_view.hpp`, `src/plot/table_view.cpp`)
is a generic ImGui-table view (cells + tint + hover + a bounded history
scrubber). `ScoreTablePlot` fills it from `AttackStatsPayload` as a scoreboard:
units as columns, ranked guesses as rows, correct-key cell tinted. It works the
same live (a frame per buffer, kept to `max_history`) and offline (one frame =
final result), so it needs no live/offline switch.

`ScorePlot`, the element that fills a `ScoreView`, lives in the separate
`leakflow_plugins_crypto_plot` target (depends on `leakflow_plugins_crypto` +
`leakflow_plot`) because it reads the crypto `AttackStatsPayload` directly. Its
factory (`crypto_plot::register_element_factories(registry, runtime)`) creates
one `ScoreView` and one `TableView`, registers both with the shared `PlotRuntime`,
and hands the `ScoreView` to every `ScorePlot` and the `TableView` to every
`ScoreTablePlot` — so all score elements/units stack together in one `group`
window (`ScorePlot` always stacks; it never overlays). Design:
`docs/design/plotting.md` (Plot View Architecture) and `docs/design/cpa_attack.md`.

## Pipeline Graph Runtime

`PipelineGraphRuntime` lives in `leakflow_plot` and consumes core
`PipelineObserver` events.

Node colors in `--graph` are derived from the element `klass` hierarchy, not just
the first profile token: `Source/*` is blue, `PassThrough/Flow/*` is slate,
`Convert/*` is teal, `Analyze/SCA/*` is amber/yellow with role-specific shades,
and `Sink/Plot/*` is violet. When adding a new first-class `klass` family or
role, update `docs/design/metadata_klass_taxonomy.md` and the local
`klass_colors(...)` palette in `src/plot/pipeline_graph.cpp`.

The graph runtime stores display copies only:

- topology snapshots,
- current copied property values,
- declared pads and link caps,
- latest routed-buffer caps/metadata/payload type by link,
- per-link observed buffer counts,
- recent event status and last error text.

It does not store payload pointers or raw trace values.

`draw_pipeline_graph(runtime)` draws the live ImGui graph into the current
frame. Applications that own an ImGui loop can attach a graph runtime as the
pipeline observer, run the pipeline however they choose, drain/draw the graph,
and draw `PlotRuntime` in the same frame.

Graph interaction:

- **Hover** a node or link for a transient tooltip (a quick peek that follows the
  cursor; ImGui tooltips are non-interactive by design — you cannot mouse into
  one).
- **Click** a node body (not its gear) or a link to **pin** a collapsible,
  mouse-interactable info window (spawned at the right edge of the viewport).
  The pinned node/link is highlighted amber as click feedback. Clicking again, or
  the window close button, unpins it. Nodes take click priority over links. Pin
  state lives in `PipelineGraphRuntime` (`toggle_pinned_element` /
  `is_element_pinned` / `pinned_elements` and the `*_link` equivalents) and is
  cleared by `clear()`.
- The tooltip and the pinned panel render the **same** info bodies and **share
  collapse state** (`PipelineGraphRuntime::section_open` / `set_section_open`,
  keyed by section label). The panel's `CollapsingHeader`s are interactive; the
  tooltip mirrors them with a static `[+]`/`[-]` disclosure. So collapsing a
  section in a panel also shrinks the tooltip — which is how a too-tall tooltip
  (whose scrollbar is useless, since tooltips can't be scrolled) is tamed.
  Heavier sections (pads, metadata, payload, declared caps) default collapsed.
- A **"Vector Clock (provenance counts)"** collapsing panel lists, per allocated
  slot index, the owning element and the current production count
  (`PipelineGraphRuntime::max_provenance()`, the running component-wise max of
  every observed buffer clock). Slot 0 is reserved and skipped.
- A **"Buffer Clocks (per link)"** collapsing panel is the per-buffer debug view:
  for the latest buffer on every link it shows the clock **decoded** into
  `element = count` (or `element.pad = count`) for each non-zero slot. Slot→label
  decoding (`build_slot_labels` / `decode_clock`) uses `provenance_base` /
  `provenance_slots` and, when an element claims one slot per output pad, names
  the slot `element.pad`. The pinned link panel/tooltip "clock" line uses the same
  decoder.

`PipelineBufferObservation` carries the full buffer vector clock
(`provenance`) and a derived scalar `generation`; the topology snapshot carries
each element's `provenance_base`/`provenance_slots`. These are copied,
SCA-safe diagnostics (no payload pointers, no raw values).

`PipelineControlRuntime` binds weakly to live `Element` instances and renders
property-based controls from `PropertySpec` / `PropertyValue`. It is separate
from `PipelineGraphRuntime`: graph observation remains copied and SCA-safe,
while controls are the explicit live mutation path.

Applications can either:

- manually create/connect a `Pipeline`, bind a `PipelineControlRuntime`,
  and call `draw_pipeline_graph(...)` / `draw_pipeline_controls(...)` inside
  their own ImGui frame, or
- use `leakflow::cli::build_builtin_pipeline_from_expression(...)` and pass the
  resulting pipeline/runtime to `run_pipeline_graph_until_closed(...)`.

`run_pipeline_graph_until_closed(...)` is the reusable app-facing equivalent of
`leakflow run --graph`: it installs a graph observer, binds controls, runs the
synchronous pipeline in a worker thread, and owns the shared graph/plot/control
window loop until it closes.

In the graph, each bound element is shown with a gear-shaped control button.
`draw_pipeline_controls(...)` can also render controls without drawing the graph,
so applications may expose controls in a separate panel.

Control widgets are generated from property type and constraints:

- bool -> checkbox,
- integer/double -> numeric input with existing validation,
- string enum -> combo,
- string/list -> text input committed through existing validation,
- intervals -> paired numeric inputs.

Successful control edits are recorded as `PipelineControlChange` records. This
is the current hook for future partial invalidation/rerun work. Property specs
can declare a `PropertyEffect`:

- `ui-control` — non-dataflow control/presentation; the element self-applies it to
  its own display state and the session does nothing (no rerun, valid in any
  state). All `TracePlot` properties are `ui-control` and are applied to the live
  snapshot in `property_changed` (so e.g. toggling `center0` shows immediately
  while Running/Paused/Idle, not only after a Stop/Start).
- `sink-display` — re-derive the sink display from its cached input buffer
  (reprocess from cache; needs the input cache),
- `metadata-output`,
- `payload-output`,
- `caps-output`,
- `lifecycle`.

They also declare an invalidation scope such as `none`, `element-ui`,
`downstream`, or `full-pipeline`, plus optional affected output pad names.
For example, `AesLeakage.channels` is `payload-output` with `downstream`
invalidation on the `leakage` output pad.

Since Phase 25 the control plane is `PipelineSession` in `leakflow_core` (design:
`docs/design/pipeline_controller.md`):

- UI controls submit `SetProperty` commands; `PipelineControlRuntime` is a UI
  client of the session and no longer mutates elements directly.
- Commands are validated through the existing property specs and applied at safe
  points (between units of work) by the worker thread.
- Accepted/rejected/applied events are copied into the observation stream.
- The session caches latest input buffers per input pad and output buffers per
  output pad and reruns only the downstream path affected by an edited element
  (`Pipeline::rerun_from`), escalating to a full restart when a
  `can_replay() == false` element is in the replay-set.
- The buffer vector clock (`Buffer::provenance()`, Phase 27) distinguishes
  generations; the graph runtime resets per-link counts when a buffer's derived
  `generation` (`provenance_generation`) changes. The session `generation`
  counter is monotonic for the session lifetime. (`Buffer::epoch()` was removed.)

`run_pipeline_graph_until_closed(PipelineSession&, PlotRuntime&, ...)` installs a
graph observer, binds controls, and owns the shared graph/plot/control window
loop with a persistent worker loop (the unified `run()` pump loop — live is
auto-detected, not a separate drive). Graph control buttons are the player
controls: **Start / Stop / Pause / Resume** plus an **Auto-apply edits** toggle.
Pass `--auto-start` to begin Running on open; a finite live stream reaching EOS
lands in **Idle** (held, inspectable). See `docs/design/dataflow_sync_model.md`
§13.

The progress-bar activity sweep animates only while the session is **Running**;
the sweep is suppressed while **Paused**, **Idle**, or **Stopped**.
Long synchronous elements can park at cooperative internal checkpoints while
Paused. Cancellation is a distinct terminal progress outcome: the graph shows a
static red 100% bar briefly, and canceled output is not routed to plot sinks.

Example of a downstream-only rerun:

```text
AesLeakage.channels: [HW(m)] -> [HW(m), HW(y)]
effect: payload-output, scope: downstream, output pad: leakage
```

reruns from `AesLeakage.leakage` downstream through `PearsonCorrelator`, `PoiSelect`,
annotation conversion, and plot sinks using cached/latest inputs, without
rerunning upstream trace loading.

Live/queue runtime (implemented — see `docs/design/dataflow_sync_model.md` §12–13):

- The unified `run()` pump loop auto-detects a live source; live + `Queue` graphs
  run threaded segments. No separate `StreamingDrive`.
- Live control changes forward-apply at a between-buffer safe point (no offline
  rerun); one-run-driven changes still recompute from cache.
- A threaded/live `Queue` is `BufferQueue` (Block / DropOldest / DropNewest);
  `QueueEpochPolicy` drain/flush is an optional generation-boundary policy.
- A cooperative `std::stop_token` cancel (CLI SIGINT + window close) lets a
  blocking source unwind to a safe point; the `Stopped/Running/Paused/Idle` player
  state machine and pause/resume are wired. Only `preroll` is reserved.

`leakflow run --graph EXPRESSION` uses this shape:

```text
main thread: GLFW/ImGui loop, pipeline graph, TracePlot windows
worker thread: existing synchronous Pipeline::run()
event bridge: copied PipelineEvent records
```

The graph UI and `TracePlot` share one window loop. `PlotRuntime` protects
snapshot registration/drawing with a narrow mutex so `TracePlot` can publish
snapshots from the worker while the main thread renders.

## TracePlot Grouping

Use `group` instead of dynamic sink pads in Phase 22.

One `group` is one ImGui plot window inside the plot runtime.

Multiple `TracePlot` elements with the same `group` enter that window together.

Snapshots are displayed in group order. The default order is the order snapshots
enter the group. `TracePlot(order=...)` provides an optional explicit ordering
key, and the plot controls window may reorder snapshots interactively.

The first `TracePlot` in a group starts the first plot panel; its `layout`
property has no effect. Later snapshots use their own `layout` value:

- `overlay` adds the snapshot to the previous panel,
- `stacked` starts a new panel.

This allows a group to contain several overlaid panels and several stacked
panels in one window. Each TracePlot panel has the same draggable horizontal
splitter style as `ScoreView`: drag the thin grip below a plot to make that
stacked graph taller or shorter.

Rank-2 trace selection uses vertical sliders on the right side of each plot
panel. The slider context label defaults to `trace` and displays one-based trace
numbers (`trace: 1`, `range: 1 - N`). `TracePlot(trace_context_label=...)` can
override the label; `unit` uses zero-based implicit values unless
`attack.unit.indexes` metadata provides actual unit values. Buffers with
`tensor.axes=attack_unit,sample` auto-derive `unit` when the property is empty.

`TracePlot(center0=true)` is the default. It sets each plot panel's y-axis to a
symmetric range around zero using `max(abs(min_leakage), abs(max_leakage))` from
the displayed trace data. The computed fit is applied when the data-fit range or
center mode changes, but user zoom/pan is preserved afterward. Set
`center0=false` to fit the displayed trace data without forcing symmetry around
zero.

Each group window has a controls button that opens a floating controls window
for per-TracePlot color, alpha, and display order.

`TracePlot(color=...)` initializes the line color for reproducible commands.
Accepted values are:

- named colors such as `red`, `blue`, `cyan`, `magenta`, `yellow`, `black`,
  `white`, `gray`/`grey`, `orange`, `purple`, `pink`, `teal`, and `lime`,
- hex RGB/RGBA such as `#ff3366` or `#ff336680`,
- functional RGB/RGBA such as `"rgb(255,51,102)"` or
  `"rgba(255,51,102,0.5)"`.

RGBA alpha initializes the same display alpha as the `alpha` property. When a
non-default `alpha` property is also provided, `alpha` wins.

Use `lock_trace_index=true` to initialize the group trace-slider lock. All
rank-2 sliders remain visible; moving one locked slider shifts the other
rank-2 traces in that group by the same delta while preserving their offsets.
The group stops when any linked slider reaches its top or bottom bound. After
launch the lock can be toggled from the floating `Controls` window, or by
right-clicking any trace slider; the right-click `Lock trace sliders` toggle
appears only when the group has more than one trace slider and works the same in
stacked and overlay layouts.

## Time Axis

Canonical metadata key:

```text
sample_rate_hz
```

Sampling-rate resolution:

```text
TracePlot(sample_rate_hz=...) > buffer metadata sample_rate_hz > sample index fallback
```

Display property:

```text
x_axis=sample|time_us
```

## Related Base Plugin

`TorchConvert` belongs in `leakflow_plugins_base`, not the plot plugin.

It explicitly converts:

```text
TorchTensorPayload -> TorchTensorPayload
```

Use it to prepare CPU float32 input for `TracePlot`.

`TorchConvert` is explicit and does not reintroduce generic `Convert`.

## Manual Validation

Phase 22 plot behavior is manual-only for GUI/window rendering.

Example:

```bash
./build/leakflow run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! TracePlot(title="AES traces")'
./build/leakflow run --graph 'FakeSrc ! Tee@t; @t.src_0 ! Summary; @t.src_1 ! FakeSink'
```

Optional tutorial app:

```bash
CXX=clang++ cmake -S . -B build-examples -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch" -DLEAKFLOW_BUILD_EXAMPLES=ON
cmake --build build-examples -j
./build-examples/leakflow_trace_plot_tutorial
```
