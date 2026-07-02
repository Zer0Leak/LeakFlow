# LeakFlow Plotting Design

This document records the Phase 22 plotting design decisions.

## Goals

- Add interactive trace plotting for SCA inspection.
- Use Dear ImGui and ImPlot.
- Do not add a new executable.
- Keep plotting outside `leakflow_core`.
- Make the first plot element useful for AES trace validation later.
- Avoid dynamic pads and multi-input executor changes in the first plot phase.

## Targets

Phase 22 should use two layers:

```text
leakflow_plot
  ImGui/ImPlot integration
  backend selection
  PlotRuntime / plot session registry
  plot-owned data snapshots

leakflow_plugins_plot
  TracePlot element
```

`leakflow_plot` should not know AES or Kyber.

`leakflow_plugins_plot` may depend on `leakflow_base` because `TracePlot`
consumes `TorchTensorPayload`.

`leakflow_core` must not depend on `leakflow_plot`,
`leakflow_plugins_plot`, ImGui, ImPlot, GLFW, OpenGL, or Vulkan.

## Backend Policy

Supported backend vocabulary:

```text
auto
opengl3
vulkan
```

Configuration should eventually support:

```text
LEAKFLOW_PLOT_BACKEND=auto|opengl3|vulkan
```

and, if CLI options are added:

```text
--plot-backend auto|opengl3|vulkan
```

Precedence should match logging:

```text
CLI option > environment variable > default
```

Default:

```text
auto
```

Auto selection should prefer Vulkan only when:

- Vulkan support is compiled,
- the Vulkan loader/runtime is available,
- a usable Vulkan physical device exists,
- window/display integration is available.

Otherwise auto should fall back to OpenGL3.

Do not choose Vulkan based only on NVIDIA visibility. Vulkan can run on AMD and
Intel through Mesa, and NVIDIA can run OpenGL3 well. In containers, backend
availability depends on display sockets, device mounts, drivers, ICD files, and
permissions.

OpenGL3 is the mandatory practical fallback. Vulkan is the preferred
performance-oriented backend when available.

## No New Executable

Phase 22 should not add `leakflow-plot`.

Instead, the existing `leakflow run` command should:

1. run the pipeline synchronously,
2. let plot elements register plot sessions,
3. enter `PlotRuntime` after the pipeline finishes if any plot sessions exist.

Future applications can reuse the same `PlotRuntime` API.

## TracePlot

`TracePlot` is the first plotting element.

Input:

- `TorchTensorPayload`,
- rank 1 or rank 2,
- CPU,
- float32 initially.

Rank behavior:

- rank 1 means one trace/vector,
- rank 2 means `[trace, sample]`,
- rank 2 uses axis 0 as trace index and axis 1 as sample index.

`TracePlot` reads its input and registers a plot-owned snapshot with its
`TraceView` (the built-in plot view; see **Plot View Architecture** below).

Because the GUI loop outlives pipeline processing, `TracePlot` must not keep
mutable access to upstream payload data. The first implementation should copy
input into plot-owned CPU float32 storage. Later phases may add lazy loading,
decimation caches, or memory-mapped storage for large datasets.

## TracePlot Properties

Initial properties:

```text
group="default"
label=""
title="Trace Plot"
x_label="Sample"
y_label="Amplitude"
layout=overlay|stacked
color=auto            # rgba; the colour carries its own alpha; auto picks a palette colour
line_width=1.0
trace_index=0
order=-1              # -1 = automatic
lock_trace_index=false
x_axis=sample|time_us
sample_rate_hz=0
```

`sample_rate_hz=0` means no property override.

## Plot Groups

Use plot groups instead of dynamic sink pads in Phase 22.

Multiple `TracePlot` elements with the same `group` are displayed together.

Example:

```bash
leakflow run '
  NumpySrc(path=a.npy) ! NumpyToTorch ! TorchConvert(dtype=float32,device=cpu) ! TracePlot(group=compare,label=A);
  NumpySrc(path=b.npy) ! NumpyToTorch ! TorchConvert(dtype=float32,device=cpu) ! TracePlot(group=compare,label=B)
'
```

Group display modes:

```text
overlay
  draw all selected traces on the same axes

stacked
  draw traces one above another, preferably with a shared x-axis
```

Overlay mode should support alpha/transparency so dense trace comparisons are
readable.

Each rank-2 series has its own trace slider by default.

When `lock_trace_index=true`, a group uses one shared trace index. If a series
has fewer traces than the shared index, clamp to the last valid trace.

## Plot View Architecture (registry pattern)

`TracePlot` is not the only kind of plot. To add a new plot type (scores, ranks,
a table, a heatmap) without growing `plot_runtime.cpp` / `imgui_plot_loop.cpp`
into a type switch, plotting uses a small registry: `PlotRuntime` is to a
`PlotView` what a `Pipeline` is to an `Element`. The runtime holds a list of
polymorphic views and, each frame, asks each one to draw itself — it never asks
"is this a score plot or a trace plot".

```text
PlotView   (include/leakflow/plot/plot_view.hpp)
  virtual void draw(const PlotDrawContext&)  // render my windows into the ImGui frame
  virtual void clear()                       // drop my data (Stop/Start recycle)
  virtual bool empty() const                 // do I have anything to show?
```

- A view owns **its own copied display data, its own UI state, and its own
  ImGui/ImPlot rendering** (its own `.cpp`). Adding a plot type is a new
  `PlotView` subclass plus the element that fills it — not an edit to the shared
  runtime.
- A view owns **its own mutex**. The pipeline worker fills it (its element pushes
  copied snapshot data) and the UI thread draws it; both serialize on that lock.
  Because the lock and the data live inside the view, a view has no lifetime
  dependency on `PlotRuntime` — the runtime just keeps a `shared_ptr` to it.
- `PlotRuntime` owns **no plot data itself**: just `views_`, a
  `vector<shared_ptr<PlotView>>`, and a typed handle to the built-in trace view.
  `add_view(view)` registers one; `draw_plot_runtime` builds a `PlotDrawContext`
  (streaming flag + control runtime) and loops `views()` calling `draw(ctx)`;
  `has_sessions()` is true if any view is non-empty; `clear()` cascades to every
  view's `clear()`. Lock order is always runtime → view (the worker only ever
  takes a single view lock), so there is no inversion.
- `PlotDrawContext` carries the run-level facts a view may need but does not own:
  `streaming` (is the pipeline Running — accumulate sliders follow the newest
  trace only while streaming) and `control_runtime` (so a view's gear button can
  open its element's control panel). Views ignore fields they do not use.

Shared, domain-free drawing primitives (colour conversion, marker shapes,
number-label, pixel distance) live in `src/plot/plot_render_util.{hpp,cpp}` so
each view's rendering TU reuses them instead of copying. They know colours,
marker shapes, and pixels — not traces or scores.

### TraceView — the built-in view

`TraceView` (`include/leakflow/plot/trace_view.hpp`, `src/plot/trace_view.cpp`)
is the trace plot as a `PlotView`. It owns the trace snapshots, all trace UI
state (per-snapshot slider index / follow-latest, per-group lock + controls-open,
per-snapshot display state, per-panel axis view), the two-way trace-index /
x-axis listeners, its own mutex, and the full trace rendering. `PlotRuntime`
creates one at construction and registers it (reachable via `trace_view()`), so
it draws and clears through the same loop as every other view — the trace path
is the *built-in* view, not a special case in the runtime or the draw loop.

`TracePlot` is the *element* that fills a `TraceView`; several `TracePlot`s
sharing one `TraceView` stack/overlay in one group window. Its factory hands each
`TracePlot` the runtime's built-in `trace_view()` (a `TracePlot(PlotRuntime)`
convenience constructor does this for app/tutorial code that only has a runtime).

### ScoreView / ScorePlot — the second view

`ScoreView` (`include/leakflow/plot/score_view.hpp`, `src/plot/score_view.cpp`)
is the first non-trace view and the reference for the pattern. It renders a
stacked set of metric panels (one per selected metric, e.g. `score` and
`relative_margin`), each with one line series per attack unit, a point per
streamed buffer at `x = observation count`, a marker shape per point (square =
success, x = failure, circle = truth unknown), per-unit "latest wrong key"
vertical lines behind the data, and draggable panel heights. Its display data is
**generic** (panels/series/points, no CPA types), so `leakflow_plot` stays
domain-free.

`ScorePlot` is the *element* that fills a `ScoreView`. It lives in a separate
plugin, `leakflow_plugins_crypto_plot` (depends on both `leakflow_plugins_crypto`
and `leakflow_plot`), because it reads the crypto `AttackStatsPayload` directly
and translates it into generic score points. This keeps the CPA knowledge in the
crypto-plot bridge and out of `leakflow_plot`.

All `ScorePlot` elements in one run share **one** `ScoreView`: the factory
(`crypto_plot::register_element_factories`) creates the view, registers it with
the shared `PlotRuntime` via `add_view`, and hands it to each `ScorePlot`. One
shared view is what lets several units/elements stack together in one `group`
window (the same `group` concept as `TracePlot`, but `ScorePlot` always stacks,
never overlays).

### TableView / ScoreTablePlot — the scoreboard view

`TableView` (`include/leakflow/plot/table_view.hpp`, `src/plot/table_view.cpp`)
is a generic table view: a grid of text cells with optional per-cell tint,
emphasis, and hover, plus a bounded per-element history (an N-scrubber). It draws
with ImGui (`BeginTable`), not ImPlot, and stays domain-free.

`ScoreTablePlot` (in `leakflow_plugins_crypto_plot`) is the element that fills it
from `AttackStatsPayload`. It complements `ScorePlot`: where the score plot shows
convergence *over N*, the table is the *current scoreboard* — **columns are the
attack units, rows are the ranked candidate guesses** (each cell = guess + score),
sorted by score (default) or by guess value, with the correct-key cell tinted and
the best-by-score cell emphasized. It reads `top_k`/`true_rank`/`true_guess` from
`AttackStats`, so it shares ScorePlot's exact wiring (one `AttackStats ! Tee` can
feed both) and needs no separate truth pad; setting `AttackStats(top_k=256)` makes
every guess a row without changing the plot.

Because the table is "the state at the current N", it is agnostic to *how* N
arrives: a live source pushes a frame per buffer (kept up to `max_history`, an
N-scrubber), and an offline source pushes exactly one frame — the final result,
held on screen after EOS. Same element, no live/offline mode switch. History
trimming: `max_history` = `1` replace (default), `N` keep last N, `0` unbounded.

See `docs/context/modules/plot.md` for the module-boundary summary and
`docs/design/cpa_attack.md` for the score/rank semantics.

## Dynamic Pads Deferred

Do not add dynamic sink pads to `TracePlot` in Phase 22.

The current executor passes only `std::optional<Buffer>` into
`Element::process(...)`; it does not tell an element which sink pad received a
buffer or when all inputs are ready. True dynamic sink pads belong with a
future multi-input execution phase.

Future shape:

```text
TracePlot@p;
@a ! @p.sink_0;
@b ! @p.sink_1;
@c ! @p.sink_2;
```

The group approach gets comparison behavior without changing the executor.

## Time Axis

Sampling rate is a data fact, so prefer metadata.

Canonical metadata key:

```text
sample_rate_hz
```

Display mode is a user choice, so use a property:

```text
x_axis=sample|time_us
```

Resolve sampling rate as:

```text
TracePlot(sample_rate_hz=...) > buffer metadata sample_rate_hz > sample index fallback
```

When `x_axis=time_us` and a sampling rate is available:

```text
time_us = sample_index / sample_rate_hz * 1e6
```

If no sampling rate is available, fall back to sample index and log a warning
or debug message depending on how explicit the user request was.

## TorchConvert

`TorchConvert` is an explicit base plugin element, not a generic autoplugging
converter.

Purpose:

```text
TorchTensorPayload -> TorchTensorPayload
```

Initial properties:

```text
dtype=preserve|float32|float64|...
device=preserve|cpu|cuda:0
contiguous=true|false
copy=auto|always
```

`TracePlot` expects CPU float32 input. Pipelines should use `TorchConvert`
explicitly when needed:

```bash
leakflow run 'NumpySrc(path=traces.npy) ! NumpyToTorch ! TorchConvert(dtype=float32,device=cpu) ! TracePlot'
```

Do not automatically insert `TorchConvert` in Phase 22.

## Window Focus Controls

The controls window shows a `Focus window:` row — one button per open plot window
(traces, scores, table, graph), each of which brings that window to the front on
click (`ImGui::SetWindowFocus`) and un-collapses it. It is a reliable alternative
to `Ctrl+Tab` window switching, which depends on the window manager delivering
keyboard focus/modifiers to the GLFW window (flaky under some Wayland or
focus-follows-mouse setups).

An earlier interactive "Save PNG" screenshot button was removed; screenshotting
is better handled by the OS/compositor, and the plot windows are for live
inspection.

## Useful First UI

The first `TracePlot` UI should include:

- trace index slider for rank-2 inputs,
- previous/next trace controls,
- optional locked trace index for grouped plots,
- overlay/stacked layout,
- title and axis labels,
- sample/time x-axis switch,
- reset zoom,
- auto-scale,
- cursor coordinates,
- basic min/max/mean for the selected trace,
- window focus buttons (Focus window row).

Do not show raw trace tables by default.

## Logging Safety

Plot logs may include:

- backend selected,
- group,
- label,
- rank,
- shape,
- dtype,
- device,
- sampling rate,
- selected trace index,
- warning/failure messages.

Plot logs must not include raw trace values, key bytes, plaintext arrays, tensor
contents, or secret intermediates.

## Validation

New GUI/window rendering behavior is manual-only for Phase 22.

Existing CTest must still pass after implementation.

Manual smoke examples:

```bash
./build/leakflow run 'NumpySrc(path=tests/fixtures/numpy/aes_sync_key_01_traces_first30.npy) ! NumpyToTorch ! TorchConvert(dtype=float32,device=cpu) ! TracePlot(title="AES traces", group=aes)'
```

```bash
./build/leakflow run 'NumpySrc(path=tests/fixtures/numpy/aes_sync_key_01_traces_first30.npy) ! NumpyToTorch ! Tee@t; @t.src_0 ! TracePlot(group=compare,label=raw); @t.src_1 ! TorchConvert(dtype=float32,device=cpu) ! TracePlot(group=compare,label=converted,layout=overlay,lock_trace_index=true)'
```

## Future Plot Elements

Implemented so far: `TracePlot` (trace view), `ScorePlot` (score view), and
`ScoreTablePlot` (table view) — the last two in `leakflow_plugins_crypto_plot`.
Each is one `PlotView` behind the registry above, so the next plot types below
are new views + elements, not runtime edits:

- `RankPlot` / guessing-entropy convergence,
- `CorrelationPlot` / `PoiOverlayPlot`,
- a `[U,G]` correlation/score **heatmap** (the dense full-field view a table
  should not be),
- `SpectrumPlot`,
- report/export helpers for AES validation.

These should wait until the corresponding SCA payloads or AES/PoI helpers
exist.
