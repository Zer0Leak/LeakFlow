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

`TracePlot` reads its input and registers a plot-owned snapshot with
`PlotRuntime`.

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

## Save To PNG

`TracePlot` should expose a Save PNG button.

The implementation may use backend-specific framebuffer or render-target
capture. If the first implementation cannot capture a plot reliably on a
backend, the UI should make that state clear and log a warning rather than
silently doing nothing.

Future properties may include:

```text
png_path="/tmp/trace.png"
png_scale=1.0
```

but the first requirement is an interactive save button.

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
- Save PNG button.

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
- PNG path,
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

Likely future elements after TracePlot:

- `CorrelationPlot`,
- `PoiOverlayPlot`,
- `SpectrumPlot`,
- report/export helpers for AES validation.

These should wait until the corresponding SCA payloads or AES/PoI helpers
exist.
