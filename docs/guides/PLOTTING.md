# LeakFlow Plotting Tutorial

Phase 22 adds the first interactive plotting path through `TracePlot`.

## Build

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
```

The plot runtime uses Dear ImGui, ImPlot, GLFW, OpenGL, and zlib. CMake fetches
Dear ImGui and ImPlot. The host system must provide GLFW, OpenGL, and zlib
development files.

## Docker Plot Windows

`TracePlot` opens a real GLFW/OpenGL window. When running inside Docker, start
the container with host display forwarding before running the plot command:

```bash
xhost +SI:localuser:$(id -un)

docker run --rm --gpus all -it \
  -e DISPLAY \
  -e WAYLAND_DISPLAY= \
  -e XDG_SESSION_TYPE=x11 \
  -e GLFW_PLATFORM=x11 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD:/workspaces/LeakFlow" \
  -w /workspaces/LeakFlow \
  leakflow-dev:latest bash
```

After leaving the container, revoke the local X11 grant if desired:

```bash
xhost -SI:localuser:$(id -un)
```

## CLI Plot Loop

`leakflow run` executes the pipeline synchronously first. `TracePlot` snapshots
incoming tensors into a plot-owned runtime and passes the buffer downstream. If
any plot sessions were registered, the CLI opens the ImGui/ImPlot loop after
pipeline execution and blocks until the window is closed.

```bash
./build/leakflow run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TracePlot(title="AES traces",group=aes,label=trace)'
```

`TracePlot` accepts CPU `float32` `TorchTensorPayload` input with rank 1 or rank
2. Rank 2 is interpreted as `[trace, sample]`.

The window includes a `Save PNG` button. It saves the current window framebuffer
to the path in the `PNG path` field.

## Groups

A `TracePlot` `group` is one ImGui plot window inside the plot runtime.

Multiple `TracePlot` elements with the same `group` enter that window in
pipeline order by default. Set `order` when a command needs a stable display
order that differs from entry order.

The first TracePlot in a group starts the first plot panel, so its `layout`
property has no effect. Later TracePlots use their own `layout` value:

- `layout=overlay` draws into the previous panel.
- `layout=stacked` starts a new panel.

Rank-2 traces use vertical trace sliders on the right side of their plot panel.
Each group window has a `Controls` button that lists the group's TracePlots (in
display order) with a gear to open each element's controls and a drag handle to
reorder them.

The grouped trace-slider lock is a group-level UI toggle (not an element
property). Enable it from the group `Controls` window or by right-clicking any
trace slider. All rank-2 sliders stay visible; moving one shifts the others by
the same delta, preserving their offsets. The lock stops at the first slider that
reaches its top or bottom bound. The right-click `Lock trace sliders` toggle
appears only when the group has more than one trace slider, and behaves the same
in stacked and overlay layouts.

`color` initializes the line color from the CLI. It accepts named colors,
hex RGB/RGBA, and functional RGB/RGBA:

```text
TracePlot(color=blue)
TracePlot(color=#4aa3ff)
TracePlot(color=#4aa3ff80)
TracePlot(color="rgb(74,163,255)")
TracePlot(color="rgba(74,163,255,0.5)")
```

Use quotes around `rgb(...)` and `rgba(...)` in pipeline expressions because
the value contains commas. RGBA alpha initializes the same display alpha as
`alpha`; a non-default `alpha` property overrides the alpha from `rgba`.

## CLI Examples

These examples use the checked-in AES trace fixture and open the plot window
after the pipeline finishes. Close the window to return to the shell.

### One-Line Trace Browser

Use this as the smallest pleasant plot command. It loads the CPU `float32`
fixture and opens a rank-2 trace browser with a trace slider.

```bash
./build/leakflow run \
  'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TracePlot(title="AES trace browser",label=trace,trace_index=4,color=#4aa3ffe6,line_width=1.4)'
```

### No Explicit Group

`group` is optional for a single plot. This keeps the command short while still
using the default plot group internally.

```bash
./build/leakflow run \
  'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TracePlot(title="Single AES trace",label=sample_trace,trace_index=12,x_label=sample,y_label=leakage,line_width=1.8)'
```

### Time Axis

Use `x_axis=time_us` and `sample_rate_hz` when the sample index should be shown
as time. The value below treats the fixture as a 1 GHz capture.

```bash
./build/leakflow run \
  'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TracePlot(title="AES trace in time",label=time_trace,trace_index=8,x_axis=time_us,sample_rate_hz=1000000000,x_label="time (us)",y_label=leakage,line_width=1.5)'
```

### Grouped Overlay

Use `Tee` to send the same tensor into two `TracePlot` elements. Matching
`group` values place the snapshots in the same plot window. The second
`TracePlot(layout=overlay)` joins the first panel, so different `trace_index`
values make the initial overlay compare two traces.

```bash
./build/leakflow run --graph \
  'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Tee@t; @t.src_0 ! TracePlot(title="AES overlay comparison",group=aes_overlay,label=trace_0,trace_index=0,color="rgba(0,0,255,0.9)",line_width=1.6); @t.src_1 ! TracePlot(group=aes_overlay,label=trace_18,trace_index=18,layout=overlay,color="rgba(255,128,0,0.55)",line_width=1.2)'
```

### Grouped Stacked View

Use `layout=stacked` on later `TracePlot` elements when overlays are too
visually dense. The first TracePlot starts the first panel; each later stacked
TracePlot starts another panel.

```bash
./build/leakflow run \
  'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Tee@t; @t.src_0 ! TracePlot(title="AES stacked comparison",group=aes_stack,label=early_trace,trace_index=2,line_width=1.5); @t.src_1 ! TracePlot(group=aes_stack,label=late_trace,trace_index=24,layout=stacked,line_width=1.5)'
```

## Library Loop

Applications can use the same model directly:

```cpp
auto runtime = std::make_shared<leakflow::plot::PlotRuntime>();

auto plot = std::make_shared<leakflow::plugins::plot::TracePlot>(runtime, "plot");
plot->set_property("title", std::string("Tutorial traces"));

leakflow::Pipeline pipeline;
// Add and link a source element, ending in plot.
pipeline.run();

if (runtime->has_sessions()) {
    leakflow::plot::run_until_closed(*runtime);
}
```

Apps that already own an ImGui frame can call `leakflow::plot::draw_plot_runtime`
inside their own frame instead of calling `run_until_closed`.

## Tutorial App

The optional tutorial executable is disabled by default so the normal project
does not add a new user-facing app.

```bash
CXX=clang++ cmake -S . -B build-examples -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch" -DLEAKFLOW_BUILD_EXAMPLES=ON
cmake --build build-examples -j
./build-examples/leakflow_trace_plot_tutorial
```

Pass a `.pt` tensor path as the first argument to plot a different trace array.
