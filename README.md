# LeakFlow

LeakFlow is a side-channel analysis framework for reproducible experiments with traces, leakage models, points of interest, labels, attacks, statistical analysis, plots, and optional learning workflows.

This repository contains the full pipeline framework through the live-streaming
phase: the modular core, the LibTorch-backed numeric layer, HDF5/NumPy/Torch I/O,
logging, the ImGui/ImPlot plotting and pipeline-graph runtime, the AES crypto
plugin (leakage + Pearson PoI), the `PipelineSession` control layer, vector-clock
buffer provenance, and threaded live streaming with a `Sync` element and a
player-style runtime. For day-to-day context start at
`docs/context/START_HERE.md`.

## Current Status

The repository contains:

- A C++23 CMake project with modular per-directory targets.
- Core pipeline types: `Caps`, `Buffer`, `Payload`, `Element`, `Pad`, and
  `Pipeline` (DAG executor with `Tee` fan-out and multi-input joins).
- Vector-clock buffer provenance (`Buffer::provenance()`); `Buffer::epoch()` was
  removed.
- `PipelineSession` control/session layer: thread-safe `SetProperty` queue,
  safe-point application, cached buffers, downstream-only rerun, and a
  `Stopped/Running/Paused/Idle` player state machine.
- Threaded live streaming: a unified `Pipeline::run()` pump loop, segments cut at
  every `Queue`, a thread-safe `BufferQueue`, an aggregator fold-match, the `Sync`
  element, and cooperative `std::stop_token` shutdown.
- A LibTorch-backed `leakflow_base` library with tensor/tensor-bundle payloads and
  generic statistics; a `leakflow_crypto` helper library (Hamming/S-box/AES
  leakage).
- A `leakflow_extras` library with a format-neutral tensor-dataset reader
  contract, HDF5 loading, a NumPy `.npy` payload/loader, and NumPy-to-Torch
  conversion.
- The `leakflow_log` logging layer and `leakflow_render` terminal/summary
  rendering.
- The `leakflow_plot` ImGui/ImPlot plotting + pipeline-graph runtime.
- The `leakflow` CLI runner and `leakflow-ls` descriptor inspection tool.
- Linked plugin libraries: `leakflow_plugins_core` (incl. `Sync`),
  `leakflow_plugins_base` (`TorchFileSrc`, `TorchConvert`, `TorchFileSink`,
  `FakeLiveSrc`), `leakflow_plugins_extras` (`Hdf5FileSrc`,
  `FakeLiveHdf5Src`, `NumpySrc`, `NumpyToTorch`),
  `leakflow_plugins_crypto` (`AesLeakage`, `PearsonPoiFinder`,
  `CorrelationPoiToPlotAnnotations`), and `leakflow_plugins_plot` (`TracePlot`).
- A tiny checked-in AES HDF5 dataset fixture under `tests/fixtures/aes/sync/`.
- A Docker/devcontainer toolchain and an optional CUDA smoke executable gated by
  `LEAKFLOW_WITH_CUDA`.

It does not yet contain Kyber, YAML/config running, a standalone GUI frontend,
or dynamic plugin loading.

The active guidance files are:

- `AGENTS.md`: repository instructions for agents and contributors.
- `ROADMAP.md`: phase boundaries and next-step sequencing.
- `TESTING.md`: validation expectations.
- `docs/context/START_HERE.md`: compact context-loading entry point.
- `docs/context/CURRENT_STATE.md`: compact current-state summary.
- `docs/context/ACTIVE_PHASE.md`: current phase/task brief.
- `docs/context/MODULE_MAP.md`: areas, targets, tests, and module context files.
- `docs/context/ARCHITECTURE_CONTRACTS.md`: current architecture boundaries.
- `docs/design/architecture.md`: current architecture and dataflow design source.
- `docs/design/dataflow_sync_model.md`: dataflow / vector-clock / live-streaming
  design of record (with a CLI cookbook).
- `docs/reference/CODING_STYLE.md`: coding conventions.
- `docs/reference/CLI_SYNTAX.md`: `leakflow run` pipeline syntax and examples.
- `docs/guides/CLI_BUILD.md`: build, test, Docker, CUDA, and CLI commands.
- `docs/guides/COOL_PIPELINES.md`: demonstration-ready LeakFlow pipelines.

## Source Layout

The top-level `CMakeLists.txt` owns project configuration and delegates target
definitions to directory-level CMake files:

```text
include/leakflow/core/                    public core library headers
include/leakflow/log/                      logging-layer headers
include/leakflow/render/                  shared terminal render headers
include/leakflow/base/                    public base data-layer headers
include/leakflow/crypto/                   crypto/SCA helper headers
include/leakflow/extras/                  public extras helper headers
include/leakflow/plot/                     plotting + pipeline-graph headers
include/leakflow/plugins/core/            core plugin-facing headers
include/leakflow/plugins/base/            base plugin-facing headers
include/leakflow/plugins/extras/          extras plugin-facing headers
include/leakflow/plugins/crypto/           crypto plugin-facing headers
include/leakflow/plugins/plot/             plot plugin-facing headers
src/core/                                 leakflow_core sources and CMake target
src/log/                                   leakflow_log logging layer
src/render/                               shared terminal render sources
src/base/                                 LibTorch-backed base tensor payload library
src/crypto/                                leakflow_crypto leakage helper library
src/extras/                               extras HDF5/NumPy dataset and conversion library
src/plot/                                  leakflow_plot ImGui/ImPlot runtime
src/apps/common/                          app-only reusable helpers
src/apps/leakflow/                        leakflow executable and CLI helper library
src/apps/leakflow_ls/                     leakflow-ls inspect executable
src/apps/cuda_smoke/                      optional CUDA smoke executable source
src/plugins/core/                         linked core plugin shared library
src/plugins/base/                         linked base plugin shared library
src/plugins/extras/                       linked extras plugin shared library
src/plugins/crypto/                        linked crypto plugin shared library
src/plugins/plot/                          linked plot plugin shared library
tests/core/                               core library tests (incl. live/threaded)
tests/log/                                 logging tests
tests/render/                              render tests
tests/base/                               base tensor payload tests
tests/crypto/                              crypto helper tests
tests/extras/                             extras HDF5/NumPy reader and payload tests
tests/apps/                               CLI and application tests
tests/plugins/core/                       core plugin element tests (incl. Sync)
tests/plugins/base/                       base plugin element tests (incl. FakeLiveSrc)
tests/plugins/extras/                     extras plugin element tests
tests/plugins/crypto/                      crypto plugin element + AES PoI correctness
tests/plugins/plot/                        plot plugin element tests
tests/fixtures/aes/                       tiny checked-in AES HDF5 dataset fixture
```

Subdirectory CMake files inherit top-level options through normal
`add_subdirectory()` scope and keep compile features, include paths, and link
dependencies attached to their own targets.

Application-facing headers live under `include/leakflow/core`,
`include/leakflow/base`, `include/leakflow/extras`, and
`include/leakflow/render`. Plugin-facing headers live under
`include/leakflow/plugins/*` because they describe linked pipeline features
rather than general application utility APIs.

## CLI Syntax

LeakFlow includes a small GStreamer-inspired CLI language for manual pipelines:

```bash
leakflow run 'FakeSrc ! Summary'
leakflow run 'FakeSrc(caps_type=sca/test)[caps=sca/fake]{dataset=smoke} ! Summary(level=3)'
leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! FakeSink'
leakflow run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! Summary ! FakeSink'
```

See `docs/reference/CLI_SYNTAX.md` for the complete target syntax, including element names, properties, pad references, pad caps annotations, metadata annotations, Tee branching, quoting, and examples.

## Development Rule

Development is intentionally incremental. Each phase should add only the smallest useful piece requested for that phase, and each implementation phase should leave the repository buildable and testable.

When a phase is requested, read `AGENTS.md` and `ROADMAP.md` before making changes.

## Toolchain

Use Clang as the default C++ compiler and Ninja as the default generator.
LibTorch is required from Phase 15 onward. Phase 16 fetches `cnpy++` and
requires Boost filesystem/iostreams for `leakflow_extras`; current dataset I/O
also requires the HDF5 development library (Ubuntu package `libhdf5-dev`). Phase 18 fetches
`fmt` for shared terminal formatting and colors. If CMake cannot find LibTorch
automatically, pass its installation prefix with `CMAKE_PREFIX_PATH`:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Future formatting and static-analysis checks should use clang tooling such as `clang-format` and `clang-tidy` when those checks are introduced by a roadmap phase.

## CUDA Smoke

CUDA support is optional and disabled by default. To build and validate the CUDA smoke executable in a CUDA-capable environment:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLEAKFLOW_WITH_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
./build-cuda/leakflow_cuda_smoke
```
