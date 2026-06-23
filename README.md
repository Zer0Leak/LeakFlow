# LeakFlow

LeakFlow is a side-channel analysis framework for reproducible experiments with traces, leakage models, points of interest, labels, attacks, statistical analysis, plots, and optional learning workflows.

This repository currently contains the core pipeline scaffold, Docker/devcontainer toolchain, optional CUDA smoke executable, Phase 13 CLI pipeline runner, Phase 14 modular CMake/source/test layout, Phase 15 LibTorch-backed base tensor payload layer, Phase 16 cnpy++-backed extras NumPy payload layer, Phase 17 linked extras plugin with `NumpySrc`, Phase 18 structured Summary rendering, post-Phase-18 source/include layout consolidation, post-Phase-18 public include tree unification, inspect tools, and linked plugin libraries.

## Current Status

The repository contains:

- A minimal C++23 CMake scaffold.
- A core library, small CPU executable, and CPU smoke test.
- Core pipeline types such as `Caps`, `Buffer`, `Payload`, `Element`, `Pad`, and `LinearPipeline`.
- A LibTorch-backed `leakflow_base` library with tensor and tensor-bundle payloads.
- A cnpy++-backed `leakflow_extras` library with a NumPy `.npy` payload and loader, layered above `leakflow_base`.
- The `leakflow` CLI runner and `leakflow-ls` descriptor inspection tool.
- The linked `leakflow_plugins_core` shared library with generic core elements.
- The linked `leakflow_plugins_base` shared library with `TorchSrc`, `TorchConvert`, and `TorchFileSink`.
- The linked `leakflow_plugins_extras` shared library with `NumpySrc` and `NumpyToTorch`.
- Structured buffer and payload summaries with shared terminal color/theme/glyph rendering.
- Modular CMake files for core, base, extras, app, plugin, and grouped test targets.
- Tiny checked-in Torch tensor fixtures derived from the local AES synchronized trace dataset.
- A Docker/devcontainer toolchain.
- An optional CUDA smoke executable gated by `LEAKFLOW_WITH_CUDA`.

It does not yet contain Kyber, YAML, a standalone GUI frontend, dataset-specific loaders, full experiment pipelines, or dynamic plugin loading.

The active guidance files are:

- `AGENTS.md`: repository instructions for agents and contributors.
- `ROADMAP.md`: phase boundaries and next-step sequencing.
- `TESTING.md`: validation expectations.
- `docs/context/START_HERE.md`: compact context-loading entry point.
- `docs/context/ARCHITECTURE_CONTRACTS.md`: current architecture boundaries.
- `docs/design/architecture.md`: current architecture and dataflow design source.
- `docs/reference/CODING_STYLE.md`: coding conventions.
- `docs/reference/CLI_SYNTAX.md`: `leakflow run` pipeline syntax and examples.
- `docs/guides/CLI_BUILD.md`: build, test, Docker, CUDA, and CLI commands.
- `docs/guides/COOL_PIPELINES.md`: demonstration-ready LeakFlow pipelines.

## Source Layout

The top-level `CMakeLists.txt` owns project configuration and delegates target
definitions to directory-level CMake files:

```text
include/leakflow/core/                    public core library headers
include/leakflow/render/                  shared terminal render headers
include/leakflow/base/                    public base data-layer headers
include/leakflow/extras/                  public extras helper headers
include/leakflow/plugins/core/            core plugin-facing headers
include/leakflow/plugins/extras/          extras plugin-facing headers
include/leakflow/plugins/base/            base plugin-facing headers
src/core/                                 leakflow_core sources and CMake target
src/render/                               shared terminal render sources
src/base/                                 LibTorch-backed base tensor payload library
src/extras/                               cnpy++-backed extras NumPy payload library
src/apps/common/                          app-only reusable helpers
src/apps/leakflow/                        leakflow executable and CLI helper library
src/apps/leakflow_ls/                     leakflow-ls inspect executable
src/apps/cuda_smoke/                      optional CUDA smoke executable source
src/plugins/core/                         linked core plugin shared library
src/plugins/extras/                       linked extras plugin shared library
src/plugins/base/                         linked base plugin shared library
tests/core/                               core library tests
tests/base/                               base tensor payload tests
tests/extras/                             extras NumPy payload tests
tests/apps/                               CLI and application tests
tests/plugins/core/                       core plugin element tests
tests/plugins/base/                       base plugin element tests
tests/plugins/extras/                     extras plugin element tests
tests/fixtures/aes/                       tiny checked-in AES Torch tensor fixtures
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
leakflow run 'TorchSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Summary ! FakeSink'
```

See `docs/reference/CLI_SYNTAX.md` for the complete target syntax, including element names, properties, pad references, pad caps annotations, metadata annotations, Tee branching, quoting, and examples.

## Development Rule

Development is intentionally incremental. Each phase should add only the smallest useful piece requested for that phase, and each implementation phase should leave the repository buildable and testable.

When a phase is requested, read `AGENTS.md` and `ROADMAP.md` before making changes.

## Toolchain

Use Clang as the default C++ compiler and Ninja as the default generator.
LibTorch is required from Phase 15 onward. Phase 16 fetches `cnpy++` and
requires Boost filesystem/iostreams for `leakflow_extras`. Phase 18 fetches
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
