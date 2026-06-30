# TESTING

This document defines validation expectations for LeakFlow. Phase 0 contains documentation only, so there are no build or test commands to run for this phase.

## Default Toolchain

Use Clang as the default compiler for build and test validation. Select it explicitly during CMake configure with `CXX=clang++`.

LibTorch is required from Phase 15 onward. Phase 16 fetches `cnpy++` and
requires Boost filesystem/iostreams for `leakflow_extras`. Phase 18 fetches
`fmt` for shared terminal formatting and colors. If CMake cannot find LibTorch
automatically, pass its installation prefix with `CMAKE_PREFIX_PATH`. The
devcontainer installs LibTorch at `/opt/libtorch` and exports
`CMAKE_PREFIX_PATH=/opt/libtorch`.

Use a fresh build directory when switching compilers.

Default validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Editor Diagnostics

CMake exports `compile_commands.json` so VS Code, clangd, and C/C++ IntelliSense can use the same compiler flags as the command-line build.

After configuring:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
```

the file `build/compile_commands.json` should exist.

Inside the devcontainer, the shorter configure command is enough because
`CMAKE_PREFIX_PATH` is already set:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

Add to PATH

```bash
export PATH="$PATH:/home/enagl/Documents/CISSA/Repository/LeakFlow/build"
```

Common Pipelines

```bash
leakflow run 'TorchSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Summary ! FakeSink'
```

If VS Code shows stale red diagnostics after a successful command-line build, reload the window or restart the language server.

## Phase 0 Validation

Phase 0 validation is a documentation review:

- Confirm the requested guidance files exist.
- Confirm the documents do not introduce implementation work.
- Confirm the roadmap identifies the next smallest phase.
- Confirm no files outside the requested documentation set were changed.

## Future Implementation Phases

Each implementation phase should leave the repository buildable and testable. Tests should be added close to the behavior they validate and should prefer small deterministic cases.

When a phase adds a smoke executable or validation executable, register it with CTest unless the roadmap explicitly says the executable is manual-only. If an executable has required success output, the CTest entry should check that output when practical.

## Test Layout

Tests are grouped by the project area they validate:

- `tests/core/` contains tests for `leakflow_core`.
- `tests/base/` contains tests for `leakflow_base`.
- `tests/extras/` contains tests for `leakflow_extras`.
- `tests/apps/` contains CLI and application behavior tests.
- `tests/plugins/core/` contains tests for `leakflow_plugins_core`.
- `tests/plugins/extras/` contains tests for `leakflow_plugins_extras`.
- `tests/fixtures/aes/` contains tiny checked-in `.pt` fixtures used by AES and Torch pipeline tests.

The top-level `tests/CMakeLists.txt` delegates to those groups. Individual test
targets should remain close to the CMake file for their group, while CTest names
should stay stable unless a roadmap phase explicitly replaces behavior.

Source targets live under `src/`. Public library headers live under
`include/leakflow/core`, `include/leakflow/base`, `include/leakflow/extras`,
and `include/leakflow/render`, while plugin-facing headers live under
`include/leakflow/plugins/*`. Tests should include headers through the target
include paths rather than using relative filesystem paths.

Future tests should cover:

- Core data structures when they are introduced.
- Boundary conditions and invalid inputs.
- Reproducibility-sensitive behavior.
- Any behavior required by the current roadmap phase.

## Reporting

After each implementation phase, report:

- Changed files.
- Build commands.
- Test commands.
- Whether commands were actually run.
- Whether the phase is complete.
- The next recommended phase.
