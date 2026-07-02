# VS Code Devcontainer

Phase 2 adds a VS Code devcontainer for building and testing the existing Phase 1 scaffold.

The container uses an official NVIDIA CUDA Ubuntu 26.04 base image so later phases can extend the environment toward NVIDIA/CUDA work. Phase 2 does not add CUDA source code, CUDA C++ compilation, libtorch, GPU-specific project logic, SCA data structures, pipeline classes, plugins, AES, Kyber, YAML, or external project dependencies.

## Included Tools

- `cmake`
- `ninja-build`
- `clang`
- `lldb`
- `git`
- CPU LibTorch installed under `/opt/libtorch`
- Boost filesystem/iostreams development packages for cnpy++ extras support
- GLFW, OpenGL, and zlib development packages for Phase 22 plotting support
- `python3`
- `clang-format`
- `clang-tidy`

## LeakFlow Logging Environment

The devcontainer sets quiet defaults for the LeakFlow-owned logging controls:

```bash
LEAKFLOW_LOG_LEVEL=warning
LEAKFLOW_LOG_COLOR=auto
LEAKFLOW_SUMMARIES=1
```

Override them at the command line when debugging:

```bash
LEAKFLOW_LOG_LEVEL=debug LEAKFLOW_LOG_COLOR=never ./build/leakflow run 'FakeSrc ! FakeSink'
```

See `docs/guides/LOGGING.md` for more examples.

## Open In VS Code

1. Install Docker.
2. Install the VS Code Dev Containers extension.
3. Open this repository in VS Code.
4. Run `Dev Containers: Reopen in Container` from the command palette.

## Validate Inside The Container

Phase 15 and later require LibTorch. The devcontainer installs CPU LibTorch at
`/opt/libtorch` and exports `CMAKE_PREFIX_PATH=/opt/libtorch`. Run these
commands from the repository root inside the devcontainer:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Expected executable output:

```text
LeakFlow dev build
```
