# CLI Build Commands

Copy and paste these commands from the repository root.

## Host CPU Build

Use this for normal host development. This does not require CUDA, but LibTorch
is required from Phase 15 onward. Phase 22 plotting also needs GLFW, OpenGL,
and zlib development files on the host. CMake fetches `fmt`, `spdlog`, Dear
ImGui, ImPlot, and other source dependencies during configure. Replace
`CMAKE_PREFIX_PATH` if your LibTorch installation lives somewhere else.

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
```

```bash
cmake --build build -j
```

```bash
ctest --test-dir build --output-on-failure
```

```bash
./build/leakflow
```

Expected output:

```text
LeakFlow 0.10 build
```

## Devcontainer CPU Build

The devcontainer image installs CPU LibTorch under `/opt/libtorch` and exports
`CMAKE_PREFIX_PATH=/opt/libtorch`, so CMake can find `TorchConfig.cmake`
without an extra prefix argument.

Run these commands inside the devcontainer or an interactive shell created from
`leakflow-dev:latest`:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

```bash
cmake --build build -j
```

```bash
ctest --test-dir build --output-on-failure
```

```bash
./build/leakflow
```

## CLI Syntax Reference

The `leakflow run` pipeline syntax is documented in `../reference/CLI_SYNTAX.md`.

Current CLI smoke commands:

```bash
./build/leakflow --help
./build/leakflow --pipeline fake-src-summary
./build/leakflow --pipeline fake-src-tee-summary-sink
./build/leakflow-ls
./build/leakflow-ls TracePlot
```

Pipeline syntax examples:

```bash
./build/leakflow run 'FakeSrc ! Summary'
./build/leakflow run 'FakeSrc(caps_type=sca/test) ! Summary(level=2)'
./build/leakflow run 'FakeSrc(caps_type=sca/test)[caps=sca/fake]{dataset=smoke} ! Summary(level=3)'
./build/leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! FakeSink'
./build/leakflow run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Summary ! FakeSink'
./build/leakflow run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TorchConvert(dtype=float32,device=cpu) ! Summary(level=3) ! FakeSink'
./build/leakflow run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TracePlot(title="AES traces",group=aes,label=trace)'
```

## Build The Docker Image

Run this once, or whenever `.devcontainer/Dockerfile` changes. The image
downloads LibTorch during the build.

```bash
docker build -t leakflow-dev:latest -f .devcontainer/Dockerfile .
```

## Check NVIDIA Visibility In Docker

```bash
docker run --rm --gpus all nvidia/cuda:13.3.0-base-ubuntu26.04 nvidia-smi
```

## Enter Docker Shell

Use this when you want an interactive shell inside the CUDA-capable environment.
The image already sets `CMAKE_PREFIX_PATH=/opt/libtorch`.

```bash
docker run --rm --gpus all -it -v "$PWD:/workspaces/LeakFlow" -w /workspaces/LeakFlow leakflow-dev:latest bash
```

Inside that shell, use the CUDA build commands below.

For `TracePlot` windows from Docker, enter the container with X11/XWayland
forwarding from the Linux desktop host:

```bash
xhost +SI:localuser:$(id -un)
docker run --rm --gpus all -it \
  -e DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD:/workspaces/LeakFlow" \
  -w /workspaces/LeakFlow \
  leakflow-dev:latest bash
```

After leaving the container, revoke that local X11 grant if desired:

```bash
xhost -SI:localuser:$(id -un)
```

## Docker CUDA Build

Run these inside the Docker shell from `/workspaces/LeakFlow`.

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLEAKFLOW_WITH_CUDA=ON
```

```bash
cmake --build build-cuda -j
```

```bash
ctest --test-dir build-cuda --output-on-failure
```

```bash
./build-cuda/leakflow_cuda_smoke
```

Expected output should include:

```text
Vector add OK
```

## One-Shot Docker CUDA Test

Use this if you do not want an interactive Docker shell.

```bash
docker run --rm --gpus all -v "$PWD:/workspaces/LeakFlow" -w /workspaces/LeakFlow leakflow-dev:latest bash -lc 'cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLEAKFLOW_WITH_CUDA=ON && cmake --build build-cuda -j && ctest --test-dir build-cuda --output-on-failure && ./build-cuda/leakflow_cuda_smoke'
```

## Clean Build Directories

```bash
rm -rf build build-cuda
```
