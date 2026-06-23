# Devcontainer Install Guide

This guide covers the host-side setup needed to use the LeakFlow VS Code devcontainer.

The devcontainer is based on the official NVIDIA CUDA Docker image `nvidia/cuda:13.3.0-devel-ubuntu26.04`. The host must provide Docker and NVIDIA GPU container support. The host does not need the CUDA Toolkit installed separately for this Phase 2 scaffold; it does need a working NVIDIA driver if GPU access is expected.

Official references:

- NVIDIA CUDA images: https://hub.docker.com/r/nvidia/cuda
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Docker GPU access: https://docs.docker.com/engine/containers/gpu/
- Docker Linux post-install steps: https://docs.docker.com/engine/install/linux-postinstall/
- VS Code Dev Containers: https://code.visualstudio.com/docs/devcontainers/containers

## Host Prerequisites

Install these on the host before opening the devcontainer:

- NVIDIA GPU driver.
- Docker Engine.
- NVIDIA Container Toolkit.
- VS Code.
- VS Code Dev Containers extension.

Check the NVIDIA driver:

```bash
nvidia-smi
```

If `nvidia-smi` does not work on the host, fix the host driver before debugging the devcontainer.

## Docker User Group

Docker can be used with `sudo`, but it is more convenient to add your user to the `docker` group.

```bash
sudo groupadd docker
sudo usermod -aG docker "$USER"
newgrp docker
docker run hello-world
```

Log out and back in if `newgrp docker` does not refresh the session cleanly.

Important: membership in the `docker` group effectively grants root-level access on the host. Only add trusted users.

## Install NVIDIA Container Toolkit

Follow NVIDIA's official Container Toolkit installation guide for your host distribution:

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

For Fedora 44, the RPM repository path is:

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo dnf install -y nvidia-container-toolkit
```

After installation, configure Docker to use the NVIDIA runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Verify Docker GPU Access

Run a CUDA container from the host:

```bash
docker run --rm --gpus all nvidia/cuda:13.3.0-base-ubuntu26.04 nvidia-smi
```

If this command works, Docker can see the NVIDIA GPU.

## Build And Run The LeakFlow Image Manually

Before opening VS Code, you can build the same image from the repository root:

```bash
docker build -t leakflow-dev:latest -f .devcontainer/Dockerfile .
```

Enter the container with GPU access and the repository mounted:

```bash
docker run --rm --gpus all -v "$PWD:/workspace/LeakFlow" -w /workspace/LeakFlow leakflow-dev:latest bash
```

For interactive `TracePlot` windows, the container also needs access to the
host desktop display. On a Linux desktop using X11 or XWayland, authorize the
local user and start the container with the X11 socket and NVIDIA graphics
driver capability:

```bash
xhost +SI:localuser:$(id -un)
docker run --rm --gpus all -it \
  -e DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD:/workspace/LeakFlow" \
  -w /workspace/LeakFlow \
  leakflow-dev:latest bash
```

After leaving the container, revoke the local X11 grant if desired:

```bash
xhost -SI:localuser:$(id -un)
```

Inside the container, build and test the source code. Phase 15 and later require
LibTorch; Phase 16 also requires Boost filesystem/iostreams for the cnpy++-backed
extras layer. The LeakFlow devcontainer image installs CPU LibTorch at
`/opt/libtorch` and exports `CMAKE_PREFIX_PATH=/opt/libtorch`, so no extra
prefix argument is needed. It also sets quiet LeakFlow logging defaults:

```bash
LEAKFLOW_LOG_LEVEL=warning
LEAKFLOW_LOG_COLOR=auto
LEAKFLOW_SUMMARIES=1
```

Build and validate:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Try a logging smoke command:

```bash
LEAKFLOW_LOG_LEVEL=debug LEAKFLOW_LOG_COLOR=never ./build/leakflow run 'FakeSrc ! FakeSink'
```

## Open LeakFlow In The Devcontainer

From VS Code:

1. Open this repository folder.
2. Run `Dev Containers: Reopen in Container`.
3. Wait for VS Code to build and attach to the container.

Inside the container, validate the current scaffold:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Expected executable output:

```text
LeakFlow dev build
```

## Notes

- The CUDA image is used as the base environment for future NVIDIA/CUDA phases.
- Phase 2 did not add CUDA source code, CUDA C++ compilation, LibTorch, or GPU-specific project logic. Current phases require LibTorch for the base numerical layer.
- If GPU access works with `docker run --gpus all` but not in VS Code, rebuild the devcontainer and confirm Docker is the active container engine used by VS Code.
- If Docker was installed after adding the NVIDIA runtime, restart Docker before reopening the devcontainer.
