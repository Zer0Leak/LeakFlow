FROM nvidia/cuda:13.3.0-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=clang
ENV CXX=clang++

ARG LIBTORCH_VERSION=2.7.1
ARG LIBTORCH_VARIANT=cpu
ARG LIBTORCH_URL=

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        clang \
        clang-format \
        clang-tidy \
        cmake \
        curl \
        git \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libgl1-mesa-dev \
        libglfw3-dev \
        zlib1g-dev \
        lldb \
        ninja-build \
        python3 \
        sudo \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    libgl1 \
    libegl1 \
    libglvnd0 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    libtorch_url="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_VARIANT}.zip}"; \
    curl -fsSL "${libtorch_url}" -o /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

ENV CMAKE_PREFIX_PATH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib
ENV LEAKFLOW_LOG_LEVEL=warning
ENV LEAKFLOW_LOG_COLOR=auto
ENV LEAKFLOW_SUMMARIES=1

ARG USERNAME=leakflow
ARG USER_UID=1000
ARG USER_GID=1000

RUN set -eux; \
    if getent group "${USER_GID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
        GROUP_NAME="${USERNAME}"; \
    fi; \
    if getent passwd "${USER_UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
        usermod -l "${USERNAME}" "${EXISTING_USER}" || true; \
        usermod -d "/home/${USERNAME}" -m "${USERNAME}" || true; \
        usermod -g "${GROUP_NAME}" "${USERNAME}"; \
    else \
        useradd --uid "${USER_UID}" --gid "${GROUP_NAME}" -m "${USERNAME}"; \
    fi; \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

USER ${USERNAME}

WORKDIR /workspace/LeakFlow
