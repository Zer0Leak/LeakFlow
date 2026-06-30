cmake_minimum_required(VERSION 3.20)

project(LeakFlow LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LEAKFLOW_WITH_CUDA "Build the optional CUDA smoke executable" OFF)
option(LEAKFLOW_BUILD_EXAMPLES "Build optional LeakFlow tutorial example applications" OFF)
option(LEAKFLOW_TORCH_DEBUG "Build and link tensor debug helpers (dtv/ptv/ps) for LLDB/GDB" ON)

include(CTest)
include(FetchContent)

set(FMT_DOC OFF CACHE BOOL "Disable fmt documentation for LeakFlow" FORCE)
set(FMT_TEST OFF CACHE BOOL "Disable fmt tests for LeakFlow" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "Disable fmt install rules for LeakFlow" FORCE)
FetchContent_Declare(fmt
    GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"
    GIT_TAG "12.1.0"
)
FetchContent_MakeAvailable(fmt)

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Disable spdlog examples for LeakFlow" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Disable spdlog tests for LeakFlow" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Disable spdlog benchmarks for LeakFlow" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "Disable spdlog install rules for LeakFlow" FORCE)
FetchContent_Declare(spdlog
    GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
    GIT_TAG "v1.15.3"
)
FetchContent_MakeAvailable(spdlog)

add_subdirectory(src/log)
add_subdirectory(src/core)
add_subdirectory(src/render)
add_subdirectory(src/base)
add_subdirectory(src/crypto)
add_subdirectory(src/extras)
add_subdirectory(src/plot)
add_subdirectory(src/plugins/core)
add_subdirectory(src/plugins/base)
add_subdirectory(src/plugins/extras)
add_subdirectory(src/plugins/crypto)
add_subdirectory(src/plugins/plot)

if(LEAKFLOW_TORCH_DEBUG)
    add_subdirectory(src/debug)
endif()

add_subdirectory(src/apps)

if(LEAKFLOW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
