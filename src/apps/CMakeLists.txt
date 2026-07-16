add_library(leakflow_cli STATIC
    leakflow/leakflow_cli.cpp
)
target_compile_features(leakflow_cli PUBLIC cxx_std_23)
target_include_directories(leakflow_cli
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/leakflow"
)
target_link_libraries(leakflow_cli
    PUBLIC
        leakflow_log
        leakflow_plugins_core
        leakflow_plugins_base
        leakflow_plugins_ml
        leakflow_plugins_extras
        leakflow_plugins_crypto
        leakflow_plugins_plot
        leakflow_plugins_ml_plot
        leakflow_plugins_crypto_plot
)
set_target_properties(leakflow_cli PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(leakflow
    leakflow/main.cpp
)
target_link_libraries(leakflow
    PRIVATE
        leakflow_cli
        leakflow_log
)
set_target_properties(leakflow PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

# Link the tensor debug helpers (dtv/ptv/ps) into leakflow for LLDB/GDB. The
# leakflow_torch_debug OBJECT library (src/debug) lands its objects directly in
# the executable, so the C-linkage entry points are never pruned. Built and
# linked only when -DLEAKFLOW_TORCH_DEBUG=ON (the default).
if(LEAKFLOW_TORCH_DEBUG)
    target_link_libraries(leakflow PRIVATE leakflow_torch_debug)
    target_compile_definitions(leakflow PRIVATE LEAKFLOW_TORCH_DEBUG=1)
endif()

add_executable(leakflow-ls
    common/terminal_pager.cpp
    leakflow_ls/main.cpp
)
target_include_directories(leakflow-ls
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(leakflow-ls
    PRIVATE
        leakflow_render
        leakflow_plugins_core
        leakflow_plugins_base
        leakflow_plugins_ml
        leakflow_plugins_extras
        leakflow_plugins_crypto
        leakflow_plugins_plot
        leakflow_plugins_ml_plot
        leakflow_plugins_crypto_plot
)
set_target_properties(leakflow-ls PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(LEAKFLOW_WITH_CUDA)
    set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} --allow-unsupported-compiler")
    enable_language(CUDA)

    add_executable(leakflow_cuda_smoke
        cuda_smoke/leakflow_cuda_smoke.cu
    )
    set_target_properties(leakflow_cuda_smoke PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    )
endif()
