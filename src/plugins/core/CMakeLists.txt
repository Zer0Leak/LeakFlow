add_library(leakflow_plugins_core SHARED
    buffer_summary.cpp
    bytes_payload.cpp
    descriptor_catalog.cpp
    fake_sink.cpp
    fake_src.cpp
    file_sink.cpp
    file_src.cpp
    queue.cpp
    summary.cpp
    sync.cpp
    tee.cpp
)
target_compile_features(leakflow_plugins_core PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_core
    PUBLIC
        leakflow_render
)
set_target_properties(leakflow_plugins_core PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
