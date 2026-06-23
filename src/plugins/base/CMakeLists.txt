add_library(leakflow_plugins_base SHARED
    descriptor_catalog.cpp
    fake_live_src.cpp
    torch_convert.cpp
    torch_file_sink.cpp
    torch_file_src.cpp
)
target_compile_features(leakflow_plugins_base PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_base
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_base
    PUBLIC
        leakflow_base
)
set_target_properties(leakflow_plugins_base PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
