add_library(leakflow_plugins_plot SHARED
    descriptor_catalog.cpp
    trace_plot.cpp
)
target_compile_features(leakflow_plugins_plot PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_plot
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_plot
    PUBLIC
        leakflow_base
        leakflow_plot
)
set_target_properties(leakflow_plugins_plot PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
