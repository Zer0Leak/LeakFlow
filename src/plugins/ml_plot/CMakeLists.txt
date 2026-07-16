add_library(leakflow_plugins_ml_plot SHARED
    clustering_metrics_table_plot.cpp
    descriptor_catalog.cpp
)
target_compile_features(leakflow_plugins_ml_plot PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_ml_plot
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_ml_plot
    PUBLIC
        leakflow_plugins_ml
        leakflow_plot
)
set_target_properties(leakflow_plugins_ml_plot PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
