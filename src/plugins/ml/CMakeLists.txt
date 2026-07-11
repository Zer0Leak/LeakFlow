add_library(leakflow_plugins_ml SHARED
    clustering_stats.cpp
    descriptor_catalog.cpp
    feature_select.cpp
    gaussian_mixture_element.cpp
)
target_compile_features(leakflow_plugins_ml PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_ml
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_ml
    PUBLIC
        leakflow_ml
        leakflow_base
)
set_target_properties(leakflow_plugins_ml PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
