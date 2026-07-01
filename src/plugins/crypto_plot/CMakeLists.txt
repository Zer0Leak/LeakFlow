add_library(leakflow_plugins_crypto_plot SHARED
    descriptor_catalog.cpp
    score_plot.cpp
)
target_compile_features(leakflow_plugins_crypto_plot PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_crypto_plot
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_crypto_plot
    PUBLIC
        leakflow_plugins_crypto
        leakflow_plot
)
set_target_properties(leakflow_plugins_crypto_plot PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
