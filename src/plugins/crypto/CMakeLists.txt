add_library(leakflow_plugins_crypto SHARED
    aes_leakage.cpp
    aes_leakage_hypothesis.cpp
    correlation_payload.cpp
    correlation_poi_to_plot_annotations.cpp
    correlation_poi_payload.cpp
    cpa_attack.cpp
    dpa_attack.cpp
    attack_payload.cpp
    attack_stats.cpp
    attack_stats_to_plot_annotations.cpp
    descriptor_catalog.cpp
    pearson_correlator.cpp
    poi_select.cpp
)
target_compile_features(leakflow_plugins_crypto PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_crypto
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_crypto
    PUBLIC
        leakflow_crypto
)
set_target_properties(leakflow_plugins_crypto PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
