add_executable(leakflow_trace_plot_tutorial
    trace_plot_tutorial.cpp
)
target_compile_features(leakflow_trace_plot_tutorial PRIVATE cxx_std_23)
target_link_libraries(leakflow_trace_plot_tutorial
    PRIVATE
        leakflow_plugins_core
        leakflow_plugins_base
        leakflow_plugins_plot
)
target_compile_definitions(leakflow_trace_plot_tutorial
    PRIVATE
        LEAKFLOW_TUTORIAL_TRACE_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01/traces_first_50.pt"
)
set_target_properties(leakflow_trace_plot_tutorial PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(leakflow_pipeline_expression_plot_tutorial
    pipeline_expression_plot_tutorial.cpp
)
target_compile_features(leakflow_pipeline_expression_plot_tutorial PRIVATE cxx_std_23)
target_link_libraries(leakflow_pipeline_expression_plot_tutorial
    PRIVATE
        leakflow_cli
)
target_compile_definitions(leakflow_pipeline_expression_plot_tutorial
    PRIVATE
        LEAKFLOW_TUTORIAL_TRACES_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01/traces_first_50.pt"
        LEAKFLOW_TUTORIAL_PLAINTEXTS_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt"
        LEAKFLOW_TUTORIAL_KEY_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01/key_first_50.pt"
)
set_target_properties(leakflow_pipeline_expression_plot_tutorial PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(leakflow_app_src_hdf5_tutorial
    app_src_hdf5_tutorial.cpp
)
target_compile_features(leakflow_app_src_hdf5_tutorial PRIVATE cxx_std_23)
target_link_libraries(leakflow_app_src_hdf5_tutorial
    PRIVATE
        leakflow_cli
        leakflow_plugins_base
        leakflow_extras
)
set_target_properties(leakflow_app_src_hdf5_tutorial PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
