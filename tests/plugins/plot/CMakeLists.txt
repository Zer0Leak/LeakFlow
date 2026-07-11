add_executable(leakflow_plugins_plot_trace_plot_tests
    trace_plot_test.cpp
)
target_link_libraries(leakflow_plugins_plot_trace_plot_tests
    PRIVATE
        leakflow_plugins_plot
)
leakflow_configure_test_target(leakflow_plugins_plot_trace_plot_tests)
add_test(NAME leakflow_plugins_plot_trace_plot COMMAND leakflow_plugins_plot_trace_plot_tests)

add_executable(leakflow_plugins_plot_heatmap_plot_tests
    heatmap_plot_test.cpp
)
target_link_libraries(leakflow_plugins_plot_heatmap_plot_tests
    PRIVATE
        leakflow_plugins_plot
)
leakflow_configure_test_target(leakflow_plugins_plot_heatmap_plot_tests)
add_test(NAME leakflow_plugins_plot_heatmap_plot COMMAND leakflow_plugins_plot_heatmap_plot_tests)

add_executable(leakflow_plugins_plot_pipeline_graph_tests
    pipeline_graph_test.cpp
)
target_link_libraries(leakflow_plugins_plot_pipeline_graph_tests
    PRIVATE
        leakflow_plot
)
leakflow_configure_test_target(leakflow_plugins_plot_pipeline_graph_tests)
add_test(NAME leakflow_plugins_plot_pipeline_graph COMMAND leakflow_plugins_plot_pipeline_graph_tests)
