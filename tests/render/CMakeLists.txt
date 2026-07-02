add_executable(leakflow_render_summary_tests
    summary_renderer_test.cpp
)
target_link_libraries(leakflow_render_summary_tests
    PRIVATE
        leakflow_render
)
leakflow_configure_test_target(leakflow_render_summary_tests)
add_test(NAME leakflow_render_summary COMMAND leakflow_render_summary_tests)
