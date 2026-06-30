add_executable(leakflow_log_tests
    logger_test.cpp
)
target_link_libraries(leakflow_log_tests
    PRIVATE
        leakflow_log
)
leakflow_configure_test_target(leakflow_log_tests)
add_test(NAME leakflow_log COMMAND leakflow_log_tests)
