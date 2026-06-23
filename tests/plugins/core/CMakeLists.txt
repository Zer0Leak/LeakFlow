add_executable(leakflow_plugins_core_elements_tests
    core_elements_test.cpp
)
target_link_libraries(leakflow_plugins_core_elements_tests
    PRIVATE
        leakflow_plugins_core
)
leakflow_configure_test_target(leakflow_plugins_core_elements_tests)
add_test(NAME leakflow_plugins_core_elements COMMAND leakflow_plugins_core_elements_tests)

add_executable(leakflow_tee_tests
    tee_test.cpp
)
target_link_libraries(leakflow_tee_tests
    PRIVATE
        leakflow_plugins_core
)
leakflow_configure_test_target(leakflow_tee_tests)
add_test(NAME leakflow_tee COMMAND leakflow_tee_tests)

add_executable(leakflow_sync_tests
    sync_test.cpp
)
target_link_libraries(leakflow_sync_tests
    PRIVATE
        leakflow_plugins_core
)
leakflow_configure_test_target(leakflow_sync_tests)
add_test(NAME leakflow_sync COMMAND leakflow_sync_tests)
