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

add_executable(leakflow_sync_threaded_tests
    sync_threaded_test.cpp
)
target_link_libraries(leakflow_sync_threaded_tests
    PRIVATE
        leakflow_plugins_core
)
leakflow_configure_test_target(leakflow_sync_threaded_tests)
add_test(NAME leakflow_sync_threaded COMMAND leakflow_sync_threaded_tests)

add_executable(leakflow_buffer_file_tests
    buffer_file_test.cpp
)
target_link_libraries(leakflow_buffer_file_tests
    PRIVATE
        leakflow_plugins_core
        leakflow_plugins_base
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_buffer_file_tests)
add_test(NAME leakflow_buffer_file COMMAND leakflow_buffer_file_tests)
