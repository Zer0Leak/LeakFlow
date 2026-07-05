add_executable(leakflow_plugins_base_torch_file_io_tests
    torch_file_io_test.cpp
)
target_link_libraries(leakflow_plugins_base_torch_file_io_tests
    PRIVATE
        leakflow_plugins_base
)
target_compile_definitions(leakflow_plugins_base_torch_file_io_tests
    PRIVATE
        LEAKFLOW_TORCH_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01"
)
leakflow_configure_test_target(leakflow_plugins_base_torch_file_io_tests)
add_test(NAME leakflow_plugins_base_torch_file_io COMMAND leakflow_plugins_base_torch_file_io_tests)

add_executable(leakflow_plugins_base_fake_live_src_tests
    fake_live_src_test.cpp
)
target_link_libraries(leakflow_plugins_base_fake_live_src_tests
    PRIVATE
        leakflow_plugins_base
)
target_compile_definitions(leakflow_plugins_base_fake_live_src_tests
    PRIVATE
        LEAKFLOW_TORCH_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01"
)
leakflow_configure_test_target(leakflow_plugins_base_fake_live_src_tests)
add_test(NAME leakflow_plugins_base_fake_live_src COMMAND leakflow_plugins_base_fake_live_src_tests)

add_executable(leakflow_plugins_base_torch_convert_tests
    torch_convert_test.cpp
)
target_link_libraries(leakflow_plugins_base_torch_convert_tests
    PRIVATE
        leakflow_plugins_base
)
leakflow_configure_test_target(leakflow_plugins_base_torch_convert_tests)
add_test(NAME leakflow_plugins_base_torch_convert COMMAND leakflow_plugins_base_torch_convert_tests)

add_executable(leakflow_plugins_base_app_src_tests
    app_src_test.cpp
)
target_link_libraries(leakflow_plugins_base_app_src_tests
    PRIVATE
        leakflow_plugins_base
)
leakflow_configure_test_target(leakflow_plugins_base_app_src_tests)
add_test(NAME leakflow_plugins_base_app_src COMMAND leakflow_plugins_base_app_src_tests)
