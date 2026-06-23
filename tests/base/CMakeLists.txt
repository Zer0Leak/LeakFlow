add_executable(leakflow_base_torch_tensor_tests
    torch_tensor_payload_test.cpp
)
target_link_libraries(leakflow_base_torch_tensor_tests
    PRIVATE
        leakflow_base
)
leakflow_configure_test_target(leakflow_base_torch_tensor_tests)
add_test(NAME leakflow_base_torch_tensor COMMAND leakflow_base_torch_tensor_tests)

add_executable(leakflow_base_torch_tensor_bundle_tests
    torch_tensor_bundle_payload_test.cpp
)
target_link_libraries(leakflow_base_torch_tensor_bundle_tests
    PRIVATE
        leakflow_base
)
leakflow_configure_test_target(leakflow_base_torch_tensor_bundle_tests)
add_test(NAME leakflow_base_torch_tensor_bundle COMMAND leakflow_base_torch_tensor_bundle_tests)

add_executable(leakflow_base_statistics_tests
    statistics_test.cpp
)
target_link_libraries(leakflow_base_statistics_tests
    PRIVATE
        leakflow_base
)
leakflow_configure_test_target(leakflow_base_statistics_tests)
add_test(NAME leakflow_base_statistics COMMAND leakflow_base_statistics_tests)

add_executable(leakflow_base_plot_annotation_payload_tests
    plot_annotation_payload_test.cpp
)
target_link_libraries(leakflow_base_plot_annotation_payload_tests
    PRIVATE
        leakflow_base
)
leakflow_configure_test_target(leakflow_base_plot_annotation_payload_tests)
add_test(NAME leakflow_base_plot_annotation_payload COMMAND leakflow_base_plot_annotation_payload_tests)
