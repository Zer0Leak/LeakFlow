add_executable(leakflow_tests
    smoke_test.cpp
)
target_link_libraries(leakflow_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_tests)
add_test(NAME leakflow_smoke COMMAND leakflow_tests)

add_executable(leakflow_caps_tests
    caps_test.cpp
)
target_link_libraries(leakflow_caps_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_caps_tests)
add_test(NAME leakflow_caps COMMAND leakflow_caps_tests)

add_executable(leakflow_buffer_tests
    buffer_test.cpp
)
target_link_libraries(leakflow_buffer_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_buffer_tests)
add_test(NAME leakflow_buffer COMMAND leakflow_buffer_tests)

add_executable(leakflow_vector_clock_tests
    vector_clock_test.cpp
)
target_link_libraries(leakflow_vector_clock_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_vector_clock_tests)
add_test(NAME leakflow_vector_clock COMMAND leakflow_vector_clock_tests)

add_executable(leakflow_buffer_queue_tests
    buffer_queue_test.cpp
)
target_link_libraries(leakflow_buffer_queue_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_buffer_queue_tests)
add_test(NAME leakflow_buffer_queue COMMAND leakflow_buffer_queue_tests)

add_executable(leakflow_buffer_payload_ownership_tests
    buffer_payload_ownership_test.cpp
)
target_link_libraries(leakflow_buffer_payload_ownership_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_buffer_payload_ownership_tests)
add_test(NAME leakflow_buffer_payload_ownership COMMAND leakflow_buffer_payload_ownership_tests)

add_executable(leakflow_payload_tests
    payload_test.cpp
)
target_link_libraries(leakflow_payload_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_payload_tests)
add_test(NAME leakflow_payload COMMAND leakflow_payload_tests)

add_executable(leakflow_property_tests
    property_test.cpp
)
target_link_libraries(leakflow_property_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_property_tests)
add_test(NAME leakflow_property COMMAND leakflow_property_tests)

add_executable(leakflow_element_tests
    element_test.cpp
)
target_link_libraries(leakflow_element_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_element_tests)
add_test(NAME leakflow_element COMMAND leakflow_element_tests)

add_executable(leakflow_element_property_tests
    element_property_test.cpp
)
target_link_libraries(leakflow_element_property_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_element_property_tests)
add_test(NAME leakflow_element_property COMMAND leakflow_element_property_tests)

add_executable(leakflow_element_pad_tests
    element_pad_test.cpp
)
target_link_libraries(leakflow_element_pad_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_element_pad_tests)
add_test(NAME leakflow_element_pad COMMAND leakflow_element_pad_tests)

add_executable(leakflow_descriptor_tests
    descriptor_test.cpp
)
target_link_libraries(leakflow_descriptor_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_descriptor_tests)
add_test(NAME leakflow_descriptor COMMAND leakflow_descriptor_tests)

add_executable(leakflow_element_factory_registry_tests
    element_factory_registry_test.cpp
)
target_link_libraries(leakflow_element_factory_registry_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_element_factory_registry_tests)
add_test(NAME leakflow_element_factory_registry COMMAND leakflow_element_factory_registry_tests)

add_executable(leakflow_linear_pipeline_tests
    linear_pipeline_test.cpp
)
target_link_libraries(leakflow_linear_pipeline_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_linear_pipeline_tests)
add_test(NAME leakflow_linear_pipeline COMMAND leakflow_linear_pipeline_tests)

add_executable(leakflow_linear_pipeline_link_tests
    linear_pipeline_link_test.cpp
)
target_link_libraries(leakflow_linear_pipeline_link_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_linear_pipeline_link_tests)
add_test(NAME leakflow_linear_pipeline_link COMMAND leakflow_linear_pipeline_link_tests)

add_executable(leakflow_pipeline_observer_tests
    pipeline_observer_test.cpp
)
target_link_libraries(leakflow_pipeline_observer_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_pipeline_observer_tests)
add_test(NAME leakflow_pipeline_observer COMMAND leakflow_pipeline_observer_tests)

add_executable(leakflow_metadata_forwarding_tests
    metadata_forwarding_test.cpp
)
target_link_libraries(leakflow_metadata_forwarding_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_metadata_forwarding_tests)
add_test(NAME leakflow_metadata_forwarding COMMAND leakflow_metadata_forwarding_tests)

add_executable(leakflow_pad_tests
    pad_test.cpp
)
target_link_libraries(leakflow_pad_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_pad_tests)
add_test(NAME leakflow_pad COMMAND leakflow_pad_tests)

add_executable(leakflow_pipeline_session_tests
    pipeline_session_test.cpp
)
target_link_libraries(leakflow_pipeline_session_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_pipeline_session_tests)
add_test(NAME leakflow_pipeline_session COMMAND leakflow_pipeline_session_tests)

add_executable(leakflow_pipeline_segments_tests
    pipeline_segments_test.cpp
)
target_link_libraries(leakflow_pipeline_segments_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_pipeline_segments_tests)
add_test(NAME leakflow_pipeline_segments COMMAND leakflow_pipeline_segments_tests)

add_executable(leakflow_threaded_runner_tests
    threaded_runner_test.cpp
)
target_link_libraries(leakflow_threaded_runner_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_threaded_runner_tests)
add_test(NAME leakflow_threaded_runner COMMAND leakflow_threaded_runner_tests)

add_executable(leakflow_threaded_safe_point_tests
    threaded_safe_point_test.cpp
)
target_link_libraries(leakflow_threaded_safe_point_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_threaded_safe_point_tests)
add_test(NAME leakflow_threaded_safe_point COMMAND leakflow_threaded_safe_point_tests)

add_executable(leakflow_threaded_pause_tests
    threaded_pause_test.cpp
)
target_link_libraries(leakflow_threaded_pause_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_threaded_pause_tests)
add_test(NAME leakflow_threaded_pause COMMAND leakflow_threaded_pause_tests)

add_executable(leakflow_aggregator_latest_tests
    aggregator_latest_test.cpp
)
target_link_libraries(leakflow_aggregator_latest_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_aggregator_latest_tests)
add_test(NAME leakflow_aggregator_latest COMMAND leakflow_aggregator_latest_tests)

add_executable(leakflow_profiling_tests
    profiling_test.cpp
)
target_link_libraries(leakflow_profiling_tests
    PRIVATE
        leakflow_core
)
leakflow_configure_test_target(leakflow_profiling_tests)
add_test(NAME leakflow_profiling COMMAND leakflow_profiling_tests)
