add_executable(leakflow_plugins_crypto_aes_leakage_tests
    aes_leakage_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_aes_leakage_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_aes_leakage_tests)
add_test(NAME leakflow_plugins_crypto_aes_leakage COMMAND leakflow_plugins_crypto_aes_leakage_tests)

add_executable(leakflow_plugins_crypto_aes_leakage_hypothesis_tests
    aes_leakage_hypothesis_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_aes_leakage_hypothesis_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_aes_leakage_hypothesis_tests)
add_test(NAME leakflow_plugins_crypto_aes_leakage_hypothesis COMMAND leakflow_plugins_crypto_aes_leakage_hypothesis_tests)

add_executable(leakflow_plugins_crypto_cpa_attack_tests
    cpa_attack_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_cpa_attack_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_cpa_attack_tests)
add_test(NAME leakflow_plugins_crypto_cpa_attack COMMAND leakflow_plugins_crypto_cpa_attack_tests)

add_executable(leakflow_plugins_crypto_dpa_attack_tests
    dpa_attack_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_dpa_attack_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_dpa_attack_tests)
add_test(NAME leakflow_plugins_crypto_dpa_attack COMMAND leakflow_plugins_crypto_dpa_attack_tests)

add_executable(leakflow_plugins_crypto_attack_stats_to_plot_annotations_tests
    attack_stats_to_plot_annotations_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_attack_stats_to_plot_annotations_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_attack_stats_to_plot_annotations_tests)
add_test(NAME leakflow_plugins_crypto_attack_stats_to_plot_annotations
    COMMAND leakflow_plugins_crypto_attack_stats_to_plot_annotations_tests)

add_executable(leakflow_plugins_crypto_poi_split_tests
    poi_split_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_poi_split_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_poi_split_tests)
add_test(NAME leakflow_plugins_crypto_poi_split COMMAND leakflow_plugins_crypto_poi_split_tests)

add_executable(leakflow_plugins_crypto_aes_poi_correctness_tests
    aes_poi_correctness_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_aes_poi_correctness_tests
    PRIVATE
        leakflow_plugins_crypto
        leakflow_extras
)
target_compile_definitions(leakflow_plugins_crypto_aes_poi_correctness_tests
    PRIVATE
        LEAKFLOW_AES_HDF5="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01.h5"
)
leakflow_configure_test_target(leakflow_plugins_crypto_aes_poi_correctness_tests)
add_test(NAME leakflow_plugins_crypto_aes_poi_correctness COMMAND leakflow_plugins_crypto_aes_poi_correctness_tests)

add_executable(leakflow_plugins_crypto_hw_class_tests
    hw_class_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_hw_class_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_hw_class_tests)
add_test(NAME leakflow_plugins_crypto_hw_class COMMAND leakflow_plugins_crypto_hw_class_tests)

add_executable(leakflow_plugins_crypto_correlation_poi_to_indexes_tests
    correlation_poi_to_indexes_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_correlation_poi_to_indexes_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_correlation_poi_to_indexes_tests)
add_test(NAME leakflow_plugins_crypto_correlation_poi_to_indexes
    COMMAND leakflow_plugins_crypto_correlation_poi_to_indexes_tests)

add_executable(leakflow_plugins_crypto_poi_correlation_tests
    poi_correlation_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_poi_correlation_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_poi_correlation_tests)
add_test(NAME leakflow_plugins_crypto_poi_correlation COMMAND leakflow_plugins_crypto_poi_correlation_tests)
