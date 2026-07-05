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
)
target_compile_definitions(leakflow_plugins_crypto_aes_poi_correctness_tests
    PRIVATE
        LEAKFLOW_AES_FIXTURE_DIR_KEY_01="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01"
        LEAKFLOW_AES_FIXTURE_DIR_KEY_02="${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_02"
)
leakflow_configure_test_target(leakflow_plugins_crypto_aes_poi_correctness_tests)
add_test(NAME leakflow_plugins_crypto_aes_poi_correctness COMMAND leakflow_plugins_crypto_aes_poi_correctness_tests)
