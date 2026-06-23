add_executable(leakflow_plugins_crypto_aes_leakage_tests
    aes_leakage_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_aes_leakage_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_aes_leakage_tests)
add_test(NAME leakflow_plugins_crypto_aes_leakage COMMAND leakflow_plugins_crypto_aes_leakage_tests)

add_executable(leakflow_plugins_crypto_pearson_poi_finder_tests
    pearson_poi_finder_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_pearson_poi_finder_tests
    PRIVATE
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_pearson_poi_finder_tests)
add_test(NAME leakflow_plugins_crypto_pearson_poi_finder COMMAND leakflow_plugins_crypto_pearson_poi_finder_tests)

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
