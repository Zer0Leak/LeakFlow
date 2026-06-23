add_executable(leakflow_crypto_hamming_tests
    hamming_test.cpp
)
target_link_libraries(leakflow_crypto_hamming_tests
    PRIVATE
        leakflow_crypto
)
leakflow_configure_test_target(leakflow_crypto_hamming_tests)
add_test(NAME leakflow_crypto_hamming COMMAND leakflow_crypto_hamming_tests)

add_executable(leakflow_crypto_aes_leakage_tests
    aes_leakage_test.cpp
)
target_link_libraries(leakflow_crypto_aes_leakage_tests
    PRIVATE
        leakflow_crypto
)
leakflow_configure_test_target(leakflow_crypto_aes_leakage_tests)
add_test(NAME leakflow_crypto_aes_leakage COMMAND leakflow_crypto_aes_leakage_tests)
