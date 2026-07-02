add_executable(leakflow_plugins_crypto_plot_score_plot_tests
    score_plot_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_plot_score_plot_tests
    PRIVATE
        leakflow_plugins_crypto_plot
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_plot_score_plot_tests)
add_test(NAME leakflow_plugins_crypto_plot_score_plot
    COMMAND leakflow_plugins_crypto_plot_score_plot_tests)

add_executable(leakflow_plugins_crypto_plot_score_table_plot_tests
    score_table_plot_test.cpp
)
target_link_libraries(leakflow_plugins_crypto_plot_score_table_plot_tests
    PRIVATE
        leakflow_plugins_crypto_plot
        leakflow_plugins_crypto
)
leakflow_configure_test_target(leakflow_plugins_crypto_plot_score_table_plot_tests)
add_test(NAME leakflow_plugins_crypto_plot_score_table_plot
    COMMAND leakflow_plugins_crypto_plot_score_table_plot_tests)
