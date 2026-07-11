add_executable(leakflow_plugins_ml_gaussian_mixture_element_tests
    gaussian_mixture_element_test.cpp
)
target_link_libraries(leakflow_plugins_ml_gaussian_mixture_element_tests
    PRIVATE
        leakflow_plugins_ml
)
leakflow_configure_test_target(leakflow_plugins_ml_gaussian_mixture_element_tests)
add_test(NAME leakflow_plugins_ml_gaussian_mixture_element
    COMMAND leakflow_plugins_ml_gaussian_mixture_element_tests)

add_executable(leakflow_plugins_ml_elements_tests
    ml_elements_test.cpp
)
target_link_libraries(leakflow_plugins_ml_elements_tests
    PRIVATE
        leakflow_plugins_ml
)
leakflow_configure_test_target(leakflow_plugins_ml_elements_tests)
add_test(NAME leakflow_plugins_ml_elements COMMAND leakflow_plugins_ml_elements_tests)
