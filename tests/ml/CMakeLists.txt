add_executable(leakflow_ml_gaussian_mixture_tests
    gaussian_mixture_test.cpp
)
target_link_libraries(leakflow_ml_gaussian_mixture_tests
    PRIVATE
        leakflow_ml
)
leakflow_configure_test_target(leakflow_ml_gaussian_mixture_tests)
add_test(NAME leakflow_ml_gaussian_mixture COMMAND leakflow_ml_gaussian_mixture_tests)

add_executable(leakflow_ml_sinkhorn_tests
    sinkhorn_test.cpp
)
target_link_libraries(leakflow_ml_sinkhorn_tests
    PRIVATE
        leakflow_ml
)
leakflow_configure_test_target(leakflow_ml_sinkhorn_tests)
add_test(NAME leakflow_ml_sinkhorn COMMAND leakflow_ml_sinkhorn_tests)

add_executable(leakflow_ml_clustering_metrics_tests
    clustering_metrics_test.cpp
)
target_link_libraries(leakflow_ml_clustering_metrics_tests
    PRIVATE
        leakflow_ml
)
leakflow_configure_test_target(leakflow_ml_clustering_metrics_tests)
add_test(NAME leakflow_ml_clustering_metrics COMMAND leakflow_ml_clustering_metrics_tests)

add_executable(leakflow_ml_clustering_evaluation_tests
    clustering_evaluation_test.cpp
)
target_link_libraries(leakflow_ml_clustering_evaluation_tests
    PRIVATE
        leakflow_ml
)
leakflow_configure_test_target(leakflow_ml_clustering_evaluation_tests)
add_test(NAME leakflow_ml_clustering_evaluation COMMAND leakflow_ml_clustering_evaluation_tests)
