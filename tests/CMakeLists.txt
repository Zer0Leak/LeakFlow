set(LEAKFLOW_TEST_RUNTIME_DIR "${PROJECT_BINARY_DIR}/tests")

function(leakflow_configure_test_target target)
    set_target_properties(${target} PROPERTIES
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${LEAKFLOW_TEST_RUNTIME_DIR}"
    )
endfunction()

add_subdirectory(log)
add_subdirectory(core)
add_subdirectory(render)
add_subdirectory(base)
add_subdirectory(ml)
add_subdirectory(crypto)
add_subdirectory(extras)
add_subdirectory(apps)
add_subdirectory(plugins/core)
add_subdirectory(plugins/base)
add_subdirectory(plugins/ml)
add_subdirectory(plugins/extras)
add_subdirectory(plugins/crypto)
add_subdirectory(plugins/plot)
add_subdirectory(plugins/crypto_plot)
