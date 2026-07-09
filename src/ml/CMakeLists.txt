find_package(Torch REQUIRED)

add_library(leakflow_ml
    clustering_metrics.cpp
    gaussian_mixture.cpp
    sinkhorn.cpp
)
target_compile_features(leakflow_ml PUBLIC cxx_std_23)
target_include_directories(leakflow_ml
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_ml
    PUBLIC
        "${TORCH_LIBRARIES}"
)
if(TORCH_CXX_FLAGS)
    separate_arguments(LEAKFLOW_TORCH_CXX_FLAGS UNIX_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(leakflow_ml
        PUBLIC
            ${LEAKFLOW_TORCH_CXX_FLAGS}
    )
endif()
set_target_properties(leakflow_ml PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
