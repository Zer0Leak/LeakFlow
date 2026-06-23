find_package(Torch REQUIRED)

add_library(leakflow_base
    numeric_caps.cpp
    plot_annotation_payload.cpp
    statistics.cpp
    torch_tensor_bundle_payload.cpp
    torch_tensor_payload.cpp
)
target_compile_features(leakflow_base PUBLIC cxx_std_23)
target_include_directories(leakflow_base
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_base
    PUBLIC
        leakflow_core
        "${TORCH_LIBRARIES}"
)
if(TORCH_CXX_FLAGS)
    separate_arguments(LEAKFLOW_TORCH_CXX_FLAGS UNIX_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(leakflow_base
        PUBLIC
            ${LEAKFLOW_TORCH_CXX_FLAGS}
    )
endif()
set_target_properties(leakflow_base PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
