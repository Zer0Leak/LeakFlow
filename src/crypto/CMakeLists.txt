add_library(leakflow_crypto
    aes.cpp
    hamming.cpp
)
target_compile_features(leakflow_crypto PUBLIC cxx_std_23)
target_include_directories(leakflow_crypto
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_crypto
    PUBLIC
        leakflow_base
)
set_target_properties(leakflow_crypto PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
