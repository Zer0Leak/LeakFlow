add_library(leakflow_log SHARED
    logger.cpp
)
target_compile_features(leakflow_log PUBLIC cxx_std_23)
target_include_directories(leakflow_log
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_log
    PRIVATE
        spdlog::spdlog
)
set_target_properties(leakflow_log PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
