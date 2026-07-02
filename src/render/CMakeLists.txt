add_library(leakflow_render
    summary_renderer.cpp
    terminal_style.cpp
)
target_compile_features(leakflow_render PUBLIC cxx_std_23)
target_include_directories(leakflow_render
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_render
    PUBLIC
        leakflow_core
        fmt::fmt
)
set_target_properties(leakflow_render PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
