add_library(leakflow_plugins_extras SHARED
    descriptor_catalog.cpp
    numpy_to_torch.cpp
    numpy_src.cpp
)
target_compile_features(leakflow_plugins_extras PUBLIC cxx_std_23)
target_include_directories(leakflow_plugins_extras
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plugins_extras
    PUBLIC
        leakflow_extras
)
set_target_properties(leakflow_plugins_extras PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
