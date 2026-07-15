add_executable(leakflow_rezaeezade_poi_finder
    src/poi_finder_main.cpp
)
target_compile_features(leakflow_rezaeezade_poi_finder PRIVATE cxx_std_23)
target_link_libraries(leakflow_rezaeezade_poi_finder
    PRIVATE
        leakflow_cli
)
set_target_properties(leakflow_rezaeezade_poi_finder PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
