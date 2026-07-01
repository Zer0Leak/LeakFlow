include(FetchContent)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(ZLIB REQUIRED)

FetchContent_Declare(imgui
    GIT_REPOSITORY "https://github.com/ocornut/imgui.git"
    GIT_TAG "v1.92.6"
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(implot
    GIT_REPOSITORY "https://github.com/epezent/implot.git"
    GIT_TAG "v1.0"
)
FetchContent_MakeAvailable(implot)

add_library(leakflow_imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
    "${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp"
)
target_compile_features(leakflow_imgui PUBLIC cxx_std_23)
target_include_directories(leakflow_imgui
    PUBLIC
        "${imgui_SOURCE_DIR}"
        "${imgui_SOURCE_DIR}/backends"
)
target_link_libraries(leakflow_imgui
    PUBLIC
        glfw
        OpenGL::GL
)
set_target_properties(leakflow_imgui PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_library(leakflow_implot STATIC
    "${implot_SOURCE_DIR}/implot.cpp"
    "${implot_SOURCE_DIR}/implot_items.cpp"
)
target_compile_features(leakflow_implot PUBLIC cxx_std_23)
target_include_directories(leakflow_implot
    PUBLIC
        "${implot_SOURCE_DIR}"
)
target_link_libraries(leakflow_implot
    PUBLIC
        leakflow_imgui
)
set_target_properties(leakflow_implot PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_library(leakflow_plot
    imgui_plot_loop.cpp
    pipeline_graph.cpp
    plot_render_util.cpp
    plot_runtime.cpp
    score_view.cpp
)
target_compile_features(leakflow_plot PUBLIC cxx_std_23)
target_include_directories(leakflow_plot
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_plot
    PUBLIC
        leakflow_core
        leakflow_log
    PRIVATE
        leakflow_imgui
        leakflow_implot
        ZLIB::ZLIB
)
set_target_properties(leakflow_plot PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
