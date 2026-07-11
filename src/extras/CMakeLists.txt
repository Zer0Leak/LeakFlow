include(FetchContent)

find_package(HDF5 REQUIRED COMPONENTS C)

if(POLICY CMP0167)
    set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
endif()

set(CNPYPP_USE_LIBZIP OFF CACHE BOOL "Disable cnpy++ NPZ support for LeakFlow Phase 16" FORCE)
set(CNPYPP_SPAN_IMPL BOOST CACHE STRING "Use Boost span for cnpy++ C++17 sources" FORCE)
FetchContent_Declare(cnpypp
    GIT_REPOSITORY "https://github.com/mreininghaus/cnpypp.git"
    GIT_TAG "c6cd4e2078e4f39e862720b66fb211c45577c510"
)
FetchContent_MakeAvailable(cnpypp)
target_compile_features(cnpy++ PUBLIC cxx_std_20)
set_target_properties(cnpy++ PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(leakflow_extras
    hdf5_tensor_dataset_reader.cpp
    numpy_payload.cpp
    numpy_to_torch.cpp
    tensor_dataset_reader.cpp
)
target_compile_features(leakflow_extras PUBLIC cxx_std_23)
target_include_directories(leakflow_extras
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_extras
    PUBLIC
        leakflow_base
        cnpy++
    PRIVATE
        HDF5::HDF5
)
set_target_properties(leakflow_extras PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
