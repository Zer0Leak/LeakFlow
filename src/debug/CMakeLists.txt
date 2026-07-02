find_package(Torch REQUIRED)

# Debug-only tensor inspection helpers (dtv/ptv/ps/pt + tstr/tsstr).
#
# Built as an OBJECT library so its objects link directly into whatever target
# pulls it in. That guarantees the C-linkage debugger entry points survive (a
# static archive would prune the unreferenced symbols). Any debuggable executable
# can link it; today only `leakflow` does.
add_library(leakflow_torch_debug OBJECT
    tensor_debug.cpp
)
target_compile_features(leakflow_torch_debug PUBLIC cxx_std_23)
target_include_directories(leakflow_torch_debug
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
# PUBLIC so consumers link libtorch (the object file references it); compile flags
# stay PRIVATE since only this unit is compiled against Torch.
target_link_libraries(leakflow_torch_debug
    PUBLIC
        "${TORCH_LIBRARIES}"
)
if(TORCH_CXX_FLAGS)
    separate_arguments(LEAKFLOW_TORCH_DEBUG_FLAGS UNIX_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(leakflow_torch_debug PRIVATE ${LEAKFLOW_TORCH_DEBUG_FLAGS})
endif()
set_target_properties(leakflow_torch_debug PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
