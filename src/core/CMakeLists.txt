add_library(leakflow_core
    buffer.cpp
    buffer_queue.cpp
    caps.cpp
    descriptor.cpp
    descriptor_registry.cpp
    element.cpp
    element_factory_registry.cpp
    pipeline.cpp
    pipeline_segments.cpp
    metadata_forwarding.cpp
    pad.cpp
    payload.cpp
    pipeline_observer.cpp
    pipeline_session.cpp
    property.cpp
    provenance.cpp
    queue_epoch_policy.cpp
    summary_document.cpp
    version.cpp
)
target_compile_features(leakflow_core PUBLIC cxx_std_23)
target_include_directories(leakflow_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(leakflow_core
    PUBLIC
        leakflow_log
)
set_target_properties(leakflow_core PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
