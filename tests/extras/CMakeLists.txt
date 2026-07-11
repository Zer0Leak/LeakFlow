find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(leakflow_extras_numpy_payload_tests
    numpy_payload_test.cpp
)
target_link_libraries(leakflow_extras_numpy_payload_tests
    PRIVATE
        leakflow_extras
)
leakflow_configure_test_target(leakflow_extras_numpy_payload_tests)
add_test(NAME leakflow_extras_numpy_payload COMMAND leakflow_extras_numpy_payload_tests)

add_executable(leakflow_extras_numpy_to_torch_tests
    numpy_to_torch_test.cpp
)
target_link_libraries(leakflow_extras_numpy_to_torch_tests
    PRIVATE
        leakflow_extras
)
leakflow_configure_test_target(leakflow_extras_numpy_to_torch_tests)
add_test(NAME leakflow_extras_numpy_to_torch COMMAND leakflow_extras_numpy_to_torch_tests)

add_executable(leakflow_extras_hdf5_tensor_dataset_reader_tests
    hdf5_tensor_dataset_reader_test.cpp
)
target_link_libraries(leakflow_extras_hdf5_tensor_dataset_reader_tests
    PRIVATE
        leakflow_extras
        HDF5::HDF5
)
leakflow_configure_test_target(leakflow_extras_hdf5_tensor_dataset_reader_tests)
add_test(
    NAME leakflow_extras_hdf5_tensor_dataset_reader
    COMMAND leakflow_extras_hdf5_tensor_dataset_reader_tests
)
