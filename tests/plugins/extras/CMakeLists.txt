add_executable(leakflow_plugins_extras_numpy_src_tests
    numpy_src_test.cpp
)
target_link_libraries(leakflow_plugins_extras_numpy_src_tests
    PRIVATE
        leakflow_plugins_extras
)
leakflow_configure_test_target(leakflow_plugins_extras_numpy_src_tests)
add_test(NAME leakflow_plugins_extras_numpy_src COMMAND leakflow_plugins_extras_numpy_src_tests)

add_executable(leakflow_plugins_extras_numpy_to_torch_tests
    numpy_to_torch_test.cpp
)
target_link_libraries(leakflow_plugins_extras_numpy_to_torch_tests
    PRIVATE
        leakflow_plugins_extras
)
leakflow_configure_test_target(leakflow_plugins_extras_numpy_to_torch_tests)
add_test(NAME leakflow_plugins_extras_numpy_to_torch COMMAND leakflow_plugins_extras_numpy_to_torch_tests)

find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(leakflow_plugins_extras_hdf5_src_tests
    hdf5_src_test.cpp
)
target_link_libraries(leakflow_plugins_extras_hdf5_src_tests
    PRIVATE
        leakflow_plugins_extras
        HDF5::HDF5
)
leakflow_configure_test_target(leakflow_plugins_extras_hdf5_src_tests)
add_test(NAME leakflow_plugins_extras_hdf5_src COMMAND leakflow_plugins_extras_hdf5_src_tests)
