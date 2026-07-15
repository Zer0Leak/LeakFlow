# AppSrc HDF5 streaming tutorial

`leakflow_app_src_hdf5_tutorial` demonstrates how an application can feed a
sequence of aligned HDF5 captures into a pipeline through the generic `AppSrc`
element. It is useful when an application owns file discovery or a device loop
and a normal file-source element is not the right boundary.

Each sorted `.h5` file under the selected directory becomes one `AppSrc` frame:

```text
AppSrc@src
  src_0 -> traces
  src_1 -> plaintexts
  src_2 -> keys
```

The tutorial also demonstrates:

- pull-mode frame production;
- aligned multi-pad emission with shared provenance;
- app-reported progress;
- application-added lifecycle properties (`path` and `max_trace_bundles`);
- restart behavior; and
- optional graph display and buffer persistence.

Build examples with:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch" \
  -DLEAKFLOW_BUILD_EXAMPLES=ON
cmake --build build --target leakflow_app_src_hdf5_tutorial -j
```

Run it against a directory containing aligned `.h5` captures:

```bash
./build/leakflow_app_src_hdf5_tutorial CAPTURE_DIRECTORY
./build/leakflow_app_src_hdf5_tutorial --graph CAPTURE_DIRECTORY
```

The default directory is `traces/aes/sync/aes_sync_poi`. Use `--help` for the
full option list.
