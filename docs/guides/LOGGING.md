# LeakFlow Logging Tutorial

These examples are intended for copy/paste testing from the repository root
after building LeakFlow.

## Build First

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
```

## Defaults

Default logging is quiet:

```bash
./build/leakflow run 'FakeSrc ! Summary'
```

Default summary output is enabled:

```bash
./build/leakflow run 'FakeSrc ! Summary'
```

## Log Levels

Show high-level pipeline and element events:

```bash
./build/leakflow --log-level info --log-color never run 'FakeSrc ! FakeSink'
```

Show detailed pipeline mechanics:

```bash
./build/leakflow --log-level debug --log-color never run 'FakeSrc ! FakeSink'
```

Disable logs completely:

```bash
./build/leakflow --log-level off run 'FakeSrc ! FakeSink'
```

Use an environment variable:

```bash
LEAKFLOW_LOG_LEVEL=debug ./build/leakflow run 'FakeSrc ! FakeSink'
```

CLI options override environment variables:

```bash
LEAKFLOW_LOG_LEVEL=trace ./build/leakflow --log-level info run 'FakeSrc ! FakeSink'
```

## Color

Disable log colors for copy/paste and tests:

```bash
./build/leakflow --log-level debug --log-color never run 'FakeSrc ! FakeSink'
```

Force log colors:

```bash
./build/leakflow --log-level debug --log-color always run 'FakeSrc ! FakeSink'
```

Use automatic terminal detection:

```bash
./build/leakflow --log-level debug --log-color auto run 'FakeSrc ! FakeSink'
```

Use environment variables:

```bash
LEAKFLOW_LOG_LEVEL=debug LEAKFLOW_LOG_COLOR=never ./build/leakflow run 'FakeSrc ! FakeSink'
```

## Filters

Only logs from the `FakeSrc` element type:

```bash
./build/leakflow --log-level debug --log-color never --log-filter 'element=FakeSrc' run 'FakeSrc ! FakeSink'
```

Only logs from a named element instance:

```bash
./build/leakflow --log-level debug --log-color never --log-filter 'element_name=src' run 'FakeSrc@src ! FakeSink@sink'
```

Only logs from source-like elements:

```bash
./build/leakflow --log-level debug --log-color never --log-filter 'element_kclass=Source' run 'FakeSrc ! FakeSink'
```

Only logs from pipeline-level mechanics:

```bash
./build/leakflow --log-level debug --log-color never --log-filter 'component=pipeline' run 'FakeSrc ! FakeSink'
```

Use multiple clauses. Clauses are ANDed:

```bash
./build/leakflow --log-level debug --log-color never --log-filter 'element=FakeSrc,element_name=src' run 'FakeSrc@src ! FakeSink@sink'
```

Filter Torch source logs:

```bash
./build/leakflow --log-level info --log-color never --log-filter 'element=TorchFileSrc' run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! FakeSink'
```

Filter converter logs:

```bash
./build/leakflow --log-level info --log-color never --log-filter 'element=TorchConvert' run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TorchConvert(dtype=float32,device=cpu) ! FakeSink'
```

Use environment variables for repeated debugging:

```bash
LEAKFLOW_LOG_LEVEL=debug \
LEAKFLOW_LOG_COLOR=never \
LEAKFLOW_LOG_FILTER='element_kclass=Source' \
./build/leakflow run 'FakeSrc ! FakeSink'
```

## Summaries

Disable summaries globally:

```bash
./build/leakflow --no-summaries run 'FakeSrc ! Summary'
```

Disable summaries with an environment variable:

```bash
LEAKFLOW_SUMMARIES=0 ./build/leakflow run 'FakeSrc ! Summary'
```

Force one Summary element to print anyway:

```bash
./build/leakflow --no-summaries run 'FakeSrc ! Summary(always_print=true)'
```

Use one silent Summary and one forced Summary:

```bash
./build/leakflow --no-summaries run 'FakeSrc ! Tee@t; @t.src_0 ! Summary; @t.src_1 ! Summary(always_print=true)'
```

Combine logs and summary suppression:

```bash
./build/leakflow --log-level debug --log-color never --no-summaries run 'FakeSrc ! Summary ! FakeSink'
```

## Payload Detail In Logs And `--graph`

The `info` "buffer flow" log line and the `--graph` link tooltips render a short
payload summary for the first buffer on each route. The detail of that summary is
controlled by `--summary-level N` (0-3, default 2), or `LEAKFLOW_SUMMARY_LEVEL`.
This is separate from the `Summary` element's own `level` property.

Lower levels are terser. For example, a tensor drops `strides`/`contiguous` below
level 2, and a `PlotAnnotationPayload` shows the count plus only the first
annotation until level 3:

```bash
./build/leakflow --log-level info --log-color never --summary-level 1 run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! FakeSink'
```

Level 2 (the default) adds per-type internals such as a tensor's
`strides`/`contiguous`. Level 3 additionally lists every plot annotation rather
than just the first:

```bash
./build/leakflow --log-level info --log-color never --summary-level 2 run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! FakeSink'
```

The environment variable is equivalent, with the CLI flag taking precedence:

```bash
LEAKFLOW_SUMMARY_LEVEL=1 ./build/leakflow --log-level info run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! FakeSink'
```

## Full Fixture Examples

Log HDF5 dataset loading only:

```bash
./build/leakflow --log-level info --log-color never --log-filter 'element=Hdf5FileSrc' run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! Summary ! FakeSink'
```

Log Torch conversion only:

```bash
./build/leakflow --log-level info --log-color never --log-filter 'element=TorchConvert' run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TorchConvert(dtype=float32,device=cpu) ! Summary ! FakeSink'
```

Log all source elements and suppress summaries:

```bash
./build/leakflow --log-level debug --log-color never --log-filter 'element_kclass=Source' --no-summaries run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! FakeSink'
```

Save a Torch tensor while logging sink events:

```bash
./build/leakflow --log-level info --log-color never --log-filter 'element_kclass=Sink' run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TorchFileSink(path=/tmp/traces_first_50_roundtrip.pt)'
```

## Invalid Filters

This fails because only `=` clauses are implemented:

```bash
./build/leakflow --log-filter 'level>=debug' run 'FakeSrc ! FakeSink'
```

This fails because `field.path` is reserved for future expansion:

```bash
./build/leakflow --log-filter 'field.path=demo.npy' run 'FakeSrc ! FakeSink'
```

## Safety Reminder

LeakFlow logs are for diagnostics. They should not print raw traces, key bytes,
plaintext arrays, tensor values, NumPy contents, or secret intermediates.
