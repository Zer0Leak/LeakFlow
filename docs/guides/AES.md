# AES Leakage Model Guide

These examples use the checked-in AES HDF5 fixture
`tests/fixtures/aes/sync/key_01.h5` and run from the repository root.

## Build First

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
```

## Fixtures

The tiny fixture set is intentionally small enough for CTest and CLI smoke
commands:

```text
tests/fixtures/aes/sync/key_01.h5
  /traces       float32 [50,5000]
  /plaintexts   uint8   [50,16]
  /keys         uint8   [16]
  /ciphertexts  uint8   [50,16]
```

`Hdf5FileSrc` exposes each present array through a named Torch output pad. One
source therefore keeps the trace, plaintext, key, and ciphertext tensors under
the same dataset provenance. Only the pads connected by the pipeline need to be
used.

In `--graph` runs, `Hdf5FileSrc` updates the element progress bar while it fills
the final tensors through internal HDF5 hyperslabs; downstream receives the
complete tensors only after loading reaches 100%. `FakeLiveHdf5Src` instead
updates progress after each emitted batch because it knows the selected trace
count in advance.

`AesLeakage` computes AES Hamming-weight leakage targets. For each selected
state byte, the output payload is a Torch `uint8` tensor with shape `[B,N,C]`:

```text
B = selected byte indexes
N = traces/plaintexts
C = selected leakage channels
axis 2 follows the requested channels order
```

The `channels` property chooses any non-empty combination of `HW(m)`,
`HW(m xor k)`, and `HW(y)`, where `y = AES_SBOX[m XOR k]`. The default is
`channels=[HW(y)]`. Use commas between properties, for example
`channels=[HW(m)],byte_indexes=[0]`.

The output buffer metadata includes:

```text
leakage.model=aes-first-round
leakage.byte_indexes=[...]
leakage.channels=HW(y)
crypto.algorithm=AES
crypto.state_bytes=16
trace.count=N
```

The `traces` and `plaintexts` sink pads are required. The `keys` sink pad is
optional in the descriptor, but runtime computation requires it for
`HW(m xor k)` and `HW(y)`.

## Inspect The Element

```bash
./build/leakflow-ls --no-colors --ascii AesLeakage
```

You should see required `traces` and `plaintexts` sink pads and an optional
`keys` sink pad:

```text
traces
plaintexts
keys
```

## One Byte Leakage

This computes the default `HW(y)` leakage for AES state byte `0`.

```bash
./build/leakflow run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); AesLeakage@leakage(byte_indexes=[0]); Summary@summary(level=3); @data.traces ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @summary'
```

Expected summary facts include:

```text
leakage.model=aes-first-round
leakage.byte_indexes=[0]
leakage.channels=HW(y)
trace.count=50
trace.input=connected
```

## Selected Bytes With Trace Alignment

Connect the trace fixture when you want the leakage model to verify that the
trace matrix has the same `N` as the plaintexts.

```bash
./build/leakflow run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); AesLeakage@leakage(byte_indexes=[0,1,2,3]); Summary@summary(level=3); @data.traces ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @summary'
```

Expected summary facts include:

```text
leakage.byte_indexes=[0,1,2,3]
trace.count=50
trace.input=connected
```

## Multiple Channels

Set `channels` to emit more than one target channel per byte. This command emits
`HW(m)` and `HW(y)` for AES byte `0`, so the output shape is `[1,50,2]`.

```bash
./build/leakflow run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); AesLeakage@leakage(byte_indexes=[0],channels=[HW(m),HW(y)]); Summary@summary(level=3); @data.traces ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @summary'
```

Expected summary facts include:

```text
leakage.byte_indexes=[0]
leakage.channels=HW(m),HW(y)
```

## All AES State Bytes

Omit `byte_indexes` to compute all 16 AES state bytes. With the default
`channels=[HW(y)]`, the output shape is `[16,50,1]` for the checked-in fixture.

```bash
./build/leakflow run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); AesLeakage@leakage; Summary@summary(level=3); @data.traces ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @summary'
```

## Save Leakage Tensor

Use `TorchFileSink` to write the `[B,N,C]` leakage tensor as a `.pt` file.

```bash
./build/leakflow run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); AesLeakage@leakage(byte_indexes=[0,1]); TorchFileSink@sink(path=/tmp/aes_leakage_bytes_0_1.pt); @data.traces ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @sink'
```

Python can load the result with:

```python
import torch

leakage = torch.load("/tmp/aes_leakage_bytes_0_1.pt")
print(leakage.shape)  # torch.Size([2, 50, 1])
```

## Pearson PoI Annotations For TracePlot

`PearsonCorrelator` joins the trace branch and the AES leakage branch, then
`PoiSelect` picks the top-k PoIs (the split of the old `PearsonPoiFinder`):

```text
traces [N,M]                 -> PearsonCorrelator.features
AES leakage [B,N,C]          -> PearsonCorrelator.targets
correlation                  -> PoiSelect.correlation
Pearson PoIs                 -> CorrelationPoiToPlotAnnotations
plot annotations             -> TracePlot.annotations
trace branch copy            -> TracePlot.sink
```

For AES leakage output `[B,N,C]`, flattened PoI target metadata follows byte
index first, then channel order:

```text
byte_indexes[0].HW(y)
byte_indexes[1].HW(y)
...
```

`PearsonCorrelator` (and `PoiSelect` downstream) forwards the AES byte-index
metadata and stamps expanded target metadata such as:

```text
leakage.byte_indexes=[3,5]
poi.target.0.label=byte_3.HW(y)
poi.target.1.label=byte_5.HW(y)
poi.target.1.byte_index=5
poi.target.1.channel=HW(y)
```

`CorrelationPoiToPlotAnnotations` uses those metadata labels to create generic
plot annotations. `TracePlot` draws one vertical marker per selected sample
index; if several targets choose the same sample index, their annotations stay
grouped at that marker.

This command plots traces and overlays per-target Pearson PoIs for AES bytes `3`
and `5`. It opens the interactive TracePlot window after pipeline execution.

```bash
./build/leakflow run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); Tee@traces; AesLeakage@leakage(byte_indexes=[3,5]); PearsonCorrelator@corr; PoiSelect@poi(top_k=[10],rank_by=[abs]); CorrelationPoiToPlotAnnotations@ann(precision=3); TracePlot@plot(title="AES bytes 3 and 5 PoIs",group=aes,label=traces,x_axis=sample); @data.traces ! @traces; @traces.src_0 ! @corr.features; @traces.src_1 ! @plot.sink; @traces.src_2 ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @corr.targets; @corr ! @poi; @poi ! @ann ! @plot.annotations'
```

For a non-GUI smoke check, stop at `Summary` after the annotation converter:

```bash
./build/leakflow run \
  'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); Tee@traces; AesLeakage@leakage(byte_indexes=[3,5]); PearsonCorrelator@corr; PoiSelect@poi(top_k=[3],rank_by=[abs]); CorrelationPoiToPlotAnnotations@ann(precision=3); Summary@summary(level=3); @data.traces ! @traces; @traces.src_0 ! @corr.features; @traces.src_1 ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @corr.targets; @corr ! @poi; @poi ! @ann ! @summary'
```

Expected summary facts include:

```text
leakflow/plot-annotations
leakage.byte_indexes=[3,5]
poi.target.0.label=byte_3.HW(y)
poi.target.1.label=byte_5.HW(y)
annotation.count=3
```
