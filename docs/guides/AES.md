# AES Leakage Model Guide

These examples use the checked-in AES Torch tensor fixtures under
`tests/fixtures/aes/sync/key_01/` and run from the repository root.

## Build First

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
```

## Fixtures

The tiny fixture set is intentionally small enough for CTest and CLI smoke
commands:

```text
tests/fixtures/aes/sync/key_01/traces_first_50.pt       float32 [50,5000]
tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt  uint8   [50,16]
tests/fixtures/aes/sync/key_01/key_first_50.pt          uint8   [16]
```

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
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); AesLeakage@leakage(byte_indexes=[0]); Summary@summary(level=3); @traces_src ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @summary'
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
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); AesLeakage@leakage(byte_indexes=[0,1,2,3]); Summary@summary(level=3); @traces_src ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @summary'
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
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); AesLeakage@leakage(byte_indexes=[0],channels=[HW(m),HW(y)]); Summary@summary(level=3); @traces_src ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @summary'
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
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); AesLeakage@leakage; Summary@summary(level=3); @traces_src ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @summary'
```

## Save Leakage Tensor

Use `TorchFileSink` to write the `[B,N,C]` leakage tensor as a `.pt` file.

```bash
./build/leakflow run \
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); AesLeakage@leakage(byte_indexes=[0,1]); TorchFileSink@sink(path=/tmp/aes_leakage_bytes_0_1.pt); @traces_src ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @sink'
```

Python can load the result with:

```python
import torch

leakage = torch.load("/tmp/aes_leakage_bytes_0_1.pt")
print(leakage.shape)  # torch.Size([2, 50, 1])
```

## Pearson PoI Annotations For TracePlot

`PearsonPoiFinder` joins the trace branch and the AES leakage branch:

```text
traces [N,M]                 -> PearsonPoiFinder.features
AES leakage [B,N,C]          -> PearsonPoiFinder.targets
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

`PearsonPoiFinder` forwards the AES byte-index metadata and stamps expanded
target metadata such as:

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
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); Tee@traces; TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); AesLeakage@leakage(byte_indexes=[3,5]); PearsonPoiFinder@poi(top_k=[10],rank_by=[abs]); CorrelationPoiToPlotAnnotations@ann(precision=3); TracePlot@plot(title="AES bytes 3 and 5 PoIs",group=aes,label=traces,x_axis=sample); @traces_src ! @traces; @traces.src_0 ! @poi.features; @traces.src_1 ! @plot.sink; @traces.src_2 ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @poi.targets; @poi ! @ann ! @plot.annotations'
```

For a non-GUI smoke check, stop at `Summary` after the annotation converter:

```bash
./build/leakflow run \
  'TorchFileSrc@traces_src(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt); Tee@traces; TorchFileSrc@plain_src(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt); TorchFileSrc@key_src(path=tests/fixtures/aes/sync/key_01/key_first_50.pt); AesLeakage@leakage(byte_indexes=[3,5]); PearsonPoiFinder@poi(top_k=[3],rank_by=[abs]); CorrelationPoiToPlotAnnotations@ann(precision=3); Summary@summary(level=3); @traces_src ! @traces; @traces.src_0 ! @poi.features; @traces.src_1 ! @leakage.traces; @plain_src ! @leakage.plaintexts; @key_src ! @leakage.keys; @leakage ! @poi.targets; @poi ! @ann ! @summary'
```

Expected summary facts include:

```text
leakflow/plot-annotations
leakage.byte_indexes=[3,5]
poi.target.0.label=byte_3.HW(y)
poi.target.1.label=byte_5.HW(y)
annotation.count=3
```
