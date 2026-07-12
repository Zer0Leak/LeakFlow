# Extras Plugins Context

Use this for pipeline elements that expose extras-library capabilities.

## Files

Public headers:

- `include/leakflow/plugins/extras`

Sources:

- `src/plugins/extras`

Tests:

- `tests/plugins/extras`

CLI/inspect files if affected:

- `src/apps/leakflow/leakflow_cli.cpp`
- `src/apps/leakflow_ls/main.cpp`
- `src/apps/common`
- `tests/apps`

## Target

- `leakflow_plugins_extras`

## Current Elements

- `Hdf5FileSrc`: reads one LeakFlow tensor-dataset HDF5 file in internal
  hyperslab batches, then emits one complete Torch payload per available named
  output.
- `FakeLiveHdf5Src`: replays aligned row batches from the same file as a finite,
  paced live source.
- `NumpySrc`: loads one `.npy` file through `leakflow_extras::load_npy(path)`
  and emits a `NumpyPayload`.
- `NumpyToTorch`: converts a `NumpyPayload` into a `TorchTensorPayload` through
  `leakflow_extras::convert_numpy_to_torch(...)`.
- `BufferFileSink` / `BufferFileSrc`: persist and reload a whole `Buffer` (caps +
  metadata + payload) to/from a single HDF5 file (the `leakflow.buffer` schema).
  Generic: they never know the concrete payload type. `property path` is the `.h5`
  file. Both take a `PayloadCodecRegistry` (injected by their factory in
  `leakflow_cli.cpp`, populated by the base/crypto codecs); the codec writes the
  payload body through an `Hdf5BufferArchiveWriter`/`Reader` (see
  `docs/context/modules/extras.md`), while the element writes the caps/metadata
  envelope as attributes. `any` caps, so they link to anything; the reloaded buffer
  carries its saved concrete caps and exact archived `payload.layout`. The source
  attaches the decoded payload before restoring archive metadata, so a saved
  semantic layout overrides the payload type's generic fallback. They live here — not in `leakflow_plugins_core` —
  because only this layer links both Torch and the HDF5 backend. Enables e.g. saving
  an expensive `PearsonCorrelator` correlation once and reloading it to re-select
  PoIs (`PoiSelect`) offline. Codecs: `TorchTensorPayload` (base),
  `CorrelationPayload` / `CorrelationPoiPayload` (crypto).

The descriptor catalog assembles the linked plugin descriptor and registers
matching element factories with `ElementFactoryRegistry`.

## HDF5 Source Pads And Metadata

Both HDF5 sources declare optional outputs:

| Pad | Payload | Logical HDF5 path |
|---|---|---|
| `traces` | `TorchTensorPayload` | `/traces` |
| `plaintexts` | `TorchTensorPayload` | `/plaintexts` |
| `keys` | `TorchTensorPayload` | `/keys` |
| `ciphertexts` | `TorchTensorPayload` | `/ciphertexts` |
| `countermeasures` | `TorchTensorBundlePayload` | `/countermeasures/**` |

The current countermeasure bundle entry is
`jitter.parameters.loop_iterations`. New countermeasure arrays become named
bundle entries instead of new source pads.

Every emitted buffer imports common capture/origin attributes from `/metadata`
and the applicable array's role-specific payload attributes, then stamps:

- `origin.file.format=hdf5`, `origin.file.path`, and `origin.file.size`,
- `origin.hdf5.dataset`,
- `origin.role`,
- `origin.row.begin`, `origin.row.count`, and `origin.row.total`,
- `tensor.axes` for ordinary tensors,
- `payload.countermeasure.tensors` for the countermeasure bundle,
- `payload.layout` for every payload.

Ordinary tensor layouts are `trace/sample`, `trace/byte`, and `key_byte`,
derived from each array's comma-separated storage `tensor.axes`. The current
bundle layout is
`jitter.parameters.loop_iterations=trace`.

Typical offline syntax:

```bash
leakflow run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! Summary ! FakeSink'
```

Typical aligned multi-input syntax:

```bash
leakflow run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); AesLeakage@leakage(byte_indexes=[0]); Summary@summary; @data.traces ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @summary'
```

Jitter files expose their nested labels as a tensor bundle:

```bash
leakflow run 'Hdf5FileSrc@data(path=traces/aes/jitter/aes_jitter_poi/key_01.h5); @data.countermeasures ! Summary ! FakeSink'
```

## Hdf5FileSrc Contract

Properties:

- `path`: required `.h5`/`.hdf5` path,
- `device=cpu`: target Torch device after CPU reads,
- `row_start=0`,
- `row_count=0`: all remaining rows,
- `io_batch_rows=256`: internal hyperslab size.

`io_batch_rows` controls storage I/O only. `Hdf5FileSrc` preallocates each final
tensor, fills it slice by slice, and emits only complete tensors. Its existing
`ProgressReported` events use aggregate logical uncompressed bytes across the
supported arrays: opening at `0`, chunk updates below `1`, and exact completion
at `1`. Each chunk update and final completion pass a cooperative lifecycle
checkpoint first; Stop reports `Cancelled` and emits no partial outputs. The
`storage_read` scope starts after file inspection and measures output
materialization, including storage reads and device transfer. Use the built-in
`process` duration for end-to-end HDF5/Zarr benchmark comparisons.

## FakeLiveHdf5Src Contract

`FakeLiveHdf5Src` shares `path`, `device`, `row_start`, `row_count`, and
`io_batch_rows`, and adds:

- `batch_size=1`: trace rows emitted together,
- `trace_rate=0`: traces per second; zero disables pacing.

It emits all available pads for the same row selection together, so every batch
shares vector-clock provenance and downstream joins use the default barrier.
The fixed `/keys` tensor accompanies each batch without being row-sliced. A
smaller final batch is allowed. The source reports trace-based determinate
progress after every emission and exact completion when the selected row count
is exhausted. It honors cooperative stop, resets on `start()`, is live, and is
not cache-replayable. Pause takes effect at a batch boundary (including pacing);
Stop during a batch read or pacing reports `Cancelled` and never commits or
emits that batch. Lifecycle teardown also closes an incomplete replay with a
terminal `Cancelled` report, including Stop while the source is blocked at a
framework queue boundary between callbacks.

```bash
leakflow run --graph 'FakeLiveHdf5Src@data(path=tests/fixtures/aes/sync/key_01.h5,batch_size=1,trace_rate=10.0); @data.traces ! TracePlot(title="AES live replay",update_mode=accumulate)'
```

## NumpySrc Contract

`NumpySrc`:

- has one required `path` property,
- declares source pad caps as `leakflow/numpy-array`,
- emits a `Buffer` with `leakflow/numpy-array` caps plus concrete runtime
  numeric caps parameters such as `dtype`, `device=cpu`, `rank`, and `shape`,
- attaches `NumpyPayload`,
- publishes the rank-derived generic `payload.layout` (`scalar` or
  `axis_0/axis_1/...`),
- stamps minimal provenance metadata:
  - `routing.element`,
  - `origin.file.path`,
  - `origin.file.size`.

Do not duplicate backend-specific NumPy facts such as word size, labels, or
memory order into caps or metadata. Those belong in `NumpyPayload` summaries or
backend-specific code.

Semantic facts such as `origin.role=traces`, `payload.crypto.algorithm=aes`, or capture details belong
to application annotations or future dataset-specific elements.

## NumpyToTorch Contract

`NumpyToTorch`:

- has one `dtype` property, default `preserve`,
- has one `device` property, default `cpu`,
- declares sink pad caps as `leakflow/numpy-array`,
- declares source pad caps as `leakflow/torch-tensor`,
- requires input buffers to carry `NumpyPayload`,
- emits a `Buffer` with `leakflow/torch-tensor` caps plus concrete runtime
  numeric caps parameters such as `dtype`, `device`, `rank`, and `shape`,
- attaches `TorchTensorPayload`,
- preserves incoming buffer metadata, including an exact semantic
  `payload.layout` override,
- stamps conversion metadata:
  - `payload.conversion.id=numpy-to-torch`,
  - `payload.conversion.element=<element instance name>`.

The conversion implementation id is `numpy-to-torch`.

## Future Zarr Direction

The next discussed file-format task is Zarr parity. `ZarrFileSrc` and its fake
live equivalent should reuse the format-neutral reader contract and preserve
the HDF5 sources' pad names, properties, row selections, Torch payloads,
logical-byte/trace-count progress, and `storage_read` profiling scope. Format
comparison should therefore measure the backend rather than different pipeline
semantics.

## Future Convert Direction

Generic `Convert`, conversion registry, and dynamic pads are now low-priority
future infrastructure. The explicit `NumpyToTorch` element remains the
near-term conversion path.

Design direction for future `Convert` when it is requested:

- follow the GStreamer-style autoplugging pattern,
- treat `Convert` as a smart wrapper element,
- select compatible ranked conversion implementations by caps, payload
  capability, and requested target,
- keep delegated converter pads internal to `Convert`,
- do not make the factory silently replace an exact requested element type.

Do not add generic `Convert` unless that phase is requested.

## Common Tests

```bash
ctest --test-dir build -R 'leakflow_plugins_extras|leakflow_cli|leakflow_ls'
```
