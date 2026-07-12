# Extras Module Context

Use this for application-callable file-format helpers above the base layer,
especially the format-neutral tensor-dataset reader, HDF5 loading, NumPy
payload/loading, and conversion helpers.

## Files

Public headers:

- `include/leakflow/extras`

Sources:

- `src/extras`

Tests:

- `tests/extras`
- generated temporary `.h5` fixtures inside HDF5 reader tests
- generated temporary `.npy` fixtures inside NumPy tests

## Target

- `leakflow_extras`

## Dependencies

`leakflow_extras` depends on:

- `leakflow_base`
- HDF5
- `cnpy++`

`leakflow_base` must not depend on `leakflow_extras`.

## Current API

- `TensorDatasetReader`: format-neutral interface for describing groups/arrays
  and reading one array into a preallocated Torch tensor.
- `TensorDatasetDescriptor`, `TensorArrayDescriptor`, and
  `TensorGroupDescriptor`: deterministic storage-independent schema views,
  including paths, dtypes, shapes, attributes, and row-alignment.
- `TensorReadOptions`: row selection plus `io_batch_rows` for internal I/O.
- `TensorReadProgress`: logical uncompressed bytes and rows read. Progress is
  reported after each hyperslab and ends exactly at completion. The progress
  callback returns `bool`: `false` requests cancellation, and `read_tensor`
  aborts promptly with `TensorReadCancelled` instead of returning a partial
  tensor. Source elements bridge this to `Element::cooperative_checkpoint()` so a
  long read honors pause/stop (see ARCHITECTURE_CONTRACTS.md, Long-Running Work).
- `Hdf5TensorDatasetReader`: recursively discovers HDF5 groups, arrays, and
  attributes; supports `uint8` and `float32`; fills a single preallocated CPU
  Torch tensor through batched hyperslab reads.
- `Hdf5BufferArchiveWriter` / `Hdf5BufferArchiveReader`: the HDF5 backend for
  whole-`Buffer` persistence (the `leakflow.buffer` schema), implementing the base
  `BufferArchiveWriter`/`Reader`. Writes the envelope as attributes (root: schema,
  `caps.type`, `payload.type`; `/caps` and `/metadata` attribute groups) and the
  payload body as native datasets plus scalar attributes under `/payload`. Supports
  the full payload dtype range (`float32/64`, `int8/16/32/64`, `uint8`); HDF5 is
  hidden behind a pimpl so the header stays HDF5-free. Drives
  `BufferFileSink`/`BufferFileSrc`. Payload tensors reload on CPU.
- `NumpyPayload`: thin wrapper around `cnpypp::NpyArray`; its layout is the
  rank-derived generic fallback (`scalar` or `axis_0/axis_1/...`).
- `load_npy(path)`: loads one `.npy` file into `NumpyPayload`.
- `NumpyPayload::caps()`: exposes concrete runtime caps metadata using the
  shared numeric caps vocabulary.

## Current NumPy Support

Supported:

- `.npy` only,
- numeric arrays,
- C-contiguous arrays,
- common bool, integer, unsigned integer, and floating dtypes,
- little-endian/common simple dtype metadata through `cnpy++`.

Concrete loaded buffers may expose standardized caps parameters:

- `dtype`,
- `device=cpu`,
- `rank`,
- `shape`.

Rejected:

- `.npz`,
- structured arrays,
- Fortran-order arrays,
- unsupported dtype shapes.

## LeakFlow Tensor-Dataset Schema

The AES HDF5 conversion uses these format-neutral logical paths:

```text
/traces                                      float32 [trace,sample]
/plaintexts                                  uint8   [trace,byte]
/keys                                        uint8   [key_byte]
/ciphertexts                                 uint8   [trace,byte]
/metadata                                    attributes
/countermeasures/jitter/parameters/
  loop_iterations                            uint8   [trace]  # jitter files only
```

Synchronized captures have no `/countermeasures` group. The ciphertext is
derived with AES-128 from the stored plaintext/key and carries explicit
derivation provenance. The jitter loop count is an exact label derived as
`plaintext[0] & 0x0f`; it is grouped under `countermeasures` so future masking,
shuffling, injected-noise, or region annotations can coexist without adding
top-level special cases.

The root carries `leakflow.schema=leakflow.sca.tensor-dataset` and
`leakflow.schema.version=1`. Array attributes include:

- `tensor.axes`,
- `leakflow.logical_sha256` and `leakflow.logical_nbytes`,
- `leakflow.row_aligned` (`false` for the fixed `/keys`, `true` for per-trace
  arrays, including the rank-1 jitter label),
- `origin.storage.layout`, `origin.storage.chunk_shape`,
  `origin.storage.compression`, `origin.storage.compression_level`,
  `origin.storage.shuffle`, and `origin.storage.checksum`.

Common capture, source, and converter facts are attributes on `/metadata`.
Role-specific payload facts stay with their arrays: `/traces` records
`payload.leakage.inverted=false`, `/keys` records
`payload.crypto.key.scope=fixed-per-file`,
`/ciphertexts` records its derived AES-128 provenance/derivation, and the jitter
parameter records its exact derivation provenance. Every array records its
actual `origin.storage.*` encoding.

Default conversion storage is gzip level 1 with byte shuffle and Fletcher32 for
chunked arrays, targeting about 1 MiB chunks; the tiny fixed key remains
contiguous and uncompressed.

## Future Zarr Parity

`TensorDatasetReader` is deliberately independent of HDF5. A future
`ZarrTensorDatasetReader` should expose the same logical paths, shapes,
attributes, row selections, and logical-byte progress. This gives HDF5 and Zarr
a fair comparison seam: identical source tensors/hashes and chunk shapes,
uncompressed-vs-uncompressed and matching gzip settings, plus a separate
format-native optimized comparison.

## Conversion Status

Direct application-callable NumPy-to-Torch conversion
(`convert_numpy_to_torch(...)`, Phase 19) is implemented, and the explicit
`NumpyToTorch` pipeline element (Phase 20) exposes it. Generic `Convert`, a
conversion registry, and dynamic pads remain deferred low-priority infrastructure.

## Common Tests

```bash
ctest --test-dir build -R leakflow_extras
```
