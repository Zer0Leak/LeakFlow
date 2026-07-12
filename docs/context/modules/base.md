# Base Module Context

Use this for work touching the Torch-backed numerical layer.

## Files

Public headers:

- `include/leakflow/base`

Sources:

- `src/base`

Tests:

- `tests/base`

## Target

- `leakflow_base`

## Dependencies

`leakflow_base` depends on:

- `leakflow_core`
- LibTorch

`leakflow_core` must not depend on `leakflow_base`.

## Current API

- Numeric caps helpers for common array/tensor caps keys:
  - `dtype`,
  - `device`,
  - `rank`,
  - `shape`.
- `TorchTensorPayload`: wraps one dense strided `torch::Tensor`.
- `TorchTensorBundlePayload`: deterministic named map of
  `shared_ptr<TorchTensorPayload>`.
- `buffer_archive.hpp`: abstract `BufferArchiveWriter`/`BufferArchiveReader` — a
  storage-neutral, Torch-aware (but HDF5-agnostic) sink/source for a `Buffer`'s
  payload body (`write_tensor`/`write_int`/`write_string` + the read mirror).
  Payload codecs (base/crypto) serialize through it; the concrete backend (HDF5
  today via `leakflow_extras`, Zarr later) implements it. It lives here so codecs
  never depend on the storage backend. `leakflow_core`'s `PayloadCodec` names these
  types only by forward declaration.
- `statistics.hpp`: `pearson_correlation` and the stateful
  `InteractivePearsonCorrelation` accumulator.
Note: GMM/EM (`gaussian_mixture.hpp`), entropic-OT Sinkhorn (`sinkhorn.hpp`), and
clustering evaluation (`clustering_metrics.hpp`) now live in the **`leakflow_ml`**
library — see `docs/context/modules/ml.md`.

## Contracts

`TorchTensorPayload`:

- accepts CPU or CUDA tensors,
- rejects undefined tensors,
- rejects non-strided tensors,
- allows non-contiguous strided tensors.
- exposes Torch-native accessors for computation,
- exposes canonical caps metadata for pipeline contracts.

`TorchTensorBundlePayload`:

- rejects empty names,
- rejects null payloads,
- stores names deterministically through `std::map`,
- preserves shared payload identity.

Payload uniqueness does not imply unique Torch storage. Torch views and handles
may share storage even when a `Payload` object is uniquely owned.

## Out Of Scope Unless Requested

- NumPy conversion.
- Torch file I/O in `leakflow_base`; pipeline file I/O belongs in
  `leakflow_plugins_base`.
- Dataset-specific helpers.
- AES/Kyber/statistics.
- DNN training/inference.

## Common Tests

```bash
ctest --test-dir build -R leakflow_base
```
