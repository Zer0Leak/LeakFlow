# Extras Module Context

Use this for application-callable helpers above the base layer, especially NumPy
payload/loading and future conversion helpers.

## Files

Public headers:

- `include/leakflow/extras`

Sources:

- `src/extras`

Tests:

- `tests/extras`
- generated temporary `.npy` fixtures inside NumPy tests

## Target

- `leakflow_extras`

## Dependencies

`leakflow_extras` depends on:

- `leakflow_base`
- `cnpy++`

`leakflow_base` must not depend on `leakflow_extras`.

## Current API

- `NumpyPayload`: thin wrapper around `cnpypp::NpyArray`.
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

## Next Likely Work

Phase 19 is expected to add direct application-callable NumPy-to-Torch
conversion using `NumpyPayload` and `TorchTensorPayload`.

Do not add a pipeline `Convert` element unless Phase 20 is requested.

## Common Tests

```bash
ctest --test-dir build -R leakflow_extras
```
