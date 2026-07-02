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

## Conversion Status

Direct application-callable NumPy-to-Torch conversion
(`convert_numpy_to_torch(...)`, Phase 19) is implemented, and the explicit
`NumpyToTorch` pipeline element (Phase 20) exposes it. Generic `Convert`, a
conversion registry, and dynamic pads remain deferred low-priority infrastructure.

## Common Tests

```bash
ctest --test-dir build -R leakflow_extras
```
