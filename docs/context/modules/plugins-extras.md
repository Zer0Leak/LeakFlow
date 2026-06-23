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

- `NumpySrc`: loads one `.npy` file through `leakflow_extras::load_npy(path)`
  and emits a `NumpyPayload`.
- `NumpyToTorch`: converts a `NumpyPayload` into a `TorchTensorPayload` through
  `leakflow_extras::convert_numpy_to_torch(...)`.

The descriptor catalog assembles the linked plugin descriptor and registers
matching element factories with `ElementFactoryRegistry`.

## NumpySrc Contract

`NumpySrc`:

- has one required `path` property,
- declares source pad caps as `leakflow/numpy-array`,
- emits a `Buffer` with `leakflow/numpy-array` caps plus concrete runtime
  numeric caps parameters such as `dtype`, `device=cpu`, `rank`, and `shape`,
- attaches `NumpyPayload`,
- stamps minimal provenance metadata:
  - `element`,
  - `file.path`,
  - `file.size`.

Do not duplicate backend-specific NumPy facts such as word size, labels, or
memory order into caps or metadata. Those belong in `NumpyPayload` summaries or
backend-specific code.

Semantic facts such as `role=traces`, `algorithm=aes`, or capture details belong
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
- preserves incoming buffer metadata,
- stamps conversion metadata:
  - `conversion.id=numpy-to-torch`,
  - `conversion.element=<element instance name>`.

The conversion implementation id is `numpy-to-torch`.

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
