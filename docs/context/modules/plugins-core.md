# Core Plugins Context

Use this for work touching generic linked pipeline elements.

## Files

Public headers:

- `include/leakflow/plugins/core`

Sources:

- `src/plugins/core`

Tests:

- `tests/plugins/core`

CLI/inspect files if affected:

- `src/apps/leakflow/leakflow_cli.cpp`
- `src/apps/leakflow/main.cpp`
- `src/apps/leakflow_ls/main.cpp`
- `src/apps/common`
- `tests/apps`

## Target

- `leakflow_plugins_core`

## Current Elements

- `FileSrc`: reads raw bytes into `BytesPayload`.
- `FileSink`: writes `BytesPayload`, with summary fallback for non-byte buffers.
- `FakeSrc`: produces simple test buffers.
- `FakeSink`: consumes buffers for validation.
- `Summary`: renders structured buffer summaries and forwards buffers.
- `Tee`: forks one input buffer to requested `src_%u` branches by copying the
  buffer envelope and sharing the payload pointer. A fresh `Tee` has only its
  always-available `sink` pad; `src_N` output pads are created on request from
  the `src_%u` pad template. Tee pad templates use `ANY` caps.
- `Queue`: synchronous buffer storage/forwarding.

## Descriptor Catalog

The linked plugin descriptor is named:

```text
leakflow_plugins_core
```

Current author/license metadata:

```text
Zer0Leak <edgard.lima@gmail.com>
Apache-2.0
```

Keep descriptors in sync with runtime element properties and pad declarations.
Element-specific descriptor data lives in each element source file through
`ElementType::descriptor()`. The descriptor catalog only assembles the linked
plugin descriptor and registers matching element factories with
`ElementFactoryRegistry`.

## Contracts

Core plugin elements are generic. They must not know AES, Kyber, Torch tensor
semantics, NumPy dtype semantics, GUI, plotting, or hardware capture.

`FileSrc` and `FileSink` are raw-byte elements only.

`Tee` must avoid deep-copying large payloads.

## Common Tests

```bash
ctest --test-dir build -R 'leakflow_plugins_core|leakflow_tee|leakflow_cli|leakflow_ls'
```
