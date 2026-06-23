# Base Plugins Context

Use this for pipeline elements that expose base Torch-backed capabilities.

## Files

Public headers:

- `include/leakflow/plugins/base`

Sources:

- `src/plugins/base`

Tests:

- `tests/plugins/base`

CLI/inspect files if affected:

- `src/apps/leakflow/leakflow_cli.cpp`
- `src/apps/leakflow_ls/main.cpp`
- `src/apps/common`
- `tests/apps`

## Target

- `leakflow_plugins_base`

## Current Elements

- `TorchFileSrc`: loads one Torch tensor `.pt` file into a `TorchTensorPayload`.
- `TorchConvert`: explicitly converts one `TorchTensorPayload` to a requested
  Torch dtype and/or device.
- `TorchFileSink`: saves one `TorchTensorPayload` as a Torch tensor `.pt` file.

Element-specific descriptor data, including pads, properties, stamped metadata,
and suggested user metadata, lives in each element source file through
`ElementType::descriptor()`. The descriptor catalog assembles the linked plugin
descriptor and registers matching element factories with
`ElementFactoryRegistry`.

## TorchFileSrc Contract

`TorchFileSrc`:

- has one required `path` property,
- has one `device` property, default `cpu`,
- accepts `device=preserve` to ask LibTorch to preserve the serialized device,
- has one `invert` property, default `false`, that multiplies the loaded tensor
  by `-1` when true,
- declares source pad caps as `leakflow/torch-tensor`,
- emits a `Buffer` with concrete runtime numeric caps parameters such as
  `dtype`, `device`, `rank`, and `shape`,
- attaches `TorchTensorPayload`,
- stamps minimal provenance metadata:
  - `origin.file.format=torch-tensor`,
  - `origin.file.path`,
  - `origin.file.size`.
- advertises typed metadata descriptors for metadata it stamps.
- advertises typed suggested metadata examples such as:
  - `capture.sample_rate_hz=29454545.454545453`,
  - `capture.source=ChipWhisperer`,
  - `payload.leakage.range=[-0.5,0.5]`,
  - `capture.dataset.name=aes_sync_poi`,
  - `payload.leakage.inverted=false`.

`TorchFileSrc` does not stamp the element instance name into data metadata. The
buffer-flow log already carries the element route.

`invert=true` multiplies the loaded tensor by `-1`. `TorchFileSrc` does not stamp
`payload.leakage.inverted`; the user declares the leakage inversion state through
metadata based on the original data and the chosen `invert` value.
`capture.sample_rate_hz` is suggested user metadata rather than a `TorchFileSrc`
property; pipelines should set it explicitly through source metadata when needed.

## TorchFileSink Contract

`TorchFileSink`:

- has one required `path` property,
- declares sink pad caps as `leakflow/torch-tensor`,
- requires input buffers to carry `TorchTensorPayload`,
- saves the tensor with LibTorch in a Python `torch.load`-compatible tensor
  archive,
- consumes the buffer.

## TorchConvert Contract

`TorchConvert`:

- declares sink and source pad caps as `leakflow/torch-tensor`,
- has a `dtype` property, default `preserve`,
- has a `device` property, default `preserve`,
- emits a new `TorchTensorPayload` with updated concrete caps,
- preserves input metadata,
- stamps `conversion.id=torch-convert` and `conversion.element`,
- remains explicit and does not imply generic `Convert` or a conversion
  registry.

## Future TorchScript Elements

Torch tensor files and TorchScript module archives are separate artifacts.
Keep them as separate elements so raw SCA tensors are not confused with
executable model artifacts.

Future `TorchScriptSrc`:

- should load a TorchScript module archive with LibTorch/JIT,
- should emit a future module/model payload type, not `TorchTensorPayload`,
- should declare distinct caps such as `leakflow/torchscript-module`,
- should not be a fallback mode of `TorchFileSrc`.

Future `TorchScriptSink`:

- should save a future module/model payload type as a TorchScript archive,
- should declare distinct caps such as `leakflow/torchscript-module`,
- should be loaded in Python with `torch.jit.load(...)`,
- should not be a fallback mode of `TorchFileSink`.

## Out Of Scope Unless Requested

- Torch tensor bundle file I/O.
- TorchScript module I/O, including future `TorchScriptSrc` and
  `TorchScriptSink`.
- Raw byte serialization elements.
- Compression or archive elements.
- Dynamic plugin loading.
- Conversion registry.
- AES/Kyber/statistics.

## Common Tests

```bash
ctest --test-dir build -R 'leakflow_plugins_base|leakflow_cli|leakflow_ls'
```
