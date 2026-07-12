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
- `FakeLiveSrc`: live source that reads a Torch `.pt` and streams one `Buffer` per
  axis-0 row (`[1, M]`), then EOS. Declares itself live; `trace_rate` paces it
  (one trace per `1/rate` s); honors the cooperative stop token. Drives the live
  phase in tests/CLI without hardware. It does not stamp `capture.sample_rate_hz`;
  add that as explicit user metadata when the trace sampling rate matters. Each
  emitted row carries the rank-derived generic `payload.layout`.
- `AppSrc`: application-fed live source. Instead of reading a file or device, the
  application supplies frames — via a producer callback (pull mode) or `push_frame`
  (push mode); each frame is an aligned set of buffers routed to `src_0..src_{n-1}`
  and emitted in one firing (shared vector clock, like `Sync`). Declares itself
  live. It is the generic seam for feeding external data (folders of `.h5`
  captures, hardware, generated frames) into a pipeline without a per-dataset
  source element. See the AppSrc Contract below.

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
- publishes the rank-derived generic `payload.layout` (`scalar` or
  `axis_0/axis_1/...`),
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
- preserves input metadata, including an exact semantic `payload.layout`
  override such as `trace/sample`,
- stamps `conversion.id=torch-convert` and `conversion.element`,
- remains explicit and does not imply generic `Convert` or a conversion
  registry.

## AppSrc Contract

`AppSrc` is the generic application-fed source. Two things it deliberately does
**not** own — progress semantics and per-instance configuration — the framework
exposes so the *application* driving it supplies them, elegantly and generically:

Application-provided buffers must already satisfy the core payload-layout
contract. Calling `Buffer::set_payload(...)` does this automatically. `AppSrc`
forwards the buffer as supplied and neither infers nor overwrites
`payload.layout`.

- **Pull frame producer**: `set_frame_producer(FrameProducer)`, where
  `FrameProducer = std::optional<std::vector<Buffer>>(std::size_t index, const
  ProgressReport& report)`. AppSrc calls it on the worker thread for frame `index`
  (`nullopt` = end of stream), owns the index, and rewinds to 0 in `start()`, so a
  Stop → Start re-streams. Push mode: `push_frame` + `end_of_stream` from a
  producer thread.
- **App-driven progress**: the producer receives a `report(fraction, message,
  index, total)` reporter that AppSrc binds to its own protected
  `Element::report_progress`, so the app drives a determinate `--graph` node bar
  per instance without touching the progress bus. AppSrc can't know the frame total
  (the producer decides EOS), so the app reports it. Push-mode drivers use the
  public `AppSrc::report_progress` passthrough (thread-safe). `fraction` 1.0
  auto-promotes to `Completed`. A new app's total cost is the `report` parameter
  plus one call.
- **App-exposed instance properties**: the app enriches its AppSrc instance with
  `Element::add_property(...)`, and the `--graph` control panel renders them (it
  draws the element's live `property_specs()`; see `ARCHITECTURE_CONTRACTS.md`,
  Properties). Mark a knob `PropertyEffectKind::Lifecycle` to make it editable only
  when Stopped.

`replications/rezaeezade_2025_blindfold/poi_finder_main.cpp` is the worked example:
it streams one `key_*.h5` trace bundle per frame, reports per-bundle progress, and
exposes `path` and `max_trace_bundles` as stopped-only lifecycle properties on its
`src` node. Tests: `tests/plugins/base/app_src_test.cpp` (frame producer, restart,
and the progress reporter → sink → `Completed`).

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
