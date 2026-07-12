# Module Map

This file maps repository areas to targets, tests, and context files.

## Dependency Direction

```text
leakflow_log
  -> leakflow_core
  -> leakflow_render
  -> leakflow_plugins_core

leakflow_core
  -> leakflow_base
  -> leakflow_plugins_base

leakflow_core
  -> leakflow_base
  -> leakflow_crypto
  -> leakflow_plugins_crypto

leakflow_core
  -> leakflow_base
  -> leakflow_extras (HDF5 + NumPy)
  -> leakflow_plugins_extras

leakflow_log + leakflow_core + leakflow_base
  -> leakflow_plot
  -> leakflow_plugins_plot

leakflow_plugins_core + leakflow_plugins_base + leakflow_plugins_extras
  + leakflow_plugins_crypto
  -> leakflow_cli
  -> leakflow

leakflow_plugins_plot
  -> leakflow_cli
  -> leakflow

leakflow_plugins_crypto + leakflow_plot
  -> leakflow_plugins_crypto_plot   (ScorePlot: AttackStatsPayload -> ScoreView)
  -> leakflow_cli
  -> leakflow

leakflow_render + plugin descriptor catalogs
  -> leakflow-ls
```

`leakflow_log` is the lowest infrastructure layer. `leakflow_core` must remain
free of Torch, NumPy, HDF5, AES, Kyber, plotting, GUI, YAML, and dynamic plugin
loading.

`ElementFactoryRegistry` lives in `leakflow_core` because it only stores
abstract descriptors and element factory callbacks. Built-in plugin factory
registration remains explicit in each linked plugin catalog. The current
expression-building helper lives in `leakflow_cli`, which links the built-in
plugins for `leakflow run` and tutorial applications.

`PayloadCodecRegistry` also lives in `leakflow_core` (callbacks only; it
forward-declares the base `BufferArchiveWriter`/`Reader`, so core stays
Torch/HDF5-free). The codecs are registered by the base/crypto plugins and the
HDF5-backed `BufferFileSink`/`BufferFileSrc` elements live in
`leakflow_plugins_extras` — the layer that links both Torch and HDF5. See the
Buffer Persistence contract in `ARCHITECTURE_CONTRACTS.md`.

## Areas

| Area | Target | Public headers | Sources | Tests | Context |
|---|---|---|---|---|---|
| Log | `leakflow_log` | `include/leakflow/log` | `src/log` | `tests/log` | `docs/context/modules/log.md` |
| Core | `leakflow_core` | `include/leakflow/core` | `src/core` | `tests/core` | `docs/context/modules/core.md` |
| Render | `leakflow_render` | `include/leakflow/render` | `src/render` | `tests/render` | `docs/context/modules/core.md` |
| Base | `leakflow_base` | `include/leakflow/base` | `src/base` | `tests/base` | `docs/context/modules/base.md` |
| Crypto | `leakflow_crypto` | `include/leakflow/crypto` | `src/crypto` | `tests/crypto` | `docs/context/modules/crypto.md` |
| Extras | `leakflow_extras` | `include/leakflow/extras` | `src/extras` | `tests/extras` | `docs/context/modules/extras.md` |
| Core plugins | `leakflow_plugins_core` | `include/leakflow/plugins/core` | `src/plugins/core` | `tests/plugins/core` | `docs/context/modules/plugins-core.md` |
| Base plugins | `leakflow_plugins_base` | `include/leakflow/plugins/base` | `src/plugins/base` | `tests/plugins/base` | `docs/context/modules/plugins-base.md` |
| Extras plugins | `leakflow_plugins_extras` | `include/leakflow/plugins/extras` | `src/plugins/extras` | `tests/plugins/extras` | `docs/context/modules/plugins-extras.md` |
| Crypto plugins | `leakflow_plugins_crypto` | `include/leakflow/plugins/crypto` | `src/plugins/crypto` | `tests/plugins/crypto` | `docs/context/modules/plugins-crypto.md` |
| Plot / graph UI | `leakflow_plot` | `include/leakflow/plot` | `src/plot` | focused non-visual tests only | `docs/context/modules/plot.md` |
| Plot plugins | `leakflow_plugins_plot` | `include/leakflow/plugins/plot` | `src/plugins/plot` | `tests/plugins/plot` plus manual GUI checks | `docs/context/modules/plot.md` |
| Crypto plot plugins | `leakflow_plugins_crypto_plot` | `include/leakflow/plugins/crypto_plot` | `src/plugins/crypto_plot` | `tests/plugins/crypto_plot` plus manual GUI checks | `docs/context/modules/plot.md` |
| Apps / CLI | `leakflow`, `leakflow-ls`, `leakflow_cli` | `src/apps/leakflow/leakflow_cli.hpp` | `src/apps/{common,leakflow,leakflow_ls,cuda_smoke}` | `tests/apps` | relevant plugin/module context |
| Torch debug | `leakflow_torch_debug` | `include/leakflow/debug` | `src/debug` | none (dev-only) | this row + `.vscode/launch.json`, `torch_lldb.py` |
| GUI | none yet | none yet | none yet | none yet | `docs/context/modules/gui.md` |

`leakflow_torch_debug` is a **dev-only** helper, not part of the runtime
architecture: an `OBJECT` library (gated by `-DLEAKFLOW_TORCH_DEBUG`, default ON)
providing C-linkage tensor-inspection entry points (`dtv`/`ptv`/`ps`/`pt`) that
LLDB/GDB calls. It links into `leakflow`; any debuggable target may opt in.

## Source Reading Patterns

For a narrow change, read the module context file first, then:

- the public header being changed,
- its matching `.cpp`,
- the closest test file,
- the relevant CMake file only if a target/source/test changes.

Avoid loading unrelated plugins, long design docs, generated reports, or binary
fixtures unless the task directly needs them.

## CMake Ownership

- Root `CMakeLists.txt` owns project setup, top-level options, `fmt`, and
  `add_subdirectory`.
- Directory-level `src/*/CMakeLists.txt` files own production targets.
- Directory-level `tests/*/CMakeLists.txt` files own test targets and CTest
  registration.

Keep target configuration target-based.

## Test Groups

- `tests/core`: core framework primitives.
- `tests/log`: logging configuration, filtering, and sink behavior.
- `tests/render`: terminal/summary rendering.
- `tests/base`: Torch tensor payloads.
- `tests/crypto`: Hamming weight/distance and AES leakage helpers.
- `tests/extras`: HDF5 tensor-dataset reading and NumPy payload/loading.
- `tests/apps`: CLI parser, expression-builder helper, CLI runner, and inspect
  tool behavior.
- `tests/plugins/core`: generic core plugin elements (incl. `Sync`, threaded).
- `tests/plugins/base`: Torch-backed base plugin elements (incl. `FakeLiveSrc`).
- `tests/plugins/extras`: HDF5 and NumPy extras plugin elements.
- `tests/plugins/crypto`: AES leakage, Pearson PoI, annotation conversion, and the
  Phase 26 AES PoI numeric correctness test.
- Plot GUI/window behavior is manual-only; `tests/plugins/plot` covers
  non-visual snapshot, graph-runtime, and descriptor behavior.
- `tests/fixtures/aes`: deterministic AES `.h5` dataset fixture plus focused
  legacy `.pt` fixtures.
