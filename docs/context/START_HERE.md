# LeakFlow Context Start

Use this file as the first project-specific context entry after `AGENTS.md`.
It exists to keep Codex startup context small.

## Read Order

For most tasks, read only:

1. `AGENTS.md`
2. `ROADMAP.md`
3. `docs/context/START_HERE.md`
4. `docs/context/CURRENT_STATE.md`
5. `docs/context/ACTIVE_PHASE.md`
6. `docs/context/MODULE_MAP.md`
7. `docs/context/ARCHITECTURE_CONTRACTS.md`
8. The relevant file under `docs/context/modules/`

Read source files only after the requested module or phase is clear.

## Source Inspection Rule

Do not recursively read all of `include/`, `src/`, `tests/`, and `docs/design/`
as default startup context.

Instead:

- For core work, read `docs/context/modules/core.md`, then inspect only
  `include/leakflow/core`, `src/core`, and `tests/core`.
- For logging work, read `docs/context/modules/log.md`, then inspect only
  `include/leakflow/log`, `src/log`, `tests/log`, and CLI/element files when
  user-facing logging behavior or element logging is affected.
- For base Torch payload work, read `docs/context/modules/base.md`, then inspect
  only `include/leakflow/base`, `src/base`, and `tests/base`.
- For crypto helper work, read `docs/context/modules/crypto.md`, then inspect
  only `include/leakflow/crypto`, `src/crypto`, and `tests/crypto`.
- For base Torch plugin work, read `docs/context/modules/plugins-base.md`, then
  inspect only `include/leakflow/plugins/base`, `src/plugins/base`,
  `tests/plugins/base`, and CLI files if command syntax is affected.
- For extras HDF5/NumPy/conversion work, read `docs/context/modules/extras.md`, then
  inspect only `include/leakflow/extras`, `src/extras`, `tests/extras`, and the
  relevant extras plugin files if pipeline exposure is requested.
- For ML/clustering/GMM/evaluation work, read `docs/context/modules/ml.md`, then
  inspect only `include/leakflow/ml`, `src/ml`, `tests/ml`, and the matching
  `include/leakflow/plugins/ml`, `src/plugins/ml`, and `tests/plugins/ml` files
  when pipeline exposure is affected. For clustering metric plots, also read the
  plot module context and inspect only the planned ML→plot bridge plus the exact
  generic views it uses.
- For core plugin work, read `docs/context/modules/plugins-core.md`, then inspect
  only `include/leakflow/plugins/core`, `src/plugins/core`,
  `tests/plugins/core`, and CLI files if command syntax is affected.
- For crypto plugin work (AES leakage / Pearson PoI / annotation conversion), read
  `docs/context/modules/plugins-crypto.md`, then inspect only
  `include/leakflow/plugins/crypto`, `src/plugins/crypto`, `tests/plugins/crypto`,
  and CLI files if command syntax is affected.
- For extras plugin work, read `docs/context/modules/plugins-extras.md`, then
  inspect only `include/leakflow/plugins/extras`, `src/plugins/extras`,
  `tests/plugins/extras`, and CLI files if command syntax is affected.
- For plot work, read `docs/context/modules/plot.md`, then inspect only
  `include/leakflow/plot`, `include/leakflow/plugins/plot`, `src/plot`,
  `src/plugins/plot`, the relevant domain→plot bridge, and CLI/base plugin files
  if plot runtime integration or `TorchConvert` syntax is affected.
- For GUI planning, read `docs/context/modules/gui.md` first. GUI work is not
  currently implemented and must not pull GUI dependencies into core.

## Long Docs

The long-form design and roadmap documents remain authoritative, but they are
not default startup context:

- `docs/design/architecture.md`
- `docs/design/dataflow_sync_model.md` — dataflow / vector-clock / liveness design
  of record (read for any executor, sync, `Queue`, or live-streaming work).
- `docs/design/dataflow_sync_walkthroughs.md` — ten worked pipeline examples that
  exercise the sync/threading/liveness model end-to-end (validation +
  implementation reference for the live phase).
- `docs/reference/PHASE_OVERVIEW.md`
- `docs/reference/CLI_SYNTAX.md`
- `docs/design/profiling.md` — execution timing telemetry / profiling design
  (per-element + op-scope timing, `--print-profile`, `--profile-file` trace).
- `docs/design/clustering_evaluation_metrics.md` — design of record for full
  clustering evaluation and its separate metric-visualization follow-up; read
  only for clustering metric/result/plot work.

Read the relevant sections of those files only when a task needs detailed
design history or CLI syntax details.

## Implementation Rule

LeakFlow is phase-limited. If the user asks for a phase, implement only that
phase. If the user asks for design/context cleanup, keep changes documentation
only unless they explicitly request code, build, or test changes.

Before editing files, summarize the exact requested phase or task and list the
planned files. Wait only if the user explicitly asks to wait.

## Default Validation

For implementation changes:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For documentation-only context changes, a build is usually optional. Report that
tests were not run if no commands were run.
