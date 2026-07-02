# Log Module Context

Use this for work touching LeakFlow logging, log filters, summary gating, or
CLI/environment logging options.

## Files

Public headers:

- `include/leakflow/log`

Sources:

- `src/log`

Tests:

- `tests/log`

Related files when user-facing behavior changes:

- `src/apps`
- `tests/apps`
- element implementations that emit log records

## Target

- `leakflow_log`

## Dependency Contract

`leakflow_log` is the lowest infrastructure layer.

It may depend on:

- spdlog.
- C++ standard library.

It must not depend on:

- `leakflow_core`,
- `leakflow_render`,
- Torch / LibTorch,
- NumPy / cnpy++,
- AES,
- Kyber / ML-KEM,
- plotting,
- YAML,
- GUI,
- dynamic plugin loading.

Higher layers use LeakFlow-owned logging APIs rather than including spdlog
directly.

## Current API

- `LogLevel`
- `LogColorMode`
- `LogRecord`
- `LogFilter`
- `LogConfig`
- `configure(...)`
- `config_from_environment()`
- `write(LogRecord)`
- `summaries_enabled()`
- `summary_level()` / `parse_summary_level(...)`

`summary_level()` (default 2) is the payload detail level used by the pipeline's
routed-buffer observation/log summary (the `--graph` link tooltips and the
`info` "buffer flow" log line). It is independent of the `Summary` element's own
`level` property.

## Environment Variables

- `LEAKFLOW_LOG_LEVEL`
- `LEAKFLOW_LOG_COLOR`
- `LEAKFLOW_LOG_FILTER`
- `LEAKFLOW_SUMMARIES`
- `LEAKFLOW_SUMMARY_LEVEL` (integer 0-3; pairs with `--summary-level`)

CLI options override environment variables.

## Filter Syntax

Current filter clauses use comma-separated AND semantics:

```text
element=TorchFileSrc,element_name=src,element_kclass=Source,component=pipeline
```

Supported keys:

- `component`
- `element`
- `element_name`
- `element_kclass`

Only `=` is implemented. Reserve richer operators and data fields for later
phases.

## Safety Contract

Logs must be SCA-safe by default.

Do not log:

- raw traces,
- tensor values,
- NumPy contents,
- key bytes,
- plaintext arrays,
- secret intermediate values.

Safe fields include:

- component,
- element type,
- element instance name,
- element kclass,
- caps strings,
- dtype,
- device,
- rank,
- shape,
- file path when explicitly configured by the user,
- file size,
- counts and lifecycle events.

## Common Tests

```bash
ctest --test-dir build -R 'leakflow_log|leakflow_cli'
```
