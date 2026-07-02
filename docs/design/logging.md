# LeakFlow Logging Design

Phase 21 adds a LeakFlow-owned logging layer backed by spdlog.

## Decisions

- Public code uses `leakflow/log/logger.hpp`, not spdlog directly.
- spdlog is an implementation detail of `leakflow_log`.
- Logs go to stderr.
- Normal pipeline output and summaries stay on stdout.
- CLI options override LeakFlow environment variables.
- Defaults are quiet:
  - log level: `warning`,
  - log color: `auto`,
  - log filter: empty,
  - summaries: enabled.
- `Summary` passes buffers through silently when summaries are disabled.
- `Summary(always_print=true)` overrides disabled summaries.
- Log filters start with simple `key=value` clauses and can expand later.
- Logs must be SCA-safe by default.

## Environment Variables

```text
LEAKFLOW_LOG_LEVEL=off|error|warning|info|debug|trace
LEAKFLOW_LOG_COLOR=auto|always|never
LEAKFLOW_LOG_FILTER='element=NumpySrc,element_name=src'
LEAKFLOW_SUMMARIES=0|1|true|false|on|off
```

## CLI Options

```text
--log-level LEVEL
--log-color auto|always|never
--log-filter FILTER
--summaries
--no-summaries
```

Precedence:

```text
CLI option > LEAKFLOW_* environment variable > default
```

## Filter Model

The current filter syntax is:

```text
key=value[,key=value...]
```

Multiple clauses use AND semantics.

Supported keys:

```text
component
element
element_name
element_kclass
```

Examples:

```text
element=NumpySrc
element_name=src
element_kclass=Source
component=pipeline,element_kclass=Source
```

Future-compatible but not implemented yet:

```text
level>=debug
element!=Summary
field.path~=fixtures
caps=leakflow/numpy-array
metadata.role=traces
```

Invalid filters fail fast.

## Element Kclasses

Current broad element classes:

```text
Source
Sink
Transform
Converter
Branch
```

These are logging/filtering categories, not C++ class names.

## Level Guidance

Use `error` for caught failures before returning diagnostics.

Use `warning` for surprising but recoverable behavior, such as queue drops.

Use `info` for high-level user-meaningful events:

- pipeline start/end,
- file loaded,
- file written,
- conversion completed.

Use `debug` for normal pipeline mechanics:

- element creation,
- links,
- start/stop,
- buffer pass-through,
- fake/test element activity.

Use `trace` for low-level library object creation and payload bookkeeping.

## Safety Rules

Never log raw experiment data by default:

- no key bytes,
- no plaintext arrays,
- no trace samples,
- no tensor values,
- no NumPy array contents,
- no secret intermediates.

Safe fields include:

- caps,
- dtype,
- device,
- rank,
- shape,
- file path,
- file size,
- element names,
- counts,
- lifecycle event names.
