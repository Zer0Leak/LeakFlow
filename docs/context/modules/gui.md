# GUI Module Context

GUI work is not implemented yet.

Use this file for planning GUI or frontend work without pulling GUI concepts
into the framework core.

## Current State

There is no GUI target, GUI source tree, GUI dependency, or GUI test suite.

There is a focused ImGui pipeline graph inspector in `leakflow_plot`. It is not
the general GUI/lab frontend described in this file. It consumes copied core
observer events and shares the existing plot window loop with `TracePlot`.

## Boundary

GUI code must not be added to `leakflow_core`.

GUI code should depend on public application-facing APIs, descriptors, summaries,
and CLI-compatible pipeline concepts. It should not force core, base, or extras
to depend on GUI libraries.

The graph inspector follows this boundary by keeping `PipelineObserver` and
event snapshots in core while keeping ImGui rendering in `leakflow_plot`.

## Possible Future Placement

When a roadmap phase asks for GUI work, choose one explicit placement:

```text
src/gui/                 GUI application target
include/leakflow/gui/    public GUI helper headers, only if needed
tests/gui/               focused non-visual GUI helper tests, if practical
```

or:

```text
src/plugins/gui/         GUI-related plugin family, if it behaves like plugins
include/leakflow/plugins/gui/
tests/gui_plugins/
```

Do not create both without a concrete reason.

## Design Direction

A future GUI should inspect descriptors, build pipeline expressions, launch or
embed pipelines, and render summaries/results. It should not become the only way
to run experiments.

The CLI remains an important reproducibility surface.

## Out Of Scope Until Requested

- ImGui/Qt/web frontend dependencies.
- Plotting plugins.
- Interactive graph scheduler.
- Live hardware capture.
- YAML/preset system.
- GUI-specific changes to core execution.
