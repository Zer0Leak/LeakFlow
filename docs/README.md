# LeakFlow Documentation

This directory holds documentation that is useful but not part of the small
root entry set.

Root Markdown is intentionally limited to:

- `AGENTS.md`
- `README.md`
- `ROADMAP.md`
- `TESTING.md`

## Context

`docs/context/` is the compact Codex/context loading pack. Start there when a
session needs project context without reading the full source tree.

## Guides

`docs/guides/` contains task-oriented command references.

- `docs/guides/CLI_BUILD.md`: host, devcontainer, Docker, CUDA, and CLI smoke
  commands.
- `docs/guides/AES.md`: AES fixture and leakage-model pipeline examples.
- `docs/guides/LOGGING.md`: logging and summary examples.
- `docs/guides/PLOTTING.md`: interactive trace plotting examples.

## Reference

`docs/reference/` contains longer reference and history documents.

- `docs/reference/CODING_STYLE.md`: coding conventions.
- `docs/reference/CLI_SYNTAX.md`: `leakflow run` pipeline syntax.
- `docs/reference/PHASE_OVERVIEW.md`: phase-by-phase strategic history.

## Design

`docs/design/` contains detailed design records.

- `docs/design/architecture.md`: accepted architecture, payload, tensor,
  plugin, and dataflow design direction.

## Reports

Generated reports are intentionally not part of the compact documentation set.
Regenerate them when needed with tools such as `tools/project_size_report.sh`.
