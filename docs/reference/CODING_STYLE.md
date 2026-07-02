# CODING_STYLE

This document defines future coding expectations for LeakFlow. Phase 0 contains no implementation code.

## Language

Future implementation should use C++23 unless a later roadmap phase explicitly says otherwise.

## Toolchain

Use Clang as the default development compiler.

Use clang tooling for future C++ formatting and static analysis:

- `clang-format` for formatting once a formatting configuration is introduced.
- `clang-tidy` for static analysis once tidy checks are introduced.

Do not add formatting or static-analysis configuration files until the current roadmap phase requires them.

## General Style

- Prefer clear, small types with explicit ownership.
- Use standard-library facilities before adding dependencies.
- Keep names descriptive and domain-specific.
- Keep functions short enough to test and review comfortably.
- Prefer deterministic behavior for experiment logic.
- Avoid global mutable state unless a phase gives a concrete reason.
- Add comments only when they explain non-obvious design intent.

## Project Structure

Future source layout should be introduced incrementally. Do not add directories, libraries, executables, or tests until the requested phase needs them.

## Dependencies

Do not add dependencies speculatively. A dependency should be introduced only when the current roadmap phase requires it and when the validation path remains clear.

## Review Expectations

Code changes should be easy to connect to a roadmap phase. Reviews should focus on correctness, reproducibility, narrow scope, and whether tests cover the behavior introduced by the phase.
