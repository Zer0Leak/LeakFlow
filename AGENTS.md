# AGENTS.md

## Project identity

This project is LeakFlow, a C++ side-channel analysis framework with a pipeline/plugin architecture inspired by GStreamer.

The project is not a generic machine learning framework. It is an SCA framework for reproducible experiments involving traces, leakage models, points of interest, labeling, attacks, and optional deep learning.

Primary validation target:
- AES, for simple classical SCA validation.

Primary research target:
- Kyber / ML-KEM, for later phases.

## Incremental development rule

Never implement more than the requested roadmap phase.

If the user asks for Phase N, do not implement Phase N+1.

Every phase must leave the repository buildable and testable.

When a phase adds a smoke executable or validation executable, register it with CTest unless the phase explicitly says it must be manual-only.

When uncertain, prefer the smallest useful implementation that satisfies the current phase.

Do not add future dependencies, abstractions, plugin families, CUDA, libtorch, AES, Kyber, YAML, GUI, or dynamic plugin loading unless required by the current phase.

## Framework-oriented rule

LeakFlow is a framework first. Always design and change code framework-oriented:
generic, reusable elements, properties, and core mechanisms that any experiment
can use — never one-off logic for a single paper or attack baked into shared code.

The only place attack/paper-specific tuning or narrowing belongs is under
`replications/` (one subdirectory per replicated paper). Those apps may hard-code
dataset paths, pipeline expressions, property choices, and attack-specific
parameters, because they are replication-specific.

Everywhere else — `include/`, `src/`, core, and all plugins — stays generic. If a
replication needs behavior the framework lacks, add it to the framework as a
general capability (a property, element, or mode) and then select/tune it from the
replication app. Do not bake the specific case into shared code.

Before editing files:

1. Read AGENTS.md and ROADMAP.md.
2. Summarize the exact requested phase.
3. List files you plan to create or modify.
4. Wait for explicit confirmation if the user asked you to wait.

After editing files:

1. List changed files.
2. Show build commands.
3. Show test commands.
4. State whether tests were actually run.
5. State whether the phase is complete.
6. State the next recommended phase.

Do not claim tests passed unless they were actually run.

## Build philosophy

Use C++23 by default.

Use Clang as the default C++ compiler for development and validation.

Use clang tooling for future C++ formatting and static analysis work, especially `clang-format` and `clang-tidy`, when a roadmap phase introduces those checks.

Use CMake with target-based configuration.

Prefer Ninja as the generator.

The default validation commands are:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```
