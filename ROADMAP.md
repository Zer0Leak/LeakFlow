# ROADMAP

LeakFlow development is phase-limited. Do not implement work from a later phase while completing the current phase.

## Phase 0: Guidance Files

Goal: establish the project identity, development rules, roadmap, architecture notes, coding style, and testing expectations.

Allowed work:

- Write and maintain repository guidance documents.
- Define phase boundaries and review expectations.
- Clarify what is intentionally out of scope for Phase 0.

Out of scope:

- Implementation source files.
- Build-system files.
- Runtime components.
- Algorithm-specific work.
- Dependency integrations.
- Containers or deployment files.
- Generated artifacts.

Exit criteria:

- `README.md` explains the repository status and document set.
- `ROADMAP.md` defines Phase 0 and the next phase.
- `docs/design/architecture.md` captures the current architecture and dataflow design direction.
- `docs/reference/CODING_STYLE.md` records future coding conventions.
- `TESTING.md` records future validation expectations.

## Phase 1: Minimal Buildable Scaffold

Goal: add the smallest possible project scaffold that can configure, build, and run a minimal test.

Expected work:

- Add a minimal build configuration.
- Add a minimal source layout.
- Add a minimal test target.
- Verify the standard build and test workflow.

Out of scope:

- Domain algorithms.
- Data loaders.
- Experiment pipelines.
- Runtime extension loading.
- Acceleration support.
- External learning libraries.
- Containers.

## Phase 2: VS Code Devcontainer Toolchain

Goal: make the existing Phase 1 scaffold build and test inside a VS Code devcontainer.

Expected work:

- Add `.devcontainer/devcontainer.json`.
- Add `.devcontainer/Dockerfile`.
- Add `docker/README.md`.
- Provide a reproducible containerized development environment for VS Code.
- Install only the development tools needed to build and test the current scaffold.

The devcontainer should include:

- `cmake`.
- `ninja-build`.
- `clang`.
- `lldb` or `gdb`.
- `git`.
- `python3`.
- `clang-format`.
- `clang-tidy`.

CUDA note:

Use a CUDA-capable NVIDIA base image if practical, because later phases will need NVIDIA/CUDA support. However, Phase 2 must not add CUDA source code, CUDA C++ compilation, libtorch, or GPU-specific project logic.

Out of scope:

- CUDA C++ source code.
- CUDA smoke tests.
- libtorch.
- SCA data structures.
- Pipeline classes.
- Plugins.
- AES.
- Kyber.
- YAML.
- External project dependencies.
- Changes to existing executable behavior.

Validation commands inside the container:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Expected executable output:

```text
LeakFlow dev build
```

## Phase 3: NVIDIA Visibility In Container

Goal: verify that the Docker/devcontainer environment can see the host NVIDIA GPU.

Validation command inside the container:

```bash
nvidia-smi
```

Out of scope:

- CUDA source code.
- CUDA smoke executable.
- libtorch.
- SCA data structures.
- Pipeline classes.
- Plugins.
- AES.
- Kyber.
- YAML.
- GUI.

## Phase 4: CUDA Smoke Executable

Goal: compile and run the smallest possible CUDA program from the existing CMake project.

Context:

The Docker/devcontainer toolchain is working, and NVIDIA visibility has been validated manually outside VS Code.

Expected work:

- Add a CMake option named `LEAKFLOW_WITH_CUDA`.
- Enable CUDA language only when `LEAKFLOW_WITH_CUDA=ON`.
- Add a small CUDA smoke executable.
- Register the CUDA smoke executable with CTest when `BUILD_TESTING=ON`.
- The executable should:
  - print CUDA device count,
  - print the first CUDA device name,
  - run a tiny vector-add kernel,
  - verify the result,
  - print `Vector add OK` on success.

Out of scope:

- libtorch.
- DNN code.
- SCA data structures.
- Pipeline classes.
- Plugins.
- AES.
- Kyber.
- YAML.
- GUI.
- Complex CUDA abstractions.
- Changing existing CPU-only executable behavior.

Validation commands:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLEAKFLOW_WITH_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
./build-cuda/leakflow_cuda_smoke
```

Expected output should include:

```text
Vector add OK
```

Also verify the CPU build still works:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

## Phase 5: Core Data Containers — Caps and Buffer

Goal: introduce the smallest core data containers needed before building pipeline elements.

Expected work:

- Add a `Caps` type for lightweight type/capability descriptors.
- Add a `Buffer` type that carries a `Caps` value and simple metadata.
- Add tests for `Caps` and `Buffer`.
- Keep the implementation independent of SCA algorithms.

Caps requirements:

- Store a type string such as `sca/traceset` or `sca/labels`.
- Store string key/value parameters.
- Use deterministic parameter ordering.
- Support construction from explicit values.
- Support stringify/`to_string` behavior.
- Do not implement full MIME parsing yet.

Buffer requirements:

- Store `Caps`.
- Store simple metadata as string key/value pairs.
- Provide methods to set, get, and check metadata keys.
- Do not store trace tensors yet.

Out of scope:

- Element.
- Pipeline.
- Plugin registry.
- TraceSet.
- LabelSet.
- PoiSet.
- AES.
- Kyber.
- CUDA changes.
- libtorch.
- YAML.
- GUI.
- Dynamic plugin loading.
- File formats.
- Complex MIME/caps negotiation.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Exit criteria:

- CPU build still works.
- Existing smoke test still passes.
- New `Caps` tests pass.
- New `Buffer` tests pass.
- Existing CUDA smoke configuration remains optional and untouched.

## Phase 6: Element Interface and Linear Execution

Goal: introduce the smallest executable pipeline concept after `Caps` and `Buffer`.

Expected work:

- Add an `Element` base class.
- Add a `LinearPipeline` class.
- Add tests for `Element` lifecycle behavior and linear buffer flow.
- Keep execution synchronous and single-threaded.
- Keep the implementation independent of SCA algorithms.

Element requirements:

- Store an element name.
- Provide `name()` accessor.
- Provide virtual `start()`.
- Provide pure virtual `process(std::optional<Buffer> input)`.
- Provide virtual `stop()`.
- Use `std::optional<Buffer>` so source-like, transform-like, and sink-like behavior can be represented.

LinearPipeline requirements:

- Store elements in insertion order.
- Start elements in insertion order.
- Pass one optional `Buffer` through the sequence.
- Stop elements in reverse order.
- Return the final optional `Buffer`.
- Use `std::unique_ptr<Element>` ownership.

Out of scope:

- Plugin registry.
- Dynamic plugin loading.
- Pads.
- Caps negotiation.
- Branching graphs.
- Async scheduler.
- TraceSet.
- LabelSet.
- PoiSet.
- Payload storage.
- AES.
- Kyber.
- CUDA changes.
- libtorch.
- YAML.
- GUI.
- File formats.
- External dependencies.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- New `Element` tests pass.
- New `LinearPipeline` tests pass.
- Existing CUDA smoke configuration remains optional and untouched.

## Phase 7: Pad Model and Caps Declarations

Goal: introduce pads as named input/output interface declarations for elements without changing the Phase 6 execution model.

Expected work:

- Add a `PadDirection` enum.
- Add a `Pad` class.
- Add tests for pad name, direction, and caps preservation.
- Keep pads as declarations only.
- Do not change `LinearPipeline` execution.

Pad requirements:

- Store a pad name.
- Store a direction.
- Store `Caps`.
- Provide `name()` accessor.
- Provide `direction()` accessor.
- Provide `caps()` accessor.

Out of scope:

- Pad linking.
- Caps negotiation.
- Caps compatibility.
- Graph execution.
- Changes to `LinearPipeline` behavior.
- Plugin registry.
- Dynamic plugin loading.
- TraceSet.
- LabelSet.
- PoiSet.
- Payload storage.
- AES.
- Kyber.
- CUDA changes.
- libtorch.
- YAML.
- GUI.
- File formats.
- External dependencies.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- New `Pad` tests pass.
- Existing CUDA smoke configuration remains optional and untouched.

## Phase 8: Element Pad Declarations

Goal: allow elements to declare input and output pads without changing the Phase 6 linear execution model.

Expected work:

- Extend `Element` so it can store input pad declarations.
- Extend `Element` so it can store output pad declarations.
- Add tests for input pad declarations.
- Add tests for output pad declarations.
- Add tests for rejecting wrong pad directions.
- Add tests for rejecting duplicate pad names within the same direction.
- Keep `LinearPipeline` execution unchanged.

Element pad declaration requirements:

- Add `add_input_pad(Pad pad)`.
- Add `add_output_pad(Pad pad)`.
- Add `input_pads()` accessor.
- Add `output_pads()` accessor.
- Validate pad direction.
- Reject duplicate pad names within the same direction.

Out of scope:

- Pad linking.
- Graph execution.
- Caps negotiation.
- Caps compatibility.
- Scheduler changes.
- Changes to `LinearPipeline` behavior.
- Plugin registry.
- Dynamic plugin loading.
- TraceSet.
- LabelSet.
- PoiSet.
- Payload storage.
- AES.
- Kyber.
- CUDA changes.
- libtorch.
- YAML.
- GUI.
- File formats.
- External dependencies.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- New `Element` pad declaration tests pass.
- Existing `LinearPipeline` tests still pass unchanged.
- Existing CUDA smoke configuration remains optional and untouched.

## Phase 9: Linked Linear Pipeline

Goal: allow `LinearPipeline` to record, validate, and execute explicit pad-to-pad links while keeping execution linear and synchronous.

Expected work:

- Add a `PadLink` model.
- Extend `LinearPipeline` to store element handles and pad links.
- Add `link()` to `LinearPipeline`.
- Add `links()` accessor.
- Add tests for valid and invalid links.
- Make linked `LinearPipeline` runs follow the declared link chain.

Link requirements:

- Use modern C++23 ownership handles, such as `std::shared_ptr<Element>`, instead of storing only element indices in links.
- Link from one element output pad handle to a later element input pad handle.
- Validate source element handle exists in the pipeline.
- Validate sink element handle exists in the pipeline.
- Validate source pad exists.
- Validate sink pad exists.
- Validate source pad is an output pad.
- Validate sink pad is an input pad.
- Validate source element handle appears before sink element handle in pipeline order.
- Validate simple caps type equality.
- Reject exact duplicate links.
- Reject links from a source pad that is already linked.
- Reject links to a sink pad that is already linked.
- Keep pad links one-to-one; fork behavior belongs in a later `tee`-style element.
- When links exist, `run()` should execute the linked elements in link-chain order and pass the single buffer along that chain.

Out of scope:

- Branching execution.
- Graph scheduler.
- Async execution.
- Multi-buffer routing.
- Caps parameter negotiation.
- Wildcard caps.
- Plugin registry.
- Dynamic plugin loading.
- TraceSet.
- LabelSet.
- PoiSet.
- Payload storage.
- AES.
- Kyber.
- CUDA changes.
- libtorch.
- YAML.
- GUI.
- File formats.
- External dependencies.
- Changing the no-link `LinearPipeline` fallback behavior.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- New `PadLink`/link tests pass.
- Existing no-link `LinearPipeline` execution tests still pass.
- Existing CUDA smoke configuration remains optional and untouched.

## Future Phase Planning

### Planning Authority

After Phase 9, future work is driven by:

- User decisions made during phase planning.
- `docs/design/architecture.md`.
- The incremental development rule in this file.

The phase list below is intentionally high-level. Before implementing any phase, review the design decisions that affect that phase, update the phase details, then implement only that phase.

## Phase 10: Core Payload Interface

Goal: let `Buffer` carry real data through the existing pipeline without making the core know about tensors, AES, Kyber, file formats, or Torch.

Expected work:

- Add a `Payload` base class.
- Add optional payload storage to `Buffer`.
- Add payload accessors and safe typed downcasting.
- Add tests using a test-only payload type.

Design decisions:

- `Buffer` is the pipeline envelope.
- `Payload` is the data body inside the buffer.
- A `Buffer` carries zero or one `Payload`.
- A payload may internally be a bundle or batch.
- Future queues store `Buffer` objects, not payloads inside one buffer.
- `set_payload(nullptr)` clears the payload.

Out of scope:

- Torch / LibTorch.
- NumPy `.npy`.
- Torch `.pt` / `.pth`.
- AES.
- Kyber / ML-KEM.
- `TraceSet`.
- Tensor payloads.
- Tensor bundle payloads.
- PoI payloads.
- Label payloads.
- Plugin families.
- Plugin registry.
- Plotting.
- Hardware capture.
- Dynamic plugin loading.
- YAML.
- GUI.
- CUDA changes.
- External dependencies.
- File formats.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- New payload tests pass.
- Existing CUDA smoke configuration remains optional and untouched.

## Phase 11: Element Properties and Plugin Descriptors

Goal: make elements configurable and make future plugin-provided elements easy to inspect before adding real plugin families.

Expected work:

- Add a small typed property value model for element configuration.
- Add property specifications with defaults, descriptions, units, and validation constraints.
- Extend `Element` so it can declare properties, expose current values, and set validated values.
- Add descriptor structs for plugin and element metadata.
- Add tests for property defaults, setting, type validation, range validation, string enum validation, list values, interval values, and descriptor storage.

Design decisions:

- Properties are small user-controlled settings, not data transport.
- Experiment data belongs in `Payload`, not properties.
- Use an inspectable C++23 value model based on `std::variant`, not `std::any`.
- Support scalar property values: bool, integer, double, and string.
- Support small list property values: integer list, double list, and string list.
- Support interval property values: integer interval and double interval.
- Treat a string enum as a constraint on a string property, not as a separate value type.
- Keep validation constraints separate from values. For example, integer range constraints validate scalar integers, while integer interval values represent windows such as sample ranges.
- Descriptors should provide enough metadata for future `leakflow-ls` listing and future `leakflow` CLI property setting.

Out of scope:

- Real `leakflow-ls` executable.
- Real `leakflow` CLI runner.
- Plugin registry.
- Dynamic plugin loading.
- Plugin families.
- YAML or config file parsing.
- AES.
- Kyber / ML-KEM.
- Torch / LibTorch.
- Tensor payloads.
- File formats.
- Plotting.
- CUDA changes.
- External dependencies.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- New property tests pass.
- New descriptor tests pass.
- Existing CUDA smoke configuration remains optional and untouched.

## Phase 12: Core CLI, Inspect Tool, Core Plugin Library, and Tee

Goal: make LeakFlow runnable and inspectable with a linked shared core plugin library before adding tensor payloads or algorithm-specific plugins.

Expected work:

- Turn `leakflow` into a basic CLI runner for small preset pipelines.
- Support setting element properties from the CLI using the Phase 11 property model.
- Add a `leakflow-ls` inspect tool.
- Make `leakflow-ls` list linked plugin/element descriptors when called without arguments.
- Make `leakflow-ls <plugin>` inspect one linked plugin descriptor.
- Make `leakflow-ls <element>` inspect one linked element descriptor.
- Support `leakflow-ls --no-colors`.
- Support `leakflow-ls --color` and `leakflow-ls -C` to force ANSI colors.
- Add `leakflow_plugins_core` as a shared library target.
- Add a descriptor catalog for `leakflow_plugins_core`.
- Store plugin author and license metadata in descriptors.
- Put core plugin element headers and sources under the core plugin target
  layout.
- Keep each core plugin element in its own header/source pair.
- Add core plugin elements:
  - `FileSrc`,
  - `FileSink`,
  - `FakeSrc`,
  - `FakeSink`,
  - `Summary`,
  - `Tee`,
  - `Queue`.
- Add minimal branching execution needed for `Tee`.
- Add payload ownership helpers needed for safe mutation decisions, such as `payload_is_unique()` and `mutable_payload_if_unique<T>()`.
- Add tests for CLI behavior, inspect behavior, property setting, core plugin elements, Tee branching, Queue behavior, raw byte file source/sink behavior, and payload sharing semantics.

Design decisions:

- Follow the GStreamer-style distinction between plugins and elements: `leakflow_plugins_core` is one plugin shared library, and `FileSrc`, `FileSink`, `FakeSrc`, `FakeSink`, `Summary`, `Tee`, and `Queue` are elements provided by that plugin.
- The CMake target is named `leakflow_plugins_core`; on Unix-like systems the shared library file is expected to be `libleakflow_plugins_core.so`.
- Current plugin/element author metadata is `Zer0Leak <edgard.lima@gmail.com>`.
- Current plugin license metadata is `Apache-2.0`.
- `leakflow-ls` is an inspect tool over descriptors linked into the executable in this phase.
- `leakflow-ls` should broadly follow `gst-inspect-1.0`: no argument lists all known features, a plugin argument shows plugin details, an element argument shows factory/element details, `--no-colors` disables ANSI styling, and `--color` / `-C` forces ANSI styling.
- The `leakflow-ls` ANSI palette should follow `gst-inspect-1.0`: yellow section headings, bright-blue field labels and property names, green feature/type names and flags, magenta hierarchy connectors, and cyan property values.
- Phase 12 does not add dynamic plugin loading or a plugin registry.
- `FileSrc` and `FileSink` are generic raw-byte elements only; format-specific NumPy, Torch, AES, and Kyber I/O remains out of scope.
- `FakeSrc` produces simple test buffers.
- `FakeSink` consumes buffers for validation.
- `Summary` originally provided text buffer descriptions with a summary level
  property; Phase 18 replaces the peer-description path with Buffer-owned
  structured summaries.
- `Tee` is generic because it forks one input buffer to multiple output branches without interpreting the payload.
- `Queue` stores `Buffer` objects synchronously; async scheduling remains out of scope.
- `Tee` copies the `Buffer` envelope for each branch.
- `Tee` does not deep-copy payloads.
- Buffer envelope copies keep caps, metadata, and payload pointer state independent per branch.
- Payload pointers are shared across Tee branches through `std::shared_ptr<Payload>`.
- In-place payload mutation is allowed only when the payload is uniquely owned.
- Use the payload `std::shared_ptr` ownership count as a conservative uniqueness check.
- Mutating elements should use an API that checks uniqueness before returning mutable access, such as `mutable_payload_if_unique<T>()`.
- If a payload is shared, a mutating element must create or set a replacement payload rather than mutating the shared payload in place.
- Do not add a generic deep-copy or clone requirement until real payload types define meaningful clone semantics.
- `Mux` is intentionally out of scope because muxing depends on input data semantics, synchronization, caps, and payload combination rules.

Out of scope:

- `Mux`.
- Multi-input execution beyond what `Tee` branching requires.
- Dynamic plugin loading.
- Plugin registry.
- YAML or config file parsing.
- Tensor payloads.
- Torch / LibTorch.
- NumPy `.npy`.
- Torch `.pt` / `.pth`.
- AES.
- Kyber / ML-KEM.
- File-format parsing beyond generic raw bytes.
- Plotting.
- Hardware capture.
- Async queues or threaded scheduling.
- GUI.
- External dependencies.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow --help
./build/leakflow-ls
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- New CLI tests pass.
- New inspect-tool tests pass.
- New `leakflow-ls` element inspection tests pass.
- New `leakflow-ls --no-colors` tests pass.
- `leakflow-ls --color` / `-C` manual smoke output uses the GStreamer-like color palette.
- New core plugin element tests pass.
- New Tee branching tests pass.
- New Queue tests pass.
- Payload sharing and uniqueness behavior is tested.
- Existing CUDA smoke configuration remains optional and untouched.

## Phase 13: Source Layout and CLI Pipeline Syntax

Goal: make the `leakflow` executable accept a small, inspectable pipeline expression language before adding tensor payloads or algorithm-specific elements.

Expected work:

- Move application entry points from `apps/` into `src/apps/`.
- Update CMake paths for `leakflow`, `leakflow-ls`, and the optional CUDA smoke executable.
- Preserve the existing `leakflow-ls` behavior from Phase 12.
- Add a `leakflow run EXPRESSION` command.
- Implement the Phase 13 CLI syntax documented in `docs/reference/CLI_SYNTAX.md`.
- Support semicolon-separated statements.
- Support element creation with optional instance names, such as `FakeSrc` and `FakeSrc@src`.
- Support existing named element references, such as `@src`.
- Support explicit pad references, such as `@tee.src_0`.
- Support property assignment with `(...)` using the Phase 11 property model.
- Support scalar, list, and interval property value syntax where the target `PropertySpec` allows it.
- Support shell-expanded paths by allowing suitable bare string values.
- Support `!` links with pad inference only when the relevant side has exactly one matching pad.
- Require explicit pad references for links from elements with multiple output pads or into elements with multiple input pads.
- Support explicit pad caps annotations using `@element.pad[caps=TYPE; key=value]`.
- Support metadata annotations using `Element{key=value}` and `@element.pad{key=value}`.
- Keep pad caps behavior minimal and deterministic; full caps negotiation remains later work.
- Parse pad caps and metadata annotations without applying them to element pads or buffers yet; leave TODOs for later runtime behavior.
- Add tests for parsing, property setting, link construction, Tee branching syntax, pad inference, required explicit pads, pad caps annotations, metadata annotations, combined annotations, and error cases.

Design decisions:

- `docs/reference/CLI_SYNTAX.md` is the source of truth for the user-facing Phase 13 CLI syntax.
- `!` is the canonical link operator, following the useful part of GStreamer pipeline syntax.
- `;` finishes one CLI statement.
- `@name` names an element instance for later references, diagnostics, and future logging.
- `@name.pad` identifies a specific pad.
- Parentheses configure element properties only.
- Square brackets on explicit pad references annotate pad caps.
- Braces annotate metadata.
- Pad inference is an ergonomic shortcut, not a graph-discovery mechanism.
- Element-level caps and metadata annotations infer the only output pad; multi-output elements require explicit pad references.
- The parser should validate element types, property names, property values, pads, and link compatibility against linked descriptors and runtime element declarations.
- The CLI language should remain small and manually typed; YAML, config files, and a full graph language remain out of scope.

Out of scope:

- Tensor payloads.
- Torch / LibTorch.
- NumPy `.npy`.
- Torch `.pt` / `.pth`.
- AES.
- Kyber / ML-KEM.
- Dynamic plugin loading.
- Plugin registry.
- Filesystem plugin scanning.
- YAML or config file parsing.
- Full graph language.
- General multi-input execution semantics beyond validating explicit pad references.
- Mux.
- Full caps negotiation.
- Wildcard caps.
- Async queues or threaded scheduling.
- Plotting.
- Hardware capture.
- GUI.
- External dependencies.
- CUDA behavior changes beyond updating the moved source path.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow --help
./build/leakflow run 'FakeSrc ! Summary'
./build/leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! FakeSink'
./build/leakflow-ls
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- Application sources are under `src/apps/`.
- Existing Phase 12 CLI compatibility tests either still pass or are intentionally replaced by Phase 13 syntax tests.
- `leakflow run 'FakeSrc ! Summary'` works.
- `leakflow run 'FakeSrc(caps_type=sca/test) ! Summary(level=2)'` works.
- `leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary; @t.src_1 ! FakeSink'` works.
- Links involving multi-output or multi-input elements require explicit pads when inference would be ambiguous.
- Element property values are parsed and validated through `PropertySpec`.
- Pad caps annotation syntax is parsed and tested.
- Metadata annotation syntax is parsed and tested.
- A command using properties, caps, and metadata together is tested.
- Invalid syntax and invalid graph references fail with useful errors.
- Existing CUDA smoke configuration remains optional and otherwise untouched.

## Phase 14: Modular CMake, Source, and Test Layout

Goal: keep the existing Phase 13 behavior while splitting the growing build and
test layout into smaller directory-level units.

Expected work:

- Keep the root `CMakeLists.txt` as the top-level project configuration and
  orchestration file.
- Move `leakflow_core` source files under `src/core/`.
- Add `src/core/CMakeLists.txt` for the core library target.
- Keep application sources under `src/apps/`.
- Add `src/apps/CMakeLists.txt` for `leakflow`, `leakflow-ls`, `leakflow_cli`,
  and the optional CUDA smoke executable.
- Add a directory-level CMake file for the linked core plugin shared library.
- Split tests into:
  - `tests/core/`,
  - `tests/apps/`,
  - `tests/plugins/core/`.
- Add a `tests/CMakeLists.txt` that only delegates to the test subdirectories.
- Preserve existing target names where practical.
- Preserve existing CTest names and pass/fail expectations.
- Keep the top-level CMake options and definitions visible to subdirectories
  through normal `add_subdirectory()` directory scope.
- Keep configuration target-based: include directories, compile features, and
  link dependencies should stay attached to the targets that need them.

Out of scope:

- Runtime behavior changes.
- New pipeline syntax.
- New elements or plugin families.
- Dynamic plugin loading.
- Tensor payloads.
- Torch / LibTorch.
- NumPy `.npy`.
- AES.
- Kyber / ML-KEM.
- YAML or config file parsing.
- CUDA behavior changes beyond preserving the existing optional smoke target.
- External dependencies.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow --help
./build/leakflow run 'FakeSrc ! Summary'
./build/leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! FakeSink'
./build/leakflow-ls
```

Exit criteria:

- CPU build still works.
- Existing tests still pass.
- Root `CMakeLists.txt` delegates core, app, plugin, and test details to
  subdirectory CMake files.
- `src/core/CMakeLists.txt` owns the `leakflow_core` target.
- `src/apps/CMakeLists.txt` owns the app and CLI targets.
- The core plugin directory-level CMake file owns the core plugin target.
- `tests/CMakeLists.txt` delegates to grouped test CMake files.
- Existing Phase 13 CLI validation commands still work.
- Existing CUDA smoke configuration remains optional and otherwise untouched.

## Phase 15: LibTorch-backed Base Tensor Payload Layer

Goal: introduce the required base numerical data layer for future LeakFlow
plugins by adding Torch-backed tensor payloads without adding I/O formats,
statistics, AES, Kyber, or DNN elements yet.

Expected work:

- Add `leakflow_base` as a required library target.
- Make `leakflow_base` depend on `leakflow_core`.
- Make `leakflow_base` depend on LibTorch through `find_package(Torch REQUIRED)`.
- Keep `leakflow_core` independent of LibTorch.
- Add `TorchTensorPayload` for one `torch::Tensor`.
- Add `TorchTensorBundlePayload` for deterministic named tensor collections.
- Store bundle entries as `std::shared_ptr<TorchTensorPayload>` so individual
  tensors can later split into different pipeline paths.
- Add tests for tensor payload construction, metadata helpers, validation, and
  buffer transport.
- Add tests for tensor bundle insertion, lookup, deterministic names, missing
  names, validation, and buffer transport.
- Update build and testing documentation for mandatory LibTorch discovery.

Design decisions:

- `leakflow_base` is the project-wide numerical/data layer from this phase
  onward.
- Future generic statistical methods belong in `leakflow_base`, but are out of
  scope for Phase 15.
- `leakflow_core` stays a small framework library with type-erased payload
  transport only.
- `TorchTensorPayload` accepts CPU or CUDA tensors.
- Phase 15 tests use CPU tensors only.
- `TorchTensorPayload` rejects undefined tensors.
- `TorchTensorPayload` rejects non-strided tensor layouts such as sparse
  tensors.
- `TorchTensorPayload` allows non-contiguous strided tensors.
- `TorchTensorBundlePayload` rejects empty names and null tensor payloads.
- Bundle names are deterministic through `std::map`.
- Mutable tensor access is allowed, but mutating elements should still respect
  `Buffer::payload_is_unique()` before mutating payload state in place.
- Payload uniqueness does not guarantee unique Torch storage because Torch views
  and tensor handles may share underlying storage.

Out of scope:

- Generic statistics.
- NumPy `.npy`.
- Torch `.pt` / `.pth` file I/O.
- Dataset loaders.
- New core plugin elements.
- AES.
- Kyber / ML-KEM.
- DNN training or inference elements.
- CUDA smoke behavior changes.
- Dynamic plugin loading.
- YAML or config file parsing.
- Plotting.
- Hardware capture.
- GUI.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow --help
./build/leakflow run 'FakeSrc ! Summary'
./build/leakflow-ls
```

Exit criteria:

- CPU build with mandatory LibTorch works.
- Existing tests still pass.
- New `leakflow_base` tests pass.
- `leakflow_core` does not include or link LibTorch.
- `leakflow_base` exposes Torch tensor and tensor-bundle payloads.
- Existing Phase 13 CLI validation commands still work.
- Existing CUDA smoke configuration remains optional and otherwise untouched.

## Phase 16: Extras NumPy Payload and Loader

Goal: add NumPy `.npy` array data as an extras-library file-format concept
before exposing it through pipeline elements or converting it to Torch tensors.

Expected work:

- Add `leakflow_extras` as a library target.
- Make `leakflow_extras` depend on `leakflow_core`.
- Make `leakflow_extras` depend on `cnpy++` for `.npy` loading.
- Keep `leakflow_base` focused on Torch-backed tensor payloads.
- Add `NumpyPayload` to `leakflow_extras`.
- Keep `NumpyPayload` as a thin wrapper around `cnpypp::NpyArray`, similar to
  how `TorchTensorPayload` wraps `torch::Tensor`.
- Add a `load_npy(path)` extras-library function that returns `NumpyPayload`.
- Support a small deterministic `.npy` subset first.
- Preserve the metadata exposed by `cnpy++`, including shape, word size, labels,
  memory order, and raw data access.
- Store the loaded data without converting to `TorchTensorPayload`.
- Add tests using tiny deterministic `.npy` fixtures.
- Keep the API callable directly from applications that link `leakflow_extras`.

Initial `.npy` support:

- Numeric arrays only.
- C-contiguous arrays first.
- Little-endian arrays first.
- Common integer, unsigned integer, floating-point, and bool dtypes.
- `.npy` only, not `.npz`.

Out of scope:

- `NumpySrc`.
- Torch conversion.
- Moving NumPy support into `leakflow_base`.
- Conversion registry.
- Dynamic pads.
- Summary redesign.
- Logging.
- Local trace-file dependency in CTest.
- AES.
- Kyber / ML-KEM.
- Plotting.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- New `NumpyPayload` tests pass.
- `load_npy(path)` can load tiny checked-in or generated fixtures.
- `leakflow_base` remains independent of `cnpy++`.
- No pipeline element is added yet.

## Phase 17: Extras Plugin `NumpySrc`

Goal: expose the extras NumPy loader through the pipeline as a plugin element.

Expected work:

- Add a new linked plugin target named `leakflow_plugins_extras`.
- Add `NumpySrc`.
- Make `NumpySrc` call `leakflow_extras::load_npy(path)`.
- Make `NumpySrc` output a `NumpyPayload`.
- Add a descriptor catalog for `leakflow_plugins_extras`.
- Make the existing inspect tooling see the linked extras plugin if practical
  within the current static descriptor model.
- Add tests using tiny deterministic `.npy` fixtures generated from the local AES
  synchronized dataset.
- Add a pipeline smoke test shaped like:

```bash
leakflow run 'NumpySrc(path=tests/fixtures/numpy/aes_sync_key_01_traces_first30.npy) ! Summary ! FakeSink'
```

Design decisions:

- `leakflow_extras` is a library/API layer with application-callable helpers
  such as `NumpyPayload` and `load_npy(path)`.
- `leakflow_base` must not depend on `leakflow_extras`; `leakflow_extras` is
  layered above `leakflow_base`.
- `leakflow_plugins_extras` is a plugin/element layer that links
  `leakflow_extras`.
- `libleakflow_plugins_extras.so` is one shared library for the extras plugin
  family; `NumpySrc` is the first element in that plugin.
- `NumpySrc` has one required property, `path`.
- `NumpySrc` declares its `src` pad caps as `leakflow/numpy-array`.
- `NumpySrc` emits `Buffer(Caps("leakflow/numpy-array"))` with a
  `NumpyPayload`.
- Generic sink pads declared as `leakflow/buffer` may accept concrete buffer
  caps such as `leakflow/numpy-array`; this preserves `NumpySrc ! Summary`
  without duplicating NumPy facts into generic caps.
- Do not duplicate NumPy intrinsic details such as dtype, rank, shape,
  word size, or memory order into buffer caps or metadata.
- `NumpySrc` may stamp minimal file provenance metadata such as `element`,
  `file.path`, and `file.size`.
- Semantic metadata such as `role=traces`, `algorithm=aes`, or
  `capture=chipwhisperer` belongs to the application, CLI metadata
  annotations, or later dataset-specific elements.
- Keep the current `Summary` behavior for this phase; the buffer/payload-based
  summary redesign belongs to Phase 18.

Out of scope:

- Torch conversion.
- `Convert`.
- Dynamic pads.
- Structured Summary redesign.
- Logging.
- AES-specific dataset source behavior.
- Local trace-file dependency in CTest.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow run 'NumpySrc(path=tests/fixtures/numpy/aes_sync_key_01_traces_first30.npy) ! Summary ! FakeSink'
```

Exit criteria:

- Existing tests still pass.
- `NumpySrc` tests pass.
- `NumpySrc ! Summary ! FakeSink` works with a tiny repository fixture.
- Tests do not depend on local trace folders.

## Phase 18: Structured Summary Documents

Goal: improve the Summary system by separating structured description data from
terminal rendering, so colors and future logging can evolve without mixing
formatting into payload logic.

Expected work:

- Add a small structured summary model, such as summary documents, sections,
  fields, and nested records.
- Make `Buffer`, not the previous peer element, the source of Summary
  descriptions.
- Add a way for payloads or payload helpers to describe themselves at a
  requested detail level.
- Update `Summary` to render structured summaries.
- Add deterministic plain rendering for tests.
- Add a small shared render layer for terminal presentation primitives used by
  both `Summary` and `leakflow-ls`.
- Add `fmt` as the formatting/color dependency for the shared render layer.
- Add color policy and color-theme concepts separated from the structured data
  model.
- Add UTF-8 glyph support with an ASCII fallback.
- Keep color disabled in automated tests unless a test explicitly checks ANSI
  behavior.

Design decisions:

- Summary descriptions accept a detail level.
- Structured summary data should be independent of color.
- Colors should be applied by renderers or themes after structured data is
  built.
- `Summary` should render the `Buffer` that flows through it and should not ask
  a previous peer element to describe the buffer.
- `leakflow-ls` and `Summary` should share terminal color/theme/glyph
  primitives, but they do not need to share the same document model.
- UTF-8 glyphs are presentation details controlled by the renderer.
- Automated tests should use deterministic no-color rendering; manual terminal
  output may use color automatically.
- Summary is not the logging system.
- Future logging and concurrent-safe output belong in a later phase.

Out of scope:

- Concurrent logging.
- Global logging sinks.
- Conversion registry.
- Dynamic pads.
- New file formats.
- AES.
- Kyber / ML-KEM.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- Summary output remains deterministic in test mode.
- Tensor and NumPy payload summaries are more informative than `type_name()`
  alone.
- Color rendering, if added, is separate from summary document construction.

## Phase 18.1: Source and Include Layout Consolidation

Goal: organize existing Phase 18 targets under a single root `include/` and
`src/` layout without changing runtime behavior.

Expected work:

- Move public application-facing library headers under `include/`.
- Move plugin-facing headers under `include/`.
- Reserve a base plugin include folder and `src/plugins/base/` for future
  base-backed plugin elements without adding a target yet.
- Move base, extras, and plugin sources under:
  - `src/base/`,
  - `src/extras/`,
  - `src/plugins/core/`,
  - `src/plugins/extras/`.
- Keep existing CMake target names, public include spelling, namespaces,
  descriptors, CLI behavior, and tests unchanged.
- Update documentation to describe the new layout.

Design decisions:

- Application-facing library headers and plugin-facing headers both live under
  the top-level `include/` tree.
- Current CLI code may still include concrete plugin element headers until a
  later factory/API cleanup phase replaces direct construction.

Out of scope:

- NumPy-to-Torch conversion.
- `Convert` element.
- `leakflow_plugins_base` target.
- New plugin elements.
- Dynamic plugin loading.
- Factory registry refactors.
- Behavior changes.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- Existing executable and library target names remain unchanged.
- Root project code is organized under `include/` and `src/`.

## Phase 18.2: Public Include Tree Unification

Goal: make all public include paths live under one `leakflow/...` namespace-like
tree without changing runtime behavior.

Expected work:

- Move core public headers under `include/leakflow/core/`.
- Keep render public headers under `include/leakflow/render/`.
- Move base public headers under `include/leakflow/base/`.
- Move extras public headers under `include/leakflow/extras/`.
- Move plugin-facing headers under:
  - `include/leakflow/plugins/core/`,
  - `include/leakflow/plugins/extras/`,
  - `include/leakflow/plugins/base/`.
- Update source files, tests, CMake include roots, and documentation for the
  new include spelling.
- Keep existing CMake target names, namespaces, descriptors, CLI behavior, and
  tests unchanged.

Design decisions:

- Public include spelling follows the C++ namespace shape:
  - `leakflow/core/...`,
  - `leakflow/base/...`,
  - `leakflow/extras/...`,
  - `leakflow/render/...`,
  - `leakflow/plugins/...`.
- Plugin headers stay visibly separated under `leakflow/plugins/*`, but remain
  part of the single LeakFlow public include tree.

Out of scope:

- NumPy-to-Torch conversion.
- `Convert` element.
- New plugin elements or plugin targets.
- Dynamic plugin loading.
- Factory registry refactors.
- Behavior changes.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- Existing executable and library target names remain unchanged.
- New public include directives compile.
- Root public headers are organized under `include/leakflow/`.

## Phase 19: NumPy-to-Torch Conversion API

Goal: add direct application-callable conversion from `NumpyPayload` to
`TorchTensorPayload` using the extras NumPy payload and the base Torch payload.

Expected work:

- Add conversion options for target dtype and target device.
- Add `convert_numpy_to_torch(...)` or an equivalent function in the appropriate
  layer that can depend on both `leakflow_extras` and `leakflow_base`.
- Support CPU conversion in tests.
- Allow CUDA as an option when the local Torch build and runtime support it.
- Keep conversion callable directly from applications without using a plugin.

Out of scope:

- Pipeline `Convert` element.
- Conversion registry.
- Dynamic pads.
- File I/O beyond existing `load_npy`.
- AES.
- Kyber / ML-KEM.
- DNN training or inference.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- NumPy-to-Torch conversion tests pass.
- Conversion preserves shape and expected values for tiny fixtures.
- CPU conversion is always tested.

## Phase 20: Explicit `NumpyToTorch` Element

Goal: expose the NumPy-to-Torch conversion API through one explicit pipeline
element before building a general conversion registry or generic `Convert`
autoplugger.

Expected work:

- Add `NumpyToTorch` to `leakflow_plugins_extras`.
- Make `NumpyToTorch` support `NumpyPayload -> TorchTensorPayload`.
- Make `NumpyToTorch` call `convert_numpy_to_torch(...)`.
- Use `numpy-to-torch` as the conversion implementation id.
- Configure conversion with simple properties such as target dtype and device.
- Add pipeline tests shaped like:

```bash
leakflow run 'NumpySrc(path=tests/fixtures/numpy/aes_sync_key_01_traces_first30.npy) ! NumpyToTorch ! Summary ! FakeSink'
```

Design decisions:

- `NumpyToTorch` is the explicit element name.
- `numpy-to-torch` is the conversion implementation id.
- The generic `Convert` element is intentionally left for a later conversion
  registry phase.
- Future generic `Convert` should follow the GStreamer-style autoplugging
  pattern: a smart wrapper element chooses among compatible ranked conversion
  implementations by caps, payload capability, and requested target, while its
  internal delegated element/pads remain implementation details.
- Exact element construction should remain exact; the factory should not
  silently replace a requested type with another element.

Out of scope:

- Generic `Convert`.
- Dynamic pads.
- Pluggable conversion registry.
- Automatic graph-wide negotiation.
- Multiple conversion families.
- AES.
- Kyber / ML-KEM.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- `NumpySrc ! NumpyToTorch ! Summary ! FakeSink` works with a tiny
  repository fixture.
- Conversion behavior matches the base-library conversion tests.

## Phase 21: Logging Foundation

Goal: add a project-owned logging layer for CLI and pipeline diagnostics before
plotting and AES validation work.

Expected work:

- Add `leakflow_log` as the logging target.
- Use spdlog internally, hidden behind LeakFlow-owned headers and APIs.
- Add LeakFlow-owned environment variables:
  - `LEAKFLOW_LOG_LEVEL`,
  - `LEAKFLOW_LOG_COLOR`,
  - `LEAKFLOW_LOG_FILTER`,
  - `LEAKFLOW_SUMMARIES`.
- Add CLI options:
  - `--log-level`,
  - `--log-color`,
  - `--log-filter`,
  - `--summaries`,
  - `--no-summaries`.
- Use precedence:

```text
CLI option > LEAKFLOW_* environment variable > default
```

- Support log levels:
  - `off`,
  - `error`,
  - `warning`,
  - `info`,
  - `debug`,
  - `trace`.
- Support log color modes:
  - `auto`,
  - `always`,
  - `never`.
- Support first filter syntax as comma-separated AND clauses:

```text
element=NumpySrc,element_name=src,element_kclass=Source,component=pipeline
```

- Implement only `=` clauses for these initial filter keys:
  - `component`,
  - `element`,
  - `element_name`,
  - `element_kclass`.
- Reserve future filter expansion for richer operators and fields.
- Add global summary enable/disable behavior.
- Make `Summary` pass buffers through silently when summaries are disabled.
- Add a `Summary(always_print=true)` property to override disabled summaries.
- Log core pipeline lifecycle and link decisions.
- Log element-level source/sink/transform/conversion events at appropriate
  levels.
- Keep logs SCA-safe by default: do not log raw traces, tensor values, NumPy
  contents, key bytes, plaintext arrays, or other secret material.
- Add a tutorial with copy/paste logging examples.
- Update the devcontainer/Docker documentation and default environment.

Out of scope:

- Plotting.
- AES helper library.
- AES plugin elements.
- AES dataset sources.
- Pearson PoI finder elements.
- Generic `Convert`.
- Conversion registry.
- Dynamic pads.
- Full graph-wide caps negotiation.
- Async scheduling.
- General multi-input execution.
- Kyber / ML-KEM.
- Logging.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- Logging unit tests pass.
- CLI logging and environment variable tests pass.
- `Summary(always_print=true)` overrides global summary suppression.
- Logs go to stderr and ordinary pipeline output remains on stdout.
- All existing elements have useful SCA-safe log records at suitable levels.
- The logging tutorial contains runnable examples.
- Docker/devcontainer docs mention the LeakFlow logging environment variables.

## Phase 22: ImGui/ImPlot TracePlot Plugin Foundation

Goal: add the first interactive plotting capability after logging is available,
without adding a new executable and without pulling GUI/plotting dependencies
into `leakflow_core`.

Expected work:

- Add a `leakflow_plot` target for Dear ImGui, ImPlot, backend selection, plot
  sessions, and plot-owned snapshots.
- Add a `leakflow_plugins_plot` target for plotting pipeline elements.
- Add `TracePlot` as the first plot plugin element.
- Add a concrete `TorchConvert` element to `leakflow_plugins_base` for
  `TorchTensorPayload -> TorchTensorPayload` conversion.
- Keep `TorchConvert` explicit. Do not reintroduce generic `Convert` or a
  conversion registry in this phase.
- Use ImGui + ImPlot for interactive plotting.
- Do not add a new executable. The existing `leakflow run` command should own
  the plot event loop when one or more plot elements register plot sessions.
- Support a shared `PlotRuntime` or equivalent session registry so multiple
  `TracePlot` elements can be displayed by one GUI loop after pipeline
  execution.
- Support plot grouping instead of dynamic sink pads for this phase.
- Let multiple `TracePlot` elements with the same `group` appear together in
  one comparison view.
- Support comparison layouts:
  - `overlay`: draw traces on the same axes,
  - `stacked`: draw traces one above another.
- Support alpha/transparency for overlaid traces.
- Support independent trace sliders by default.
- Support a lock option so grouped rank-2 traces can move together by trace
  index.
- Support `TracePlot` input as `TorchTensorPayload` with rank 1 or rank 2.
- Make `TracePlot` pass the input buffer through after taking a plot-owned
  snapshot.
- Treat rank 1 as one trace/vector.
- Treat rank 2 as `[trace, sample]`, where axis 0 is selected by a slider and
  axis 1 is plotted.
- Require `TracePlot` runtime input to be CPU `float32` at first.
- Declare plot sink caps for `leakflow/torch-tensor` with expected numeric caps
  vocabulary such as `dtype=float32` and `device=cpu` where practical.
- Make `TracePlot` take a plot-owned snapshot because the GUI loop outlives
  pipeline processing. The snapshot must not expose mutable access to the
  upstream payload.
- Add a save-to-PNG button to the UI.
- Add properties for:
  - `group`,
  - `label`,
  - `title`,
  - `x_label`,
  - `y_label`,
  - `layout`,
  - `alpha`,
  - `line_width`,
  - `trace_index`,
  - `lock_trace_index`,
  - `x_axis`,
  - `sample_rate_hz` as an optional override.
- Use `sample_rate_hz` as the canonical metadata key for trace sampling rate.
- Prefer data facts in metadata and display choices in properties.
- Resolve time-axis sampling rate as:

```text
TracePlot(sample_rate_hz=...) > buffer metadata sample_rate_hz > sample index fallback
```

- Support `x_axis=sample|time_us`.
- Log plot/session/backend decisions through `leakflow_log`, without logging
  trace values.
- Provide manual validation commands and documentation. This GUI phase is
  manual-only for new plot behavior; existing CTest validation must still pass.
- Provide a simple optional tutorial application that builds a minimal
  application-owned plot pipeline using the LeakFlow libraries. Keep it out of
  the default build.

Out of scope:

- Dynamic sink pads.
- Multi-input executor changes.
- A new `leakflow-plot` executable.
- A new default-built user-facing executable.
- Headless rendering tests.
- CTest registration for GUI/window rendering.
- Automatic graph insertion of `TorchConvert`.
- AES helper library unless explicitly requested as part of a later phase.
- AES PoI logic.
- Kyber / ML-KEM.
- Generic `Convert`.
- Dynamic plugin loading.
- GUI framework abstraction beyond the chosen ImGui/ImPlot backend shape.
- CUDA/OpenGL or CUDA/Vulkan interop.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Manual-only GUI validation examples:

```bash
./build/leakflow run 'NumpySrc(path=tests/fixtures/numpy/aes_sync_key_01_traces_first30.npy) ! NumpyToTorch ! TorchConvert(dtype=float32, device=cpu) ! TracePlot(title="AES traces", group=aes)'
./build/leakflow run 'NumpySrc(path=tests/fixtures/numpy/aes_sync_key_01_traces_first30.npy) ! NumpyToTorch ! Tee@t; @t.src_0 ! TracePlot(group=compare,label=raw); @t.src_1 ! TorchConvert(dtype=float32,device=cpu) ! TracePlot(group=compare,label=converted,layout=overlay,lock_trace_index=true)'
```

Exit criteria:

- Existing CTest suite still passes.
- `leakflow_plot` and `leakflow_plugins_plot` build.
- `TorchConvert` can prepare CPU float32 `TorchTensorPayload` input for
  plotting.
- `TracePlot` accepts rank-1 and rank-2 CPU float32 Torch tensors.
- Rank-2 input exposes a trace index slider.
- Grouped `TracePlot` sessions can be viewed in overlay or stacked layout.
- Locked trace index mode works for grouped rank-2 sessions.
- The GUI has a save-to-PNG button.
- Plot logs are SCA-safe and do not print trace values.
- No plotting dependency enters `leakflow_core`.

## Phase 23: `leakflow_crypto` Helpers

Goal: add the smallest crypto/SCA helper library needed for near-term classical
AES validation without adding crypto pipeline elements yet.

Expected work:

- Add `leakflow_crypto` as a reusable helper library above `leakflow_base`.
- Keep `leakflow_core` independent of crypto helpers.
- Add optimized scalar Hamming weight helpers for 1, 2, 4, and 8 byte unsigned
  values.
- Add optimized scalar Hamming distance helpers using Hamming weight of XOR.
- Add Torch `uint8` Hamming weight and Hamming distance helpers.
- Add AES S-box helpers.
- Add AES first-round S-box leakage helpers for scalar bytes and Torch tensors.
- For key byte `k` and plaintext byte `m`, compute `y = AES_SBOX[m XOR k]`.
- Expose public leakage outputs as `m`, `y`, `HW(m)`, and `HW(y)`.
- Do not expose `m XOR k` as a public leakage output.
- Support Torch vector leakage helpers for scalar key bytes, Torch scalar key
  bytes, per-trace key byte vectors, and `[N,16]` key/plaintext block tensors
  with a selected byte index.
- Add deterministic unit tests with small known-value checks.

Out of scope:

- `leakflow_plugins_crypto`.
- Crypto pipeline elements.
- AES dataset sources or source elements.
- Pearson PoI finder elements.
- CPA/report generation.
- AES full encryption/decryption.
- AES key schedule.
- Generic `Convert`.
- Conversion registry.
- Dynamic pads.
- Full graph-wide caps negotiation.
- Async scheduling.
- General multi-input execution.
- Kyber / ML-KEM.
- Logging redesign.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- Crypto helper tests pass.
- Crypto helpers are available without pulling crypto concepts into
  `leakflow_core`.
- No new pipeline behavior is added.

## Post-Phase 24 Graph-Inspection Support

Goal: add a live pipeline graph inspector without turning `leakflow_core` into a
GUI or asynchronous execution framework.

Implemented work:

- Add neutral copied observation snapshots in core through `PipelineObserver`
  and `PipelineEvent`.
- Let `LinearPipeline` emit topology, lifecycle, error, and routed-buffer
  observations.
- Keep observations SCA-safe by carrying caps, metadata, payload presence/type,
  sequence counters, and no raw payload data.
- Add an ImGui `PipelineGraphRuntime` and graph renderer in `leakflow_plot`.
- Add `leakflow run --graph EXPRESSION`.
- In graph mode, run the existing synchronous pipeline in a `std::jthread`
  worker while the main thread owns the ImGui graph/plot loop.
- Draw the graph and any `TracePlot` sessions in one UI loop.
- Add a reusable app-facing `run_pipeline_graph_until_closed(...)` helper so
  linked applications can show the same graph/plot window for manually built
  pipelines or pipelines built through the CLI helper.
- Add `PipelineControlRuntime` in `leakflow_plot` as an explicit live-control
  path separate from copied graph observation.
- Render element controls from `PropertySpec` / `PropertyValue`, with graph gear
  buttons plus a standalone controls panel path.
- Add `PropertyEffect` metadata to `PropertySpec` so properties can declare
  whether changes are `ui-only`, `sink-display`, `metadata-output`,
  `payload-output`, `caps-output`, or `lifecycle`.
- Add copied `PropertyChanged` observer events carrying element identity,
  property name, stringified old/new values, and `PropertyEffect` metadata.
- Mark `AesLeakage.channels` as `payload-output` with downstream invalidation
  on the `leakage` output pad.

Design decisions:

- Core owns only neutral observer/event data and remains free of GUI concepts.
- The UI consumes snapshots/events and does not scrape live `Element` objects
  from another thread.
- The control plane is separate from the observation plane:
  `PipelineGraphRuntime` stores copied display state only, while
  `PipelineControlRuntime` weakly binds to live elements and is the explicit
  mutation path.
- Observer callbacks are diagnostic and must not affect experiment correctness.
- Buffer observations are coalesced by link in the graph runtime; the latest
  buffer summary and an observed count are shown.
- `PlotRuntime` uses a narrow mutex so `TracePlot` snapshots can be published
  from the worker while the main thread renders.
- Property changes that only affect display should not create new buffers.
  Property changes that affect caps, metadata, or payload are dataflow changes
  and must eventually produce new downstream observations.
- `PropertyEffect` records intent only in this support phase; it does not yet
  implement partial rerun.
- Cooperative stop/cancel for long-running live sources remains future work.

Out of scope:

- General async pipeline scheduling.
- Thread-safe command application through a `PipelineController`.
- Cached per-pad/per-element buffers for partial rerun.
- Downstream-only rerun after property changes.
- Live stream epochs and queue flush/drain policies.
- Dynamic pads or graph-wide negotiation.
- Headless GUI rendering tests.
- Hardware capture.
- A full GUI/lab frontend.

## Phase 25: Full `PipelineController` and Incremental Rerun

Goal: add the first real control/session layer for property changes, cached
buffers, downstream-only rerun in synchronous pipelines, and live-stream epoch
semantics, without turning `leakflow_core` into a GUI or a general async
scheduler.

This is the next recommended numbered implementation phase.

The detailed, authoritative design (decisions `D1`..`D12`, with rationale) lives
in `docs/design/pipeline_controller.md`. The finalized decisions are:

- D1: one executor; `run()` is sugar over `start_all`/`run_sweep`/`rerun_from`/
  `stop_all`; dead `run_from()` removed.
- D2: `PipelineSession` in `leakflow_core`; cache + rerun primitive inside
  `LinearPipeline`; `PipelineControlRuntime` becomes a session UI client.
- D3: `Element::can_replay()` (default true, mirrored to `ElementDescriptor`,
  `Queue` returns false); two-clause contract (replay purity + lifecycle reset);
  effective scope = max(`PropertyEffect`, escalation from a non-replayable
  element in the replay-set).
- D4: caps-output changes validated against downstream links before commit;
  rejected transactionally if a link would break.
- D5: `Buffer::epoch()` added now, executor is single writer; session epoch is
  monotonic for the whole session; `QueueEpochPolicy` enum defined as a contract
  only; `Queue` stays synchronous.
- D6: thread-safe command queue (last-wins coalescing per `(element, property)`);
  safe-point application (between units of work); copied
  `CommandAccepted/Rejected/Applied` events on the existing observer stream;
  persistent worker loop with a pluggable drive policy; `StreamingDrive` /
  cooperative-cancel seams reserved.
- D7: per-input-pad and per-output-pad cache (per-input-pad mandatory for
  multi-input rerun); global caching toggle, default ON; OFF disables partial
  rerun (full re-sweep fallback); per-pad opt-out reserved.
- D8: only queued command is `SetProperty`; restart, re-run-from-sources, caching
  toggle, and live start/stop are direct session controls applied at safe points.
- D9: `PipelineSession` owns the `LinearPipeline` (move-in); app UI takes
  `PipelineSession&`.
- D10: session owns lifecycle (start once → run/rerun many → stop at teardown).
- D11: session-level state machine (`Stopped/Started/Running`; `Paused` +
  pause/resume + preroll reserved); two-step graceful stop (cooperative cancel →
  reverse-order `stop_all`); graph control buttons.
- D12: a rerun that throws is caught without tearing down the session; emits
  `CommandApplied{status=failed}`, marks affected cache entries stale, still bumps
  the epoch.

Expected work:

- Add a `PipelineController` or `PipelineSession` abstraction that owns safe
  command application for a `LinearPipeline`.
- Make UI controls send commands such as `SetProperty`, not direct element
  mutations.
- Apply commands at safe points and emit copied accepted/rejected/applied
  events.
- Treat element properties as the source of truth for all property-backed UI
  controls.
- Route graph gear controls, standalone control panels, and future element-local
  controls such as `TracePlot.trace_index` through the same control API when the
  setting is a property.
- Use `PropertyEffect` to choose between:
  - UI redraw only,
  - sink-display update,
  - downstream metadata-output rerun,
  - downstream payload-output rerun,
  - downstream caps-output rerun with link revalidation,
  - lifecycle/full-pipeline restart.
- Cache the latest accepted input buffers per element input pad and the latest
  output buffers per element output pad.
- On output-affecting property changes, invalidate the affected output pad(s)
  and walk downstream links to rerun only the affected synchronous path.
- If caps change, revalidate downstream links before rerun and report graph
  invalidation if links no longer match.
- When only metadata changes, create a new `Buffer` envelope with updated
  metadata and the appropriate shared payload policy rather than treating
  metadata as a side channel.
- Add generation or epoch identifiers to control-applied outputs so UI state can
  distinguish old and new observations.
- Define live-stream semantics for future threaded/live sources:
  - property changes start a new stream/config epoch,
  - queues use explicit drain/flush/keep-mixed policy,
  - buffers carry enough sequence/epoch information to avoid silent mixing.
- Keep `Queue` synchronous unless this phase explicitly introduces a narrow
  thread-boundary queue contract.
- Add non-visual tests for property command validation, property-change events,
  downstream invalidation, metadata-only rerun, payload-output rerun, caps-output
  rejection, and source-output rerun with cached buffers.

Design decisions:

- Observation plane:
  - copied, SCA-safe events only,
  - no mutable `Element` handles,
  - no raw payload data.
- Control plane:
  - explicit command API,
  - validated property changes,
  - accepted/rejected/applied events,
  - safe-point application.
- Dataflow changes are changes to caps, metadata, payload, or lifecycle.
  They must not be represented as UI-only state.
- Display changes are changes to how already observed data is shown. They should
  not create new buffers.
- For example, `AesLeakage.channels: [HW(m)] -> [HW(m), HW(y)]` is a
  payload-output change on `AesLeakage.leakage`. It should rerun that output
  path downstream to `PearsonPoiFinder`, annotation conversion, and plot sinks
  using cached/latest inputs.
- For example, a sink-side `TracePlot.sample_rate_hz` override is
  sink-display, while an upstream source stamping `capture.sample_rate_hz` is a
  metadata-output dataflow change.
- Live mode is epoch-based rather than "partial rerun but faster":
  future buffers after a control change use a new config epoch, and queues must
  make the old/new epoch policy explicit.

Out of scope:

- A general async scheduler.
- Dynamic pads.
- Graph-wide negotiation beyond caps revalidation needed by changed outputs.
- Dynamic plugin loading.
- New AES, Kyber, or plotting algorithms.
- Hardware capture implementations.
- Headless GUI rendering tests.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- Property control commands are validated and observable.
- UI-only and sink-display property changes do not rerun dataflow.
- Metadata-output, payload-output, and caps-output property changes trigger
  deterministic downstream invalidation behavior in the synchronous executor.
- Caps-output changes either rerun with compatible links or report a clear graph
  invalidation error.
- Live-stream epoch policy is represented in copied observations even if real
  hardware capture remains future work.
- No GUI or plotting dependencies enter `leakflow_core`.

## Post-Phase 24 Metadata Forwarding and Klass Taxonomy

Goal: replace ad-hoc per-element metadata copying with a single policy keyed on a
metadata taxonomy and the element forwarding profile, and restructure element
`klass` strings so the profile is derivable from the klass.

Implemented work:

- Add a four-group metadata taxonomy in `leakflow_core`: `capture`, `origin`,
  `payload`, `routing` (`metadata_group(key)`), resolved by dotted prefix plus a
  small dotless table, with unknown keys defaulting to `payload` so core stays
  domain-free.
- Add `ForwardingProfile` and `profile_for_klass(klass)` keyed on the leading
  klass token: `Source`, `Sink`, `PassThrough`, `Convert` (Reframe), `Analyze`.
- Add `forward_metadata(...)`: PassThrough copies capture/origin/payload;
  Reframe copies capture/origin and drops payload; Analyze unions capture
  (conflict is an error), relabels origin as `origin.<pad>.<key>`, and drops
  payload; routing is never forwarded.
- Restructure all element `klass` strings to `<Profile>/<Domain>[/<Specific>]`.
- Rewire new-buffer-building elements (`TorchConvert`, `NumpyToTorch`,
  `CorrelationPoiToPlotAnnotations`, `AesLeakage`, `PearsonPoiFinder`) to use
  `forward_metadata`, replacing the previous blind whole-map copies.
- `docs/design/metadata_klass_taxonomy.md` is the authoritative design document,
  including the SCA building-block (Lego) klass families and future elements.

Design decisions:

- Caps stay general; generality is set by the most generic consumer, not the
  producer. Raw loaders emit generic transport caps; semantics ride as metadata
  `role`. New payload types are introduced only for non-tensor structures
  (near-term: `sca-model`).
- Library layering: `leakflow_base` keeps generic statistics; a future
  `leakflow_sca` holds algorithm-agnostic SCA helpers and the shared caps/role
  vocabulary; `leakflow_crypto` stays algorithm-specific.

Deferred:

- Oracle (`Analyze/Oracle/*`) and Solver (`Analyze/Solver/*`) families are
  taxonomized (klass + caps reserved) but their iterative query→measure→refine
  execution waits for a future multi-input / feedback executor.
- Active fault injection (`Control/Fault/*`) and pre-silicon RTL (`*/Design/*`)
  are scoped as future families.
- A Reframe brick that rewrites a capture key it owns (for example `Resample`
  updating `sample_rate_hz`) waits for the signal-processing brick phase.
- The `leakflow_sca` / `leakflow_plugins_sca` targets and the physical move of
  generic SCA elements such as `PearsonPoiFinder` are a later phase.

## Phase 26: AES PoI Pipeline Correctness Validation

Status: DONE. Headless numeric correctness tests now run over the checked-in AES
fixtures. See `tests/plugins/crypto/aes_poi_correctness_test.cpp` (CTest
`leakflow_plugins_crypto_aes_poi_correctness`): it loads `key_01` and `key_02`,
asserts `AesLeakage` HW(m)/HW(y) for a known byte against values computed
directly from the fixture bytes, asserts `PearsonPoiFinder` selects the
strongest-correlation sample with a correlation matching an independent float64
recompute and clearing a sane threshold, and asserts
`CorrelationPoiToPlotAnnotations` sample indexes/values/precision formatting
match the PoI output.

Original brief follows; the full AES PoI plotting pipeline is already built and
runs end to end:

```text
TorchFileSrc(traces) ─ Tee ┬─ TracePlot.sink
                           ├─ PearsonPoiFinder.features
                           └─ AesLeakage.traces
TorchFileSrc(plaintexts) ─ AesLeakage.plaintexts
TorchFileSrc(key)        ─ AesLeakage.keys
AesLeakage ─ PearsonPoiFinder.targets
PearsonPoiFinder ─ CorrelationPoiToPlotAnnotations ─ TracePlot.annotations
```

It has been validated manually in `leakflow run --graph`, and `tests/apps` has
headless CLI smoke tests that assert the right metadata/caps flow through
(`leakflow_cli_run_aes_leakage`, `leakflow_cli_run_pearson_poi_finder`,
`leakflow_cli_run_correlation_poi_to_plot_annotations`). What is missing is proof
of numeric correctness.

Goal: add rigorous, headless correctness validation over the checked-in AES
fixtures, so the pipeline is proven to produce correct results, not just to run.

Expected work:

- Add a headless test (in `tests/plugins/crypto` or `tests/apps`) that runs the
  AES data path over `tests/fixtures/aes/sync/key_01` (and `key_02` where useful)
  and asserts numeric correctness, not just presence:
  - AesLeakage: HW(m)/HW(y) for a known byte index match values computed directly
    from the fixture plaintext/key via the `leakflow_crypto` helpers.
  - PearsonPoiFinder: the selected PoI sample index(es) are deterministic and land
    on the strongest-correlation sample(s); correlation magnitude exceeds a sane
    threshold.
  - CorrelationPoiToPlotAnnotations: annotation sample indexes/values match the
    PearsonPoiFinder output (including `precision` formatting).
- Prefer building the chain through
  `leakflow::cli::build_builtin_pipeline_from_expression(...)` or by wiring
  elements directly and running via `LinearPipeline::run()` / `PipelineSession`,
  so the test exercises the real pipeline rather than isolated elements.
- Keep tests independent of the ignored local `traces/` tree; use only checked-in
  fixtures.

Out of scope:

- Headless GUI/plot rendering tests (`TracePlot` remains manual-only).
- New elements, CPA/report generation (Phase 28), correlation/PoI overlay plot
  elements (Phase 29).
- Multi-input executor changes and dynamic plot sink pads (Phase 27).

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- Existing tests still pass.
- A new AES correctness test passes deterministically against the checked-in
  fixtures and asserts numeric leakage/PoI/annotation correctness (not just
  metadata presence).
- No dependency on the local `traces/` tree.

## Active Phase: Full Clustering Evaluation Metrics

Status: **in progress**. Phase A1, the exact numeric evaluator, is implemented;
the semantic metric slice is next. Phase A is not complete until its numeric,
alignment, payload, persistence, and pipeline exit criteria are all green.

Authoritative design:
`docs/design/clustering_evaluation_metrics.md` (Phase A).

Goal: replace the current scalar-class-only evaluation ceiling with a generic,
structured evaluator for predicted cluster IDs and vector-valued semantic truth,
without changing clustering itself or breaking existing `ClusteringStats`
pipelines.

Implementation sequence:

- **A1 — exact numeric core (done):** core-free `evaluate_clustering(...)` and
  `ClusteringEvaluationResult`; `[N]`/`[U,N]` arbitrary int64-representable IDs;
  `[N,D]`/`[U,N,D]` exact vector grouping; sparse contingency detail; ARI,
  arithmetic AMI, homogeneity, completeness, V-measure, purity, pair
  precision/recall/F1, and arithmetic NMI; explicit undefined reasons/supports;
  deterministic normalization and checked 64-bit unordered-pair counts.
- **A2 — semantic and fragmentation metrics (next):** normalized power-cost
  options/validation, merge rate and conditional severity, impurity, and
  fragmentation aggregates/details.
- **A3 — rectangular alignments (pending):** separate exact-overlap and
  semantic-cost assignments with unmatched support.
- **A4 — pipeline contract (pending):** `ClusteringEvaluationPayload`,
  `ClusteringEvaluate`, typed-unit alignment, summaries, persistence, descriptor
  registration, and compatibility tests.

Locked decisions:

- Numeric evaluation is GMM-independent and lives in `leakflow_ml`.
- Inputs are labels `[N]`/`[U,N]` and semantic truth
  `[N,D]`/`[U,N,D]`; exact groups come from full-vector equality.
- The required exact set is ARI, AMI, homogeneity, completeness, V-measure,
  purity, pair precision/recall/F1, plus compatibility NMI.
- Semantic results keep merge frequency, merge severity, impurity, and
  fragmentation separate; both micro and macro supports are explicit.
- The semantic quantity is a normalized power **cost**, not a mathematical
  distance. Exact-only mode needs no semantic ranges; semantic mode supports
  `power=1|2` (default 2), explicit ranges, and strict range validation so
  normalized results stay in `[0,1]`.
- Undefined denominators produce an unavailable metric with reason and support,
  never a misleading zero.
- Exact-overlap and semantic-cost Hungarian mappings are distinct, rectangular,
  and expose unmatched support.
- `ClusteringEvaluationResult` is a core-free numeric result in `leakflow_ml`;
  `ClusteringEvaluationPayload` and the new `ClusteringEvaluate` element live in
  `leakflow_plugins_ml`.
- Structured/per-group data stays in the payload, not string metadata.
- Current `ClusteringStats` input/output/caps remain unchanged as the legacy
  confusion-matrix adapter.

Remaining work:

- Extend the evaluator options/result model with semantic and fragmentation
  records and aggregated kernels.
- Add rectangular alignment support with hand-checked reference fixtures.
- Add `ClusteringEvaluate`, its descriptor/properties, bounded summary, typed-unit
  alignment, and a versioned persistence codec.
- Validate semantic metrics and alignments with hand-computed cases, then cover
  unit alignment and pipeline/persistence behavior. A1 already covers the
  conventional scikit-learn fixtures, exact degeneracies, arbitrary IDs,
  `D=1/2/4`, numeric batches/dtypes/validation, undefined pair metrics, and a
  non-quadratic expected-MI stress case.

Out of scope:

- Plot views/elements (the follow-up phase below).
- GMM fitting changes or using truth during clustering.
- Removal or silent repurposing of `ClusteringStats`.
- AES/Kyber-specific class encodings, inferred semantic ranges, arbitrary real
  power exponents, GPU kernels, bootstrap confidence intervals, or composite
  score use as a training objective.

Validation commands:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Exit criteria:

- All required exact and fragmentation results, plus semantic and requested
  alignment results when enabled, are available through the structured
  result/payload for batched and unbatched inputs.
- Undefined/value/support/direction/averaging semantics are explicit and tested.
- Numeric reference, pipeline, persistence, and compatibility tests pass.
- Existing `ClusteringStats ! HeatmapPlot` pipelines remain valid.
- No plotting dependency enters core, `leakflow_ml`, or `leakflow_plugins_ml`.

## Planned Follow-up Phase: Clustering Metric Visualization

Status: **planned, blocked on completion of Phase A, not implemented**.

Authoritative design:
`docs/design/clustering_evaluation_metrics.md` (Phase B).

Goal: make the complete structured evaluation result inspectable without placing
clustering semantics in the generic plot runtime or recomputing metrics in plot
elements.

Locked decisions:

- Add `leakflow_plugins_ml_plot`, depending on `leakflow_plugins_ml` and
  `leakflow_plot`, following the existing crypto→plot bridge pattern.
- Reuse domain-free `TableView` and `HeatmapView`; add a generic `MetricView` for
  grouped scalar and per-dimension bars.
- Plan `ClusteringMetricsTablePlot`, `ClusteringMetricsPlot`, and
  `ClusteringMatrixPlot` as consumers of `ClusteringEvaluationPayload`.
- Tables show value, direction, support, and `N/A` reason. Matrix views select
  raw, exact-aligned, or semantic-aligned stored results and may apply only
  display normalization (`none|row|col`).
- Style changes are `ui-control`; selecting another stored result is
  `sink-display`. Neither reruns clustering or evaluation, including while Idle.

Expected validation:

- Headless tests over copied table/metric/heatmap view data, selection,
  normalization, undefined values, unmatched annotations, history, and reset.
- A fixture payload proves bridge elements do not call metric computation.
- GUI rendering remains manual-only; record an offline multi-unit smoke and an
  Idle-state display-property change.

Exit criteria:

- Every Phase A result is available in a table, headline/per-dimension metrics
  have a generic visual form, and raw/aligned matrices are selectable.
- Plot changes do not trigger evaluation recomputation.
- No clustering-specific branch enters `PlotRuntime` or a generic plot view.

## Provisional Future Phase Plan

These phases are not detailed implementation specs yet. They should be refined one at a time before implementation.

```text
Phase 27: DAG executor, per-pad outputs, and vector-clock buffer provenance;
  remove Buffer::epoch(); rename LinearPipeline -> Pipeline (offline only). Design
  of record: docs/design/dataflow_sync_model.md. (DONE) Bundles/Mux/Demux were
  superseded by the vector clock for offline and moved to Phase 27.next.
Phase 27.next: Live streaming + Sync element. Fake live-capture source (reads a
  Torch .pt file, emits one Buffer per axis-0 row), liveness model (live/one-run
  sources; liveness-aware property changes), threaded segments + real Queue, and
  the generic N->N Sync element (sole front door for custom cross-source pairing).
  Default sync stays the per-slot barrier; one counter per slot (no global
  version). Bundles/Mux/Demux are DROPPED (superseded by the Sync element). Design
  of record: docs/design/dataflow_sync_model.md Sections 10-11.
Phase 28: AES CPA/report refinement
Phase 29: Additional plot elements such as correlation and PoI overlay plots
Phase 30: Kyber dataset source once traces exist
Phase 31+: Kyber PC-oracle / SPA building blocks
Phase 31+: Optional DNN, hardware capture, and GUI/lab frontend
Future low-priority: Generic Convert, conversion registry, and dynamic pads
```

The first full AES pipeline should arrive around the middle of this sequence, not at the end. Its target shape is:

```text
Read one AES key folder
  key.npy
  plain_texts.npy
  traces.npy
-> compute AES intermediate leakage model
-> find PoI indexes with Pearson correlation
-> plot traces/correlation/PoI overlays
```

AES algorithm helpers such as S-box and Hamming weight helpers belong in
`leakflow_crypto`.

AES pipeline elements such as an AES dataset source and AES Pearson PoI finder
belong in `leakflow_plugins_crypto`. The AES PoI finder should accept
properties such as the number of PoIs to find, with a reasonable default.

Generic `Convert`, the conversion registry, and dynamic pads are intentionally
deferred as low-priority future infrastructure. The explicit `NumpyToTorch`
element remains the preferred near-term path for NumPy-to-Torch conversion.

## Phase: Execution Timing Telemetry and Trace Export

Goal: add a built-in, domain-aware performance-profiling layer so users can see
which element (and which internal operation) is the bottleneck, without bolting
on an external profiler that does not understand the pipeline model.

Authoritative design: `docs/design/profiling.md`.

Design decisions:

- Timing is **time-flavored telemetry**: it reuses the existing telemetry
  plumbing (specs on descriptors, the observer stream, `leakflow-ls`) but has its
  own gate, because the clock read has a cost and a self-observer effect. The rule
  is "share the plumbing, split the gate".
- `TelemetryKind` tags each `TelemetrySpec`/`TelemetrySnapshot` as `Size`
  (monitoring, gated by `--telemetry`) or `Duration` (profiling, gated by the
  profiling switch). Profiling defaults OFF.
- Per-element timing is automatic: the executor (`Pipeline::timed_process_pads`)
  wraps every `process_pads(...)` call in the element's built-in `process`
  duration channel. The common `process` channel is published by
  `with_common_element_properties`, like the common `name` property.
- Internal op timing is opt-in: an element declares a duration spec and opens
  `Element::profile_scope("name")` (a no-op when profiling is off). `AesLeakage`
  ships the example (`leakage_compute`).
- Output is a frontend concern: `--print-profile` prints a per-element timing
  table (bottleneck-first) at exit; `--profile-file PATH` writes Chrome Trace
  Event JSON for `chrome://tracing` / Perfetto (one track per element, the only
  heavy, per-event path); and under `--graph` a live timing overlay publishes each
  element's duration telemetry to the graph panel.
- `leakflow_core` stays domain-free; the timing primitives are generic.

`Queue` profiles wait time (`backpressure` = downstream too slow, `starvation` =
upstream too slow), the clearest live "compute-bound vs stalling" signal.

Out of scope (deferred): per-pad timing and GPU/CUDA kernel timing.
