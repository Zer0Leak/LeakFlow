# LeakFlow Phase Overview

This document explains how LeakFlow is evolving phase by phase.

`ROADMAP.md` is the strict control document for the current implementation phase. This file is more strategic: it explains the intended direction, why phases are ordered this way, and how the project grows from a minimal scaffold into a side-channel analysis pipeline framework.

## Development Philosophy

LeakFlow is being developed incrementally.

Each phase should:

* add one small useful capability,
* leave the repository buildable and testable,
* avoid speculative abstractions,
* avoid bringing future dependencies too early,
* preserve a clear path toward reproducible SCA experiments.

The main rule is:

```text
Build the smallest working thing first, then increment it.
```

LeakFlow should eventually feel conceptually similar to GStreamer, but adapted to side-channel analysis instead of media streams.

However, the project should not copy GStreamer mechanically. GStreamer is an inspiration for architectural concepts such as elements, pads, caps, buffers, and pipelines. LeakFlow should keep only what is useful for SCA experiments.

## Long-Term Goal

LeakFlow should become a modular SCA framework where experiments can be expressed as pipelines.

Example future pipelines:

```text
TraceSource -> Normalize -> WelchTTest -> TopKPoi -> ReportSink
```

```text
SyntheticAesTraceSource -> AesSboxHwLabeler -> CpaAttack -> GuessResultSink
```

```text
KyberTraceSource -> Segmenter -> Labeler -> PoiFinder -> ModelTrainer -> MetricsSink
```

The framework should eventually support:

* sources,
* preprocessing,
* leakage assessment,
* segmentation,
* point-of-interest selection,
* labeling,
* clustering,
* classical attacks,
* DNN training/inference,
* metrics,
* visualization,
* sinks,
* reproducible experiment presets.

The core must remain algorithm-agnostic and acceleration-agnostic.

The core should not know:

* AES,
* Kyber / ML-KEM,
* CUDA,
* LibTorch,
* ImGui,
* file formats,
* YAML experiment syntax.

Those belong in later plugins, helper libraries, or frontends.

## Current Strategic Direction

The current development path is:

```text
environment
  -> minimal build
  -> CUDA/toolchain validation
  -> core data descriptors
  -> minimal execution model
  -> pads and caps declarations
  -> payload transport
  -> element properties and plugin descriptions
  -> CLI, inspect tool, core plugin library, and tee branching
  -> source layout and CLI pipeline syntax
  -> modular CMake, source, and test layout
  -> LibTorch-backed tensor and dataset-bundle payloads
  -> NumPy payload loading
  -> NumpySrc pipeline input
  -> structured summary output
  -> unified public include tree
  -> NumPy-to-Torch conversion
  -> simple Convert element
  -> conversion registry and dynamic pads
  -> real AES dataset input
  -> AES helper library
  -> AES Pearson PoI finder
  -> trace/PoI plotting
  -> first full AES PoI pipeline
  -> broader graph/dataflow features
  -> Kyber-specific helpers when traces exist
  -> optional DNN, hardware, and GUI plugins
```

This sequence is intentional.

We validate the environment first so that later failures are more likely to be real code/design issues, not toolchain issues.

We introduce `Caps` and `Buffer` before pipeline execution so that elements have a common object to exchange.

We introduce a minimal `Element` and `Pipeline` before pads and graph scheduling so that execution can be tested early.

After Phase 9, the design direction changed from a `TraceSet`-first plan to a payload-first plan. A `TraceSet` or trace-batch helper may still appear later, but the immediate need is for `Buffer` to carry arbitrary payloads.

The first substantial end-to-end validation target should arrive relatively soon: read one AES key folder, compute Pearson-correlation PoIs from AES leakage hypotheses, and plot the result.

## Completed Phases

## Phase 0: Guidance Files

Purpose:

Establish project identity, development rules, roadmap, architecture notes, coding style, and testing expectations.

Main files:

* `AGENTS.md`
* `README.md`
* `ROADMAP.md`
* `docs/design/architecture.md`
* `docs/reference/CODING_STYLE.md`
* `TESTING.md`

Result:

The repository gained rules for controlled AI-assisted development.

Important decision:

Codex should not implement future phases early. Each phase must be buildable and testable.

## Phase 1: Minimal Buildable Scaffold

Purpose:

Create the smallest possible C++23 project that configures, builds, runs, and tests.

Added concepts:

* root CMake build,
* `leakflow_core` library,
* `leakflow` executable,
* `leakflow_tests` test target,
* minimal version/banner behavior.

Validation:

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/leakflow
```

Expected output:

```text
LeakFlow dev build
```

Result:

LeakFlow became a real buildable C++ project.

## Phase 2: VS Code Devcontainer Toolchain

Purpose:

Make the project build inside a reproducible Docker/devcontainer environment.

Added concepts:

* `.devcontainer/devcontainer.json`,
* `.devcontainer/Dockerfile`,
* Docker-based development toolchain,
* Clang/CMake/Ninja inside the container.

Important practical note:

Codex inside the VS Code devcontainer was unstable/frozen during development. The current preferred workflow is:

```text
Codex locally edits files.
Docker/devcontainer/manual container builds and validates.
```

Result:

The project can be built in a containerized toolchain without depending on host compiler setup.

## Phase 3: NVIDIA Visibility In Container

Purpose:

Verify that the Docker/container environment can see the host NVIDIA GPU.

Validation:

```bash
nvidia-smi
```

Result:

The environment was prepared for later CUDA and LibTorch phases.

Important decision:

Phase 3 validates GPU visibility only. It does not add CUDA source code.

## Phase 4: CUDA Smoke Executable

Purpose:

Compile and run the smallest possible CUDA executable from the existing CMake project.

Added concepts:

* `LEAKFLOW_WITH_CUDA` CMake option,
* optional CUDA language enablement,
* `leakflow_cuda_smoke`,
* CTest registration for CUDA smoke when enabled.

Validation:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLEAKFLOW_WITH_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
./build-cuda/leakflow_cuda_smoke
```

Expected output includes:

```text
Vector add OK
```

Result:

CUDA can be compiled and executed, but CUDA remains optional and outside the core.

Important decision:

CUDA is validated early but not integrated into the core pipeline.

## Phase 5: Core Data Containers — Caps and Buffer

Purpose:

Introduce the smallest data descriptors needed before building pipeline execution.

Added concepts:

* `Caps`,
* `Buffer`,
* deterministic string key/value parameters,
* deterministic metadata map,
* tests for both.

Design decisions:

`Caps` is a lightweight type/capability descriptor:

```text
type + string parameters
```

Example:

```text
sca/traceset; dtype=float32; layout=trace,sample
```

`Buffer` is the smallest object that can later flow through a pipeline:

```text
Caps + metadata
```

Important decisions:

* Use `std::map<std::string, std::string>` for deterministic ordering.
* Do not add payload yet.
* Do not add trace tensors yet.
* Do not add caps negotiation yet.
* Do not add pads yet.
* Do not add SCA-specific types yet.

Result:

LeakFlow now has the basic vocabulary for describing data objects.

## Phase 6: Element Interface and Linear Execution

Purpose:

Introduce the smallest executable pipeline concept after `Caps` and `Buffer`.

Added concepts:

* `Element`,
* `Pipeline`,
* lifecycle methods,
* direct buffer passing,
* tests for element behavior and linear execution.

Design decisions:

`Element` is the smallest unit of execution.

For now, an element:

* has a name,
* has `start()` and `stop()`,
* processes zero or one input `Buffer`,
* returns zero or one output `Buffer`.

The simplified Phase 6 API uses:

```cpp
std::optional<Buffer> process(std::optional<Buffer> input)
```

This allows:

* source-like elements,
* transform-like elements,
* sink-like elements.

`Pipeline` is a temporary synchronous executor:

```text
input = null
for each element:
    input = element.process(input)
return input
```

Important design note:

This is not the final GStreamer-like architecture.

GStreamer has pads. LeakFlow should also introduce pads later. Phase 6 intentionally skips pads to validate lifecycle and execution as soon as possible.

Result:

LeakFlow now has a minimal executable pipeline model.

## Phase 7: Pad Model and Caps Declarations

Purpose:

Introduce pads as named input/output ports of elements.

Why:

Phase 6 passes buffers directly from element to element. That is enough for a linear smoke pipeline, but not enough for real SCA flows.

Future SCA elements often need multiple inputs and outputs:

```text
CorrelationPoiFinder
  input: traces
  input: labels
  output: poi-set
```

```text
DnnTrainer
  input: traces
  input: labels
  output: model
  output: metrics
```

Expected concepts:

* `PadDirection`,
* `Pad`,
* input pads,
* output pads,
* declared caps,
* named ports,
* no full caps negotiation yet.

Possible minimal API:

```cpp
enum class PadDirection {
    Input,
    Output
};

class Pad {
public:
    Pad(std::string name, PadDirection direction, Caps caps);

    const std::string& name() const;
    PadDirection direction() const;
    const Caps& caps() const;
};
```

Out of scope:

* graph scheduler,
* dynamic linking,
* full negotiation,
* plugin registry,
* trace tensors,
* SCA algorithms.

## Phase 8: Element Pad Declarations

Purpose:

Allow elements to declare their input and output pads.

Why:

Pads should become part of the element interface, but execution can still remain simple.

Expected concepts:

* `Element` exposes declared input pads,
* `Element` exposes declared output pads,
* tests verify pad declaration,
* no graph scheduler yet.

Example future declaration:

```text
Normalize
  input pad: traces, sca/traceset
  output pad: traces, sca/traceset
```

## Phase 9: Linked Linear Pipeline

Purpose:

Move from direct insertion-order execution toward explicit named pad connections while keeping execution linear and synchronous.

Phase 9 records, validates, and executes one linear chain of pad-to-pad links:

```text
source:out -> normalize:in -> sink:in
```

Expected concepts:

* `PadLink`,
* `Pipeline::link(...)`,
* `Pipeline::links()`,
* element ownership handles using `std::shared_ptr<Element>`,
* link validation using element handles instead of stored numeric indices,
* simple caps type equality,
* linked `run()` execution.

Design decisions:

`PadLink` stores source and sink element handles plus source and sink pad names.

The pipeline still owns topology. Pads remain declarations rather than storing peer links inside `Pad`.

When links exist, `Pipeline::run()` follows the declared link chain and passes the single `std::optional<Buffer>` along that chain.

When no links exist, `Pipeline::run()` preserves the previous insertion-order behavior.

Pad links are one-to-one in this phase:

* one output pad can have at most one outgoing link,
* one input pad can have at most one incoming link,
* exact duplicate links are rejected,
* fork behavior belongs in a later `tee`-style element.

Out of scope:

* full graph scheduling,
* branching execution,
* fork/tee behavior,
* async execution,
* multi-buffer routing,
* complex negotiation,
* caps parameter negotiation,
* wildcard caps,
* multiple simultaneous streams,
* per-pad buffer queues.

Important decision:

Phase 9 uses links to drive one linear chain, but this is still not a general graph scheduler. Branching, merging, multi-buffer routing, and asynchronous execution remain future work.

## Near-Term Future Phases

> **Numbering note (read this first).** The phase numbers in *this* section are the
> **original high-level proposal** and have **diverged** from the numbering actually
> used during implementation. They are kept for strategic history only. The
> authoritative, as-built phase list is in `ROADMAP.md`,
> `docs/context/CURRENT_STATE.md`, and `docs/context/ACTIVE_PHASE.md`.
>
> As actually built, the later phases were: 19 NumPy→Torch API, 20 `NumpyToTorch`
> element, 21 logging, 22 ImGui/ImPlot `TracePlot`, 23 `leakflow_crypto` helpers,
> 24 `leakflow_plugins_crypto` (AES leakage + Pearson PoI), 25
> `PipelineSession`/control layer, 26 AES PoI numeric correctness, 27 DAG executor
> + vector-clock provenance (`LinearPipeline` → `Pipeline`), then the **live phase**
> (threaded segments, `BufferQueue`, `Sync`, `FakeLiveSrc`, liveness, cooperative
> stop, player state machine). Deferred: 28 AES CPA/report, 29 overlay plots, 30+
> Kyber. Do not treat the numbers below as current.

The following phases are proposed at a high level. They should be reviewed and detailed one at a time in `ROADMAP.md` before implementation.

## Phase 10: Core Payload Interface

Purpose:

Let `Buffer` carry real data without forcing the core to know about tensors, AES, Kyber, or file formats.

Expected concepts:

* `Payload` base class,
* optional payload storage in `Buffer`,
* safe typed payload access,
* tests with a test-only payload.

Important decision:

Do not implement `TraceSet` here. Payload support comes first.

Concept split:

```text
Pad     = named element port
Buffer  = pipeline envelope
Payload = data body
Queue   = future runtime storage for buffers
```

A `Buffer` carries zero or one `Payload`. A payload may be a bundle or batch. Future queue elements or schedulers should queue buffers, not payloads inside a buffer.

## Phase 11: Element Properties and Plugin Descriptors

Purpose:

Give elements a small property mechanism and document how compiled plugin families describe their elements.

Why:

Future elements need user-controlled settings, such as the number of PoIs to find.

Design direction:

```text
PropertyValue
  typed runtime setting value

PropertySpec
  name, default, description, unit, and validation constraint

Element
  declares supported properties and stores current values

ElementDescriptor
  describes one element type for listing and discovery

PluginDescriptor
  describes a compiled plugin family and its provided elements
```

Property values are small configuration values. They are not data transport. Traces, labels, PoI sets, reports, and tensor bundles belong in payloads.

The Phase 11 property value model should be inspectable and CLI-friendly:

* bool,
* integer,
* double,
* string,
* integer interval,
* double interval,
* integer list,
* double list,
* string list.

String enum values should be modeled as a validation constraint on a string property:

```text
method string default=pearson allowed=pearson,spearman
```

Numeric ranges should be validation constraints, not confused with interval values:

```text
poi_count int default=20 range=1..10000
sample_window int-interval default=1000..2500
```

This keeps future `leakflow-ls` output readable and future `leakflow` CLI property setting straightforward, without adding those tools in Phase 11.

## Phase 12: Core CLI, Inspect Tool, Core Plugin Library, and Tee

Purpose:

Make LeakFlow runnable and inspectable before adding tensor payloads or algorithm-specific plugins.

Expected direction:

```text
leakflow
  basic CLI runner
  supports setting element properties

leakflow-ls
  no args: list linked plugins/elements
  plugin arg: inspect one linked plugin
  element arg: inspect one linked element
  --no-colors: plain output
  --color / -C: force ANSI colors

leakflow_plugins_core
  shared library target: libleakflow_plugins_core.so
  author: Zer0Leak <edgard.lima@gmail.com>
  license: Apache-2.0
  FileSrc
  FileSink
  FakeSrc
  FakeSink
  Summary
  Tee
  Queue
```

`leakflow_plugins_core` follows the GStreamer-style distinction between plugins and elements: the plugin is one shared library, and the element factories live inside it.

Each core plugin element has its own header/source pair in the core plugin
layout, with `core_elements.hpp` kept as a convenience umbrella include.

`leakflow-ls` follows the broad `gst-inspect-1.0` output shape:

```text
leakflow-ls
  list all known features as plugin: element: description

leakflow-ls leakflow_plugins_core
  show Plugin Details and the elements in that plugin

leakflow-ls fakesink
  show Factory Details, Plugin Details, flags, pads, and properties
```

The `leakflow-ls` color scheme also follows `gst-inspect-1.0`:

```text
section headings                 yellow
field labels and property names  bright blue
feature/type names and flags     green
hierarchy connectors             magenta
property values                  cyan
```

`Tee` is the first branching element.

`Queue` stores `Buffer` objects synchronously in this phase. It is not an async scheduler.

`FileSrc` and `FileSink` move raw bytes only. NumPy, Torch, AES, and Kyber file formats stay in later plugin families.

Tee behavior:

```text
input Buffer
  -> branch Buffer copy
  -> branch Buffer copy
```

`Tee` copies the `Buffer` envelope and shares the payload pointer:

```text
Caps      copied
Metadata  copied
Payload   shared through std::shared_ptr<Payload>
```

This gives each branch independent caps and metadata while avoiding expensive payload copies.

Payload mutation rule:

```text
payload uniquely owned  -> in-place mutation may be allowed
payload shared          -> mutating element must create/set a replacement payload
```

Use `std::shared_ptr` ownership count as the first conservative uniqueness mechanism. Mutation APIs should check uniqueness before returning mutable access, for example:

```cpp
bool payload_is_unique() const;

template <typename T>
T* mutable_payload_if_unique();
```

`Mux` is intentionally not part of this phase. Muxing depends on input data semantics, synchronization, caps, and payload combination rules.

## Phase 13: Source Layout and CLI Pipeline Syntax

Purpose:

Make the command-line runner expressive enough to build small linked pipelines
without hard-coded presets, while keeping the language small and inspectable.

Expected direction:

```text
src/apps/
  leakflow/
  leakflow_ls/
  cuda_smoke/

leakflow run
  semicolon-separated statements
  element creation and naming
  property assignment
  explicit pad references
  explicit pad caps annotations
  metadata annotations
  ! links with unambiguous pad inference
```

The user-facing syntax contract lives in `docs/reference/CLI_SYNTAX.md`.

Important syntax decisions:

* `!` connects elements or pads.
* `;` finishes a statement.
* `Type@name` creates a named element instance.
* `@name` references an existing element.
* `@name.pad` references a specific pad.
* `(...)` configures element properties.
* `@name.pad[caps=TYPE; key=value]` annotates pad caps.
* `{key=value}` annotates metadata.
* Pad inference is allowed only when an element has exactly one relevant pad.
* Elements with multiple input or output pads require explicit pad references.
* Element-level caps annotations infer the only output pad.
* Element-level metadata annotations apply to all output pads.
* Pad-template metadata annotations such as `@tee.src_%u{branch=...}` apply to
  matching output pads; `%u` is a pad-template placeholder, not a general regex.

Phase 13 should not become YAML, a full graph language, a plugin registry, or a
caps negotiation phase. It gives later payload and SCA phases a practical manual
runner that can express Tee branching, property settings, caps annotations, and
metadata annotations.

## Phase 14: Modular CMake, Source, and Test Layout

Purpose:

Keep the growing project easy to navigate before adding tensor payloads or
algorithm-specific plugins.

Expected direction:

```text
CMakeLists.txt
  project setup
  global options
  add_subdirectory(...)

src/core/
  leakflow_core target

src/render/
  leakflow_render target

src/apps/
  common/
    app-only reusable helpers
  leakflow/
    leakflow executable
    leakflow_cli helper library
  leakflow_ls/
    leakflow-ls inspect executable
  cuda_smoke/
    optional CUDA smoke source

src/base/
  leakflow_base target

src/extras/
  leakflow_extras target

src/plugins/core/
  leakflow_plugins_core target

src/plugins/extras/
  leakflow_plugins_extras target

include/
  leakflow/
    core/
    render/
    base/
    extras/
    plugins/
      core/
      extras/
      base/

tests/
  core/
  apps/
  plugins/core/
```

The root CMake file should become orchestration rather than a list of every
source file and test. Directory-level CMake files inherit root options through
normal `add_subdirectory()` scope, while target-specific configuration remains
attached to each target.

This phase is intentionally behavior-preserving. It should not introduce new
pipeline syntax, elements, plugin loading, tensor payloads, AES, Kyber, YAML, or
external dependencies.

## Phase 15: LibTorch-backed Base Tensor Payload Layer

Purpose:

Introduce the required numerical data layer outside the core.

Expected direction:

```text
leakflow_core
  framework primitives only
  no LibTorch dependency

leakflow_base
  depends on leakflow_core
  depends on LibTorch
  TorchTensorPayload
  TorchTensorBundlePayload
```

`leakflow_base` is the shared data layer for future numerical plugins. It is
Torch-backed by design because future preprocessing, statistics, DNN, AES, and
Kyber plugin families should operate on one tensor substrate rather than a
custom imitation tensor type.

Phase 15 should add the payload floor only:

```text
TorchTensorPayload
  one torch::Tensor
  accepts CPU or CUDA
  rejects undefined tensors
  rejects non-strided layouts
  allows non-contiguous strided tensors

TorchTensorBundlePayload
  deterministic named collection
  stores shared_ptr<TorchTensorPayload>
  lets named tensors split into different future pipeline paths
```

Generic statistics belong in `leakflow_base`, but they should arrive in a later
phase after the payload transport is validated.

## Phase 16: Extras NumPy Payload and Loader

Purpose:

Add NumPy `.npy` arrays as an extras-library file-format concept without
exposing them through a pipeline element yet.

Expected direction:

```text
leakflow_extras
  NumpyPayload
  load_npy(path) -> NumpyPayload
```

`NumpyPayload` should be a thin wrapper around `cnpypp::NpyArray`, similar to
how `TorchTensorPayload` wraps `torch::Tensor`.

The first loader should support a small deterministic `.npy` subset through
`cnpy++`: numeric arrays, C-contiguous layout, little-endian data, and common
bool/integer/float dtypes. It should preserve the metadata exposed by `cnpy++`,
including shape, word size, labels, memory order, and raw data access.

This phase intentionally does not convert to Torch yet. Applications can call
the extras loader directly, while pipeline elements come later. `leakflow_base`
remains the Torch-backed tensor payload layer.

## Phase 17: Extras Plugin `NumpySrc`

Purpose:

Expose `load_npy(path)` through the pipeline.

Expected direction:

```text
leakflow_plugins_extras
  NumpySrc(path=...)
    output: NumpyPayload
```

The plugin should call the extras-library loader rather than owning NumPy parsing
logic itself. Tests should use tiny repository fixtures and should not depend on
local trace folders.

Important decision:

`leakflow_extras` is the application-callable library layer. The linked plugin
layer is `leakflow_plugins_extras`, built as `libleakflow_plugins_extras.so`.
`NumpySrc` is the first element in that extras plugin family.

`leakflow_base` must not depend on `leakflow_extras`; `leakflow_extras` is
layered above `leakflow_base`.

`NumpySrc` emits a buffer with:

```text
Pad caps: leakflow/numpy-array
Caps: leakflow/numpy-array
Payload: NumpyPayload
```

Generic sink pads declared as `leakflow/buffer`, such as `Summary.sink`, accept
concrete buffer caps such as `leakflow/numpy-array`. A generic source does not
automatically satisfy a concrete sink.

NumPy intrinsic facts such as dtype, shape, word size, and memory order stay in
the payload. `NumpySrc` stamps only minimal file provenance metadata, such as the
element name, source path, and file size. Semantic metadata such as traces, key,
plaintexts, AES, ChipWhisperer, power, or synchronized capture belongs to the
application, CLI metadata annotations, or later dataset-specific elements.

The current `Summary` peer behavior is intentionally left unchanged in Phase 17.
The better long-term direction is for `Summary` to ask the `Buffer`, and for the
`Buffer` to let its `Payload` complete the description. That redesign belongs in
Phase 18.

Target smoke shape:

```text
NumpySrc ! Summary ! FakeSink
```

## Phase 18: Structured Summary Documents

Purpose:

Improve Summary by separating structured description data from rendering.

Expected direction:

```text
SummaryDocument
  sections
  fields
  nested records

Buffer::describe(level)
  produces a SummaryDocument for caps, metadata, and payload

Payload::describe(section, level)
  lets payloads add payload-specific facts

renderer
  plain deterministic output
  color/theme application through a shared terminal render layer
  optional UTF-8 glyphs with ASCII fallback
```

`Summary` should describe the `Buffer` that flows through it rather than asking
a previous peer element to describe that buffer. Payload descriptions should
accept a detail level. Color should be applied by a renderer or theme after
structured data is built, not embedded in payload description logic.
`leakflow-ls` and `Summary` should share terminal style primitives, while keeping
their own domain-specific document shapes. Summary remains a pipeline element,
not a logging system.
Concurrent-safe logging belongs in a later phase.

## Phase 18.1: Source and Include Layout Consolidation

Purpose:

Keep the root project shape simple after the base, extras, and plugin libraries
were added.

Expected direction:

```text
include/
  leakflow/
    core/
    render/
    base/
    extras/
    plugins/
      core/
      extras/
      base/

src/
  core/
  render/
  base/
  extras/
  apps/
  plugins/
    core/
    extras/
    base/
```

Important decisions:

Application-facing library headers live under `include/leakflow/core`,
`include/leakflow/base`, `include/leakflow/extras`, and
`include/leakflow/render`. Plugin-facing headers live under
`include/leakflow/plugins/*` because applications should usually use plugin
features through descriptors, factories, CLI syntax, or linked plugin APIs
rather than treating every element class as general utility API.

This phase is layout-only. It keeps target names, namespaces, descriptors, CLI
behavior, and tests unchanged. The `leakflow/plugins/base` folder is reserved
for future base-backed elements, but no `leakflow_plugins_base` target or
element is added here.

## Phase 18.2: Public Include Tree Unification

Purpose:

Make public include spelling mirror the C++ namespace shape while keeping
runtime behavior unchanged.

Expected direction:

```text
#include "leakflow/core/buffer.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/extras/numpy_payload.hpp"
#include "leakflow/plugins/core/core_elements.hpp"
#include "leakflow/plugins/extras/numpy_src.hpp"
```

Important decisions:

The CMake targets keep their existing names, such as `leakflow_core`,
`leakflow_base`, `leakflow_extras`, `leakflow_plugins_core`, and
`leakflow_plugins_extras`. Only public include paths and physical header
locations change.

This phase must not add NumPy-to-Torch conversion, a `Convert` element, dynamic
plugin loading, new plugin elements, or behavior changes.

## Phase 19: NumPy-to-Torch Conversion API

Purpose:

Add direct application-callable conversion from `NumpyPayload` to
`TorchTensorPayload` using the extras NumPy payload and base Torch payload.

Expected direction:

```text
convert_numpy_to_torch(payload, options) -> TorchTensorPayload

options
  target dtype
  target device
```

CPU conversion should be tested first. CUDA can be allowed as an option when the
Torch build and runtime support it.

## Phase 20: Simple `Convert` Element

Purpose:

Expose the base conversion API through a small pipeline element before adding a
general conversion registry.

Expected direction:

```text
leakflow_plugins_extras
  Convert(to=torch, dtype=..., device=...)

NumpySrc ! Convert(to=torch) ! Summary ! FakeSink
```

This phase should support the NumPy-to-Torch path only. Dynamic pads and
pluggable conversion lookup stay later.

## Phase 21: Conversion Registry and Dynamic Pads

Purpose:

Generalize conversion so future libraries and plugins can register conversion
functions and `Convert` can select an implementation from source and target
capabilities.

Expected direction:

```text
conversion registry
  source capabilities
  destination capabilities
  conversion function

Convert
  choose registered function from pad or payload capabilities
```

Dynamic pad behavior should appear only as needed for conversion and should be
covered by focused tests.

## Phase 22: `libleakflow_aes`

Purpose:

Introduce AES helper functions outside the core.

Expected concepts:

* AES S-box helper,
* Hamming weight helper,
* first-round intermediate helpers.

AES helpers belong in `libleakflow_aes`, not `libleakflow` core.

## Phase 23: `leakflow_plugins_aes`

Purpose:

Introduce AES pipeline elements.

Expected elements:

```text
AesSyncNpyDatasetSource
AesSboxHwLabeler
AesPearsonPoiFinder
```

`AesPearsonPoiFinder` should accept properties such as PoI count, with a default value.

## Phase 24: Plot Plugin

Purpose:

Add the first headless plotting sink for traces, correlation curves, and PoI overlays.

This can live in `leakflow_plugins_plot`. It should not become a core dependency.

## Phase 25: First Full AES PoI Plot Pipeline

Purpose:

Demonstrate a complete useful pipeline on the local AES synchronized traces.

Target flow:

```text
Read one AES key folder
  key.npy
  plain_texts.npy
  traces.npy
-> compute AES leakage hypotheses
-> find PoI indexes using Pearson correlation
-> plot the result
```

This should be the first real "it all works together" milestone.

## Phase 26: Multi-Input Execution Model

Purpose:

Support elements that naturally need multiple named inputs, such as labelers, trainers, and richer attack reports.

## Phase 27: AES CPA and Report Refinement

Purpose:

Turn the AES PoI flow into a stronger classical AES validation path with key ranking and attack metrics.

## Phase 28+: Kyber / ML-KEM Direction

Purpose:

Start Kyber work only once the relevant dataset shape and traces are clear.

Expected direction:

* Kyber dataset source,
* Kyber parameter helpers,
* PC-oracle building blocks,
* SPA segmentation/message-recovery building blocks.

Kyber must stay outside the core.

## Optional Later Directions

Possible later plugin families:

* DNN training/inference,
* hardware capture,
* GUI/lab frontend,
* preset runner.

These should not be pulled forward until the manual pipeline architecture is stable.

## Architectural North Star

LeakFlow should eventually have these conceptual layers:

```text
libleakflow
  Caps
  Buffer
  Payload
  Metadata
  Element
  Pad
  Pipeline
  Execution
  Error handling

libleakflow_base
  LibTorch-backed tensor payloads
  LibTorch-backed tensor bundles
  Reusable statistics helpers

libleakflow_aes
  AES S-box
  AES leakage helpers
  AES intermediate helpers

leakflow_plugins_core
  FileSrc/FileSink
  FakeSrc/FakeSink
  Summary
  Tee

  leakflow_plugins_extras
  NumPy and Torch I/O

leakflow_plugins_sca
  Generic SCA preprocessing
  Generic PoI and statistics elements
  Generic reports

leakflow_plugins_aes
  AES dataset source
  AES labelers
  AES Pearson PoI finder
  AES attacks

leakflow_plugins_kyber
  Kyber dataset source
  Kyber oracle and SPA helpers

leakflow_plugins_plot
  Trace and PoI plotting

optional plugins
  DNN
  hardware capture
  GUI/lab frontend

cli
  leakflow
```

The core should remain small.

The core should not depend on:

* AES,
* Kyber,
* CUDA,
* LibTorch,
* ImGui,
* NumPy,
* YAML,
* Python.

## Prompt and Reproducibility Practice

Prompts are tracked under:

```text
docs/prompts/
```

The preferred prompt structure is one folder per phase:

```text
docs/prompts/phase-XX-name/
  README.md
  00-define-roadmap.prompt.md
  01-implement.prompt.md
  02-validate.md
```

This allows a developer to either copy/paste a prompt or tell Codex:

```text
Read and execute docs/prompts/phase-XX-name/01-implement.prompt.md
```

Prompt files are part of the reproducibility record, but they do not replace:

* source code,
* tests,
* commit history,
* documentation,
* validation commands.

## Important Process Rule

Before every implementation phase:

```text
1. Review the relevant design decisions with the user.
2. Update `docs/design/architecture.md` if needed.
3. Update `ROADMAP.md` and `docs/reference/PHASE_OVERVIEW.md` if the phase plan changed.
4. Implement only the requested phase.
5. Build and test.
6. Record deviations in docs/prompts/ when useful.
```

This keeps the project controlled and reproducible.
