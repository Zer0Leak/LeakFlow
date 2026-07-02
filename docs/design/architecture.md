# LeakFlow Architecture

## Status

Historical architecture notes plus broad project architecture context.

For the current dataflow/runtime design of record, prefer:

- `docs/design/dataflow_sync_model.md`
- `docs/design/runtime_execution.md`

Some older sections below intentionally describe earlier roadmap phases and may say
that Queue/threaded scheduling is future work. The implemented runtime now uses
threaded segments cut at Queue-like boundaries; `Queue` owns its `BufferQueue`
through `ThreadBoundaryRuntime`.

This document records the current LeakFlow architecture direction, including the
accepted payload, tensor, plugin, and dataflow decisions made after Phase 9 and
before Phase 10.

It exists because LeakFlow development requires a clear distinction between:

* disk file formats,
* in-memory numeric backends,
* pipeline buffers,
* payloads,
* elements,
* plugins,
* source/sink behavior,
* SCA-specific semantic objects.

## Current Project State

LeakFlow currently has the core control-plane pieces:

* `Caps`
* `Buffer`
* metadata
* `Element`
* `Pad`
* `Pipeline`
* pad-to-pad links
* linked linear execution

The current pipeline can represent a linked linear chain, but it is not yet a full graph scheduler.

Phase 9 intentionally keeps links one-to-one:

* one output pad has at most one outgoing link,
* one input pad has at most one incoming link,
* fork behavior belongs in a future tee-style element,
* merge behavior belongs in a future join/aggregator-style element.

## Local Data And Paper Context

The current local trace root is:

```text
traces/
```

The current real trace dataset is an AES ChipWhisperer synchronized dataset stored as NumPy `.npy` files:

```text
traces/aes/sync/
  README.md
  aes_sync_poi/
  aes_sync_attack/
```

Each AES key folder contains:

```text
key.npy
plain_texts.npy
traces.npy
```

The documented array formats are:

```text
key.npy          NumPy uint8   shape (16,)
plain_texts.npy  NumPy uint8   shape (traces_per_key, 16)
traces.npy       NumPy float32 shape (traces_per_key, num_samples)
```

In the current capture configuration, `num_samples = 5000`.

Dataset sizes:

```text
aes_sync_poi     50 keys x 2,000 traces/key  = 100,000 traces
aes_sync_attack  30 keys x 10,000 traces/key = 300,000 traces
```

The saved traces are sign-flipped relative to the raw ChipWhisperer waveform, so the saved trace convention is intended to make higher relative saved trace values correspond to higher target power consumption.

The current local paper root is:

```text
papers/
```

Papers currently in scope for architecture planning:

```text
Ueno et al. 2021
  Curse of Re-encryption: generic Power/EM analysis on FO-style KEMs.

Wang et al. 2024
  SPA-GPT: automated trace segmentation for public-key SPA, including Kyber.

Zhou et al. 2024
  LLM-assisted automated SPA on public-key cryptosystems, including Kyber.

Chen et al. 2025
  DL profiled PC-oracle attack against Kyber under anti-tampering covers.

Rezaeezade et al. 2025
  DL blind SCA and multi-point cluster-based labeling on AES, ASCON, and Kyber.
```

These papers should inform future SCA building blocks and plugin families, but they must not pull AES, Kyber, Torch, CUDA, or DNN implementation into `leakflow_core`.

### Future Kyber Dataset Shape

Kyber / ML-KEM traces are not present yet.

A future Kyber dataset should preserve the same broad pattern as the AES data. A flat tensor-bundle layout is likely the best first convention:

```text
traces/kyber/<capture-kind>/<dataset-name>/
  pvt_keys.npy
  pub_keys.npy
  ciphertexts.npy
  traces.npy
```

Depending on the capture and attack target, additional files may exist:

```text
messages.npy
decaps_messages.npy
shared_secrets.npy
coins.npy
operation_labels.npy
segment_boundaries.npy
oracle_labels.npy
metadata.json
```

A per-case folder layout should also be acceptable if it is easier for capture tooling:

```text
traces/kyber/<capture-kind>/<dataset-name>/
  case_000001/
    pvt_key.npy
    pub_key.npy
    ciphertext.npy
    traces.npy
```

The exact Kyber file names can be revised when real traces arrive. The important design rule is that the source element should output a named tensor bundle, so the pipeline does not assume AES-specific names such as `key` and `plaintexts`.

## Terminology

Use these terms consistently.

### Trace

A trace is one complete captured waveform from one cryptographic operation or capture event.

For a trace matrix with shape:

```text
trace_count x sample_count
```

one row is one trace.

Example:

```text
traces[0]
```

means the first complete captured waveform.

### Sample

A sample is one time-indexed numeric value inside a trace.

Example:

```text
traces[0][733]
```

means sample index `733` inside trace index `0`.

### Trace index

A trace index selects one trace from a batch.

Example:

```text
trace_index = 10
```

selects the eleventh trace.

### Sample index

A sample index selects one time sample inside each trace.

Example:

```text
sample_index = 733
```

selects one point in time inside a trace.

### Point of Interest

A PoI is usually represented as one or more sample indexes.

If a PoI finder selects 100 PoIs from traces with 5000 samples, it selects:

```text
100 sample indexes out of 5000 samples
```

It does not select 100 trace indexes.

### Avoid "measurement"

Avoid using `measurement` as a core term because it can ambiguously mean:

* one full trace, or
* one sample inside a trace.

Use `trace` and `sample` instead.

## File Format vs In-Memory Backend

A file format is not the same thing as an in-memory backend.

### NumPy `.npy`

`.npy` is a NumPy disk file format.

It is not a Torch format.

Use precise wording:

```text
NumPy .npy input
```

or:

```text
.npy file parsed and converted into Torch tensor payload
```

Do not say:

```text
Torch NPY format
```

### Torch `.pt` / `.pth`

`.pt` and `.pth` are Torch/PyTorch-style serialized artifact formats.

These are appropriate for intermediate saved tensors, tensor bundles, models, or future reusable artifacts.

### Internal numeric backend

For serious numeric SCA pipelines, the preferred internal backend will be Torch tensors.

That means:

```text
Input can be .npy.
Internal payload can be torch::Tensor.
Output/intermediate cache can be .pt/.pth.
```

Example:

```text
key.npy, plain_texts.npy, traces.npy
  -> AesNpyToTorchSource
  -> TorchTensorBundlePayload
  -> TorchSaveSink
  -> key_01.pt
```

Later:

```text
key_01.pt
  -> TorchLoadSource
  -> PoIFinder
  -> MCLabel
  -> GuessingEntropyReport
```

## Core Design Rule

The core must remain algorithm-agnostic and backend-agnostic.

`leakflow_core` may know:

* `Caps`
* metadata
* `Payload`
* `Buffer`
* `Element`
* `Pad`
* links
* pipeline execution
* basic errors

`leakflow_core` must not know:

* `torch::Tensor`
* LibTorch
* NumPy `.npy`
* Torch `.pt` / `.pth`
* AES
* Kyber / ML-KEM
* CUDA implementation details
* HDF5
* YAML experiment syntax
* GUI concepts

This keeps the kernel small and lets Torch become first-class without forcing every core user to compile against LibTorch.

## Torch Design Rule

Torch is expected to become the main practical numeric backend for serious SCA workflows.

However, Torch belongs in a Torch base layer, not in the absolute core.

Recommended future layers:

```text
leakflow
  Command line CLI tool to start a Pipeline flow from command line.

leakflow-ls
  Command line CLI tool to list and find plugins.
  Must allow filter by name, keywords, etc..

libleakflow
  Core pipeline kernel.
  Application can Link with and start a Pipeline.
  The application can have hooks to modify or inspect the Buffer content.
  Lightweight Logger API.
  No Torch dependency.

leakflow_plugins_core
  Shared library target.
  Unix-like output file: libleakflow_plugins_core.so.
  FileSrc
  FileSink
  FakeSrc
  FakeSink
  Summary - Renders a structured description of the Buffer flowing through it.
  A summary level is passed. Buffer and payload description data are separate
  from terminal colors and glyphs.
  Tee
  Queue

libleakflow_base
  Official Torch-backed numeric layer.
  Depends on LibTorch.
  Defines Torch payloads and Torch helpers.
  Utilitary statistical methods will live here.

leakflow_extras
  Optional extra file-format payload helpers.
  Layered above libleakflow_base.
  Depends on cnpy++ for NumPy .npy loading.
  NumpyPayload wraps cnpypp::NpyArray.

leakflow_plugins_extras
  Extras file-format and bridge elements:
    NumpySrc
    future NumPy-to-Torch conversion elements

NumpySrc declares concrete `leakflow/numpy-array` source caps and emits buffers
with `Caps("leakflow/numpy-array")`. Generic sink pads declared as
`leakflow/buffer` may accept concrete buffer caps so summary and sink elements
can consume typed buffers without becoming NumPy-specific.

leakflow_plugins_sca
  Generic SCA payloads and elements:
    PoIFinder
    TTestPoiFinder
    CorrelationPoiFinder
    WelchTTest
    TraceCrop
    TraceNormalize
    TraceAlign
    TraceSegmenter
    SegmentClassifier
    HammingWeightLabeler
    MCLabel
    DistinguisherTrainer
    DistinguisherInference
    KeyRanker
    GuessingEntropyReport

leakflow_plugins_dl
  Optional deep-learning SCA elements:
    TorchModelTrain
    TorchModelInfer
    NoisyLabelTrainer
    CnnDistinguisher
    AspnDistinguisher

leakflow_plugins_aes
  AES-specific sources, leakage models, labelers, and attacks.

leakflow_plugins_kyber
  Kyber/ML-KEM-specific sources, chosen-ciphertext helpers,
  oracle adapters, message recovery helpers, and attacks.

leakflow_plugins_plot
  Plotters using ImGui and ImPlot
  Plot to screen (live inspection)

leakflow_plugins_hardware
  Live capture and acquisition adapters.
```

This is inspired by GStreamer-style layering, but adapted for LeakFlow.

The repository filesystem mirrors these boundaries:

```text
include/leakflow/core    core application-facing headers
include/leakflow/render  shared terminal rendering headers
include/leakflow/base    Torch-backed base application-facing headers
include/leakflow/extras  extras helper application-facing headers
include/leakflow/plugins plugin-facing headers grouped by plugin family
src/core                 core implementation
src/render               shared terminal rendering implementation
src/base                 Torch-backed base implementation
src/extras               extras helper implementation
src/apps                 command-line applications
src/plugins/*            linked plugin implementations
```

Public include spelling follows the unified `leakflow/...` tree. For example,
applications and tests include `leakflow/base/...` and plugin-linked code
includes `leakflow/plugins/core/...`; CMake supplies the single public include
root.

## Plugin vs Element

An element is one processing block used inside a pipeline.

Examples:

```text
AesNpyToTorchSource
PoIFinder
Tee
MCLabel
TorchSaveSink
GuessingEntropyReport
```

A plugin is a distribution or registration unit that provides one or more elements.

Examples:

```text
leakflow_plugins_torch_io
  provides:
    NpyToTorchSource
    NpyFolderToTorchBundleSource
    TorchLoadSource
    TorchSaveSink

leakflow_plugins_sca
  provides:
    PoIFinder
    MCLabel
    DistinguisherTrainer
    GuessingEntropyReport

leakflow_plugins_aes
  provides:
    AesSyncNpyDatasetSource
    AesSboxHwLabeler
    AesCpaAttack

leakflow_plugins_kyber
  provides:
    KyberNpyToTorchSource
    KyberPcOracleAdapter
    KyberMessageRecovery
```

A plugin should describe:

* plugin name,
* plugin author or owner,
* plugin license,
* plugin purpose,
* provided elements,
* each element's input and output caps,
* each element's configurable properties,
* useful search keywords.

Search keywords should make it possible to discover elements by intent, for example:

```text
aes
kyber
ml-kem
plot
label
blind
profiled
poi
segment
oracle
attack-report
```

Early LeakFlow may implement these as normal compiled libraries or statically registered components. Dynamic plugin loading can come later.

## Element Properties and Descriptors

Elements need user-controlled settings, but those settings must remain easy to inspect, validate, document, and eventually set from a CLI.

Use these concepts separately:

```text
PropertyValue
  The current runtime value of one setting.

PropertySpec
  The declaration for one setting:
  name, value type, default value, description, unit, and validation constraint.

PropertyEffect
  The declared behavior of changing one setting:
  whether the change is UI-only, sink-display, metadata-output, payload-output,
  caps-output, or lifecycle-affecting, plus invalidation scope and affected
  output pads.

Element
  The runtime processing object.
  It declares supported properties and stores current values.

ElementDescriptor
  Static metadata for one element type:
  name, purpose, pads, properties, and keywords.

PluginDescriptor
  Static metadata for one compiled plugin family:
  plugin name, owner, author, license, purpose, keywords, and provided element descriptors.
```

This follows the spirit of GStreamer-style inspectable element properties, but LeakFlow should keep the implementation small and C++23-native.

### Property Value Model

Properties are small configuration values.

They are not payloads, and they must not become a second data transport path.

Good property examples:

```text
poi_count = 20
method = pearson
sample_window = 1000..2500
poi_indexes = [12, 40, 91]
selected_bytes = [0, 1, 2, 15]
```

Bad property examples:

```text
traces = huge tensor
labels = full label matrix
poi_scores = thousands of correlation values
model_weights = trained DNN artifact
```

Those large experiment objects belong in `Payload`.

The Phase 11 property value vocabulary should be:

```text
bool
integer
double
string
integer interval
double interval
integer list
double list
string list
```

In C++23 terms, the intended shape is a small `std::variant`, not `std::any`.

Conceptual sketch:

```cpp
struct IntInterval {
    std::int64_t begin;
    std::int64_t end;
};

struct DoubleInterval {
    double begin;
    double end;
};

using IntList = std::vector<std::int64_t>;
using DoubleList = std::vector<double>;
using StringList = std::vector<std::string>;

using PropertyValue = std::variant<
    bool,
    std::int64_t,
    double,
    std::string,
    IntInterval,
    DoubleInterval,
    IntList,
    DoubleList,
    StringList>;
```

This keeps values printable, comparable in tests, parseable from strings later, and easy to show in future `leakflow-ls` output.

### Property Constraints

Constraints describe valid values. They are separate from values.

Use numeric range constraints for scalar numeric properties:

```text
poi_count int default=20 range=1..10000
threshold double default=0.0 range=0.0..1.0
```

Use interval values for actual windows or operation spans:

```text
sample_window int-interval default=1000..2500
time_window double-interval default=0.0..2.5
```

Use a string enum constraint for properties with named choices:

```text
method string default=pearson allowed=pearson,spearman
leakage_model string default=sbox_hw allowed=sbox_hw
```

A string enum is a constraint on a string property, not a separate value type.

List values should represent short configuration lists:

```text
poi_indexes int-list default=[]
selected_bytes int-list default=0,1,2,15
plot_channels int-list default=0
columns string-list default=traces,plaintexts,key
```

Large lists of analysis results should remain payloads.

### Property Change Effects

Properties can be controlled by CLI syntax, graph controls, standalone control
panels, or future element-local widgets. All property-backed controls should use
the element property as the source of truth.

Each property may declare a `PropertyEffect`:

```text
ui-only
sink-display
metadata-output
payload-output
caps-output
lifecycle
```

and an invalidation scope:

```text
none
element-ui
downstream
full-pipeline
```

The effect describes what a property change means, not where the element sits in
the pipeline. A property on a source can still be `ui-only` if it only changes
how the source is displayed in a UI. A property on a sink can be dataflow
affecting if it changes produced buffers in a future sink/source hybrid.

Rules:

* `ui-only` changes redraw UI only and do not create buffers.
* `sink-display` changes update how already observed data is shown and do not
  rerun upstream dataflow.
* `metadata-output` changes produce new buffer metadata and therefore affect
  downstream elements.
* `payload-output` changes produce new payload content or shape and therefore
  affect downstream elements.
* `caps-output` changes may change output caps; downstream links must be
  revalidated before rerun.
* `lifecycle` changes require restart or full-pipeline lifecycle handling.

Changing metadata is a dataflow change when downstream elements can observe it.
It should eventually create a new `Buffer` envelope with updated metadata, not a
side-channel update.

Example:

```text
AesLeakage.channels: [HW(m)] -> [HW(m), HW(y)]
effect: payload-output
scope: downstream
affected output: leakage
```

This changes target payload shape and target metadata. A future incremental
executor should rerun `AesLeakage.leakage` and downstream consumers such as
`PearsonPoiFinder`, annotation conversion, and plot sinks using cached/latest
inputs. It should not reload unrelated upstream traces.

### Descriptor Use

Descriptors should make future tools straightforward.

Future `leakflow-ls` should be able to show something like:

```text
AesPearsonPoiFinder
  purpose: find PoI indexes using Pearson correlation
  keywords: aes, poi, pearson, correlation
  properties:
    poi_count int default=20 range=1..10000
    method string default=pearson allowed=pearson
```

Future `leakflow` CLI runner should be able to parse property assignments using the descriptor:

```text
poi_finder.poi_count=50
poi_finder.method=pearson
crop.sample_window=1000..2500
plot.poi_indexes=120,488,901
```

Phase 11 should not add these executables yet. It should only build the property and descriptor model that will make them possible.

## Core CLI, Inspect Tool, Core Plugin Library, and Tee

After element properties and descriptors exist, the next useful step is to make LeakFlow runnable and inspectable with a linked core plugin library.

Phase 12 should introduce:

```text
leakflow
  Basic CLI runner for small preset pipelines.
  Supports setting element properties using the Phase 11 property model.

leakflow-ls
  Inspect tool over linked descriptors.
  With no arguments, lists known plugins/elements.
  With a plugin argument, inspects one linked plugin descriptor.
  With an element argument, inspects one linked element descriptor.
  Supports --no-colors for plain output.
  Supports --color and -C to force ANSI colors.

leakflow_plugins_core
  Shared library target.
  Expected Unix-like file name: libleakflow_plugins_core.so.
  Author: Zer0Leak <edgard.lima@gmail.com>.
  License: Apache-2.0.
  FileSrc
  FileSink
  FakeSrc
  FakeSink
  Summary
  Tee
  Queue
```

This phase should link `leakflow` and `leakflow-ls` against `leakflow_plugins_core`.

`leakflow_plugins_core` follows the GStreamer-style distinction between plugins and elements:

```text
plugin
  leakflow_plugins_core

elements
  FileSrc
  FileSink
  FakeSrc
  FakeSink
  Summary
  Tee
  Queue
```

The plugin is one shared library. The listed names are element types provided by that plugin, not separate plugins.

Each core plugin element should live in its own header/source pair, with
headers under `include/leakflow/plugins/core/` and sources under
`src/plugins/core/`. The umbrella header `core_elements.hpp` may include all of
them for convenience.

The inspect output should follow the broad `gst-inspect-1.0` shape:

```text
leakflow-ls
  plugin: element: description

leakflow-ls leakflow_plugins_core
  Plugin Details
  element list
  feature counts

leakflow-ls fakesink
  Factory Details
  Plugin Details
  element hierarchy
  flags
  pad templates
  pads
  properties
```

The inspect color scheme should mirror the useful parts of `gst-inspect-1.0`:

```text
section headings                 yellow
field labels and property names  bright blue
feature/type names and flags     green
hierarchy connectors             magenta
property values                  cyan
```

It should not add dynamic plugin loading, a plugin registry, YAML, tensors, AES, Kyber, or format-specific file I/O.

### Core Plugin Elements

`FileSrc` reads one file as raw bytes and emits a byte payload.

`FileSink` writes raw byte payloads to a file, with a textual fallback for non-byte buffers.

These are not NumPy, Torch, AES, Kyber, or dataset-specific loaders.

`FakeSrc` produces simple test buffers.

`FakeSink` consumes buffers for validation.

`Summary` renders a structured description of the buffer that flows through it,
using a summary level property. `Buffer` describes caps and metadata, while
payloads can add payload-specific fields. The structured description is
independent of color; terminal renderers apply colors and UTF-8 or ASCII glyphs.

`Tee` forks one input buffer to multiple downstream branches.

`Tee` is generic because it does not inspect or combine payload data.

`Queue` stores `Buffer` objects synchronously. It is not an async queue or scheduler in this phase.

### Why Mux Is Not Phase 12

`Mux` should not be added in Phase 12.

Muxing is data-specific. A mux element must answer questions such as:

```text
Which inputs are required?
When is the mux ready?
How are buffers synchronized?
How are caps chosen?
How are payloads combined?
What happens when one input is missing?
Is joining based on trace index, timestamp, key id, batch id, or something else?
```

Those questions depend on the concrete dataset and analysis flow.

Therefore `Mux` belongs in a later phase after a specific data shape makes the semantics clear.

### Tee Buffer Semantics

`Tee` turns one incoming buffer into multiple outgoing buffers:

```text
input Buffer
  -> branch A Buffer
  -> branch B Buffer
```

`Tee` should copy the `Buffer` envelope and share the payload pointer:

```text
Caps      copied
Metadata  copied
Payload   shared through std::shared_ptr<Payload>
```

Copying the `Buffer` envelope keeps branches independent.

For example:

```text
FakeSrc -> Tee -> Summary
              -> Normalize
```

The `Summary` branch can add summary metadata without affecting the `Normalize` branch.

The `Normalize` branch can replace its payload without changing the `Summary` branch's payload pointer.

The expensive payload object is not deep-copied by `Tee`.

This is important for large traces, such as a future 200 MB tensor payload.

### Payload Mutation Semantics

Payload mutation should be allowed in normal single-path pipelines when the payload is not shared:

```text
TorchFileSrc -> Normalize -> ...
```

In that case, forcing a deep copy would be wasteful.

After `Tee`, the payload is shared:

```text
TorchFileSrc -> Tee -> Summary
                -> Normalize
```

In that case, in-place mutation would affect another branch.

Adopt this rule:

```text
payload uniquely owned  -> in-place mutation may be allowed
payload shared          -> mutating element must create/set a replacement payload
```

Use `std::shared_ptr<Payload>` ownership count as the first conservative uniqueness mechanism.

The check is conservative: extra local `std::shared_ptr` copies can increase the count even inside one element. Therefore mutation APIs should check uniqueness before returning mutable access.

Conceptual API direction:

```cpp
bool payload_is_unique() const;

template <typename T>
T* mutable_payload_if_unique();
```

Conceptual behavior:

```cpp
template <typename T>
T* Buffer::mutable_payload_if_unique()
{
    if (!payload_ || payload_.use_count() != 1) {
        return nullptr;
    }

    return dynamic_cast<T*>(payload_.get());
}
```

Read-only typed access can still use shared pointers.

Mutable access should avoid taking another `std::shared_ptr` before the uniqueness check.

If `mutable_payload_if_unique<T>()` fails, a transforming element should create a new payload and call `set_payload(...)`.

Do not require a generic `Payload::clone()` yet. Deep-copy semantics depend on real payload types such as tensors, reports, model artifacts, and future hardware buffers.

## Payload Decision

The pipeline should be `Buffer`-centric.

### Pad, Buffer, Payload, and Queue

Use these concepts separately.

```text
Pad
  A named input or output port on an element.
  It declares what can enter or leave an element.
  It does not own data.

Buffer
  The pipeline envelope that flows between elements.
  It carries caps, metadata, and zero or one payload.

Payload
  The data body inside a buffer.
  It can be a tensor, tensor bundle, PoI set, label set,
  attack report, model artifact, or another future data object.

Queue
  A future runtime/scheduler concept.
  It stores buffers over time, not payloads inside one buffer.
```

The core rule is:

```text
Buffer has zero or one Payload.
Payload may internally be a bundle or batch.
Queues, when introduced, store Buffers.
Pads connect elements and declare caps; they do not store payloads.
```

This avoids turning `Buffer` into a large container of unrelated objects and keeps future streaming/async behavior separate from the data envelope.

A `Buffer` carries:

```text
Caps
metadata
optional payload
```

The payload is an abstract object owned through a smart pointer.

Core interface intent:

```cpp
namespace leakflow {

class Payload {
public:
    virtual ~Payload() = default;
    virtual std::string type_name() const = 0;
};

}
```

`Buffer` should eventually support:

```cpp
bool has_payload() const;
std::shared_ptr<Payload> payload() const;
void set_payload(std::shared_ptr<Payload> payload);

template <typename T>
std::shared_ptr<T> payload_as() const;
```

`payload_as<T>()` should use safe downcasting, for example `std::dynamic_pointer_cast<T>`.

`set_payload(nullptr)` should be allowed and should clear the payload.

## Why Payload Before TraceSet

The next architectural problem is not just "how to store traces."

The next architectural problem is:

```text
How does a Buffer carry real data through elements?
```

Future objects flowing through the pipeline include:

* Torch tensors,
* tensor bundles,
* PoI sets,
* label probability matrices,
* attack reports,
* metric reports,
* model artifacts,
* segment sets.

A `TraceSet` class alone does not solve this.

Therefore the Phase 10 implementation decision was:

```text
Phase 10: Core Payload Interface
```

not:

```text
Phase 10: TraceSet
```

A semantic `TraceSet` may still exist later, but it should not be the core design center.

## Torch Payloads

Torch-specific payloads belong in the Torch base layer.

### TorchTensorPayload

Represents one tensor.

Conceptual structure:

```cpp
class TorchTensorPayload final : public Payload {
public:
    explicit TorchTensorPayload(torch::Tensor tensor);

    const torch::Tensor& tensor() const;
    torch::Tensor& tensor();

    std::string type_name() const override;

private:
    torch::Tensor tensor_;
};
```

Example caps for `traces.npy` after conversion:

```text
tensor/torch; dtype=float32; rank=2; shape=2000,5000
```

Example metadata:

```text
source.format = npy
source.path = traces.npy
tensor.backend = torch
tensor.device = cpu
trace_count = 2000
sample_count = 5000
```

### TorchTensorBundlePayload

Represents a named set of tensors.

This is useful for datasets such as AES trace captures.

Example bundle:

```text
key         -> torch::Tensor uint8   [16]
plaintexts  -> torch::Tensor uint8   [trace_count, 16]
traces      -> torch::Tensor float32 [trace_count, sample_count]
```

Example caps:

```text
sca/dataset; kind=aes; backend=torch
```

This avoids forcing the source to emit three unrelated buffers before the graph supports multi-input routing well.

## SCA Payload Vocabulary

The following payload names are future design vocabulary. They should not be implemented in Phase 10 unless the roadmap explicitly asks for them.

### DatasetBundlePayload

Represents a named set of arrays or tensors plus dataset-level metadata.

Common names:

```text
traces
sample_rate
trace_count
sample_count
```

AES-specific names:

```text
key
plaintexts
```

Kyber-specific names may include:

```text
pvt_keys
pub_keys
ciphertexts
messages
decaps_messages
shared_secrets
coins
```

### PoiSetPayload

Represents selected sample positions and optional scores.

```text
indices -> int64 [poi_count]
scores  -> float32 [poi_count]
```

Example caps:

```text
sca/poi-set; index_type=sample; count=100
```

### SegmentSetPayload

Represents intervals inside one or more traces.

```text
starts -> int64 [segment_count]
ends   -> int64 [segment_count]
labels -> optional int64 or string [segment_count]
scores -> optional float32 [segment_count]
```

Example caps:

```text
sca/segment-set; index_type=sample; source=trace
```

This is needed for public-key SPA flows where a long trace is split into operation segments before classification.

### LabelPayload

Represents discrete labels produced by a leakage model, oracle dataset builder, clusterer, or classifier.

```text
labels -> int64 [trace_count]
```

Example caps:

```text
sca/labels; label_space=hamming-weight
sca/labels; label_space=pc-oracle
sca/labels; label_space=operation
```

### LabelProbabilityPayload

Represents class probabilities or scores.

```text
probabilities -> float32 [trace_count, class_count]
```

Example caps:

```text
sca/label-probabilities; method=mc
sca/label-probabilities; method=cnn
sca/label-probabilities; method=aspn
```

### DistinguisherPayload

Represents a trained statistical or neural distinguisher.

Examples:

```text
reduced-template distinguisher
CNN distinguisher
ASPN distinguisher
DNN classifier trained with noisy labels
```

Example caps:

```text
sca/distinguisher; task=pc-oracle; backend=torch
sca/distinguisher; task=operation-classification; backend=torch
```

### OracleResponsePayload

Represents responses from an inferred oracle, such as a plaintext-checking oracle.

```text
responses -> int64 [query_count]
scores    -> optional float32 [query_count]
```

Example caps:

```text
sca/oracle-response; oracle=plaintext-checking
sca/oracle-response; oracle=decryption-failure
```

### OperationSequencePayload

Represents a recovered operation sequence from segmented traces.

```text
operations -> int64 or string [operation_count]
scores     -> optional float32 [operation_count]
```

Example caps:

```text
sca/operation-sequence; algorithm=kyber; target=poly_frommsg
```

### AttackReportPayload

Represents attack or validation outputs.

Common fields:

```text
best_key_guess
key_rank
guessing_entropy
success_rate
traces_used
queries_used
recovered_message
recovered_operation_sequence
```

Example caps:

```text
sca/attack-report; target=aes
sca/attack-report; target=kyber
```

## Input and Output Elements

### NpyToTorchSource

Reads one NumPy `.npy` file and outputs one Torch tensor payload.

Input:

```text
some_array.npy
```

Output:

```text
Buffer
  Caps: tensor/torch; dtype=...; rank=...; shape=...
  Payload: TorchTensorPayload
```

This element reads NumPy format but outputs Torch.

The name means:

```text
NumPy .npy -> Torch tensor
```

It does not mean `.npy` is a Torch format.

### AesNpyToTorchSource

Reads one AES ChipWhisperer key folder:

```text
key.npy
plain_texts.npy
traces.npy
```

Output:

```text
Buffer
  Caps: sca/dataset; kind=aes; backend=torch
  Payload: TorchTensorBundlePayload
    key
    plaintexts
    traces
```

This is AES-specific and should not live in `leakflow_core`.

### AesSyncNpyDatasetSource

Reads the current local AES synchronized dataset root:

```text
traces/aes/sync/aes_sync_poi
traces/aes/sync/aes_sync_attack
```

Output options:

```text
one Buffer per key folder
one Buffer for the whole dataset
```

The first option is simpler for streaming and batching. The second option is useful for whole-dataset summary and validation sinks.

Example caps:

```text
sca/dataset; algorithm=aes; acquisition=chipwhisperer-sync; backend=torch
```

### KyberNpyToTorchSource

Reads one future Kyber / ML-KEM capture folder.

Tentative input:

```text
pvt_keys.npy
pub_keys.npy
ciphertexts.npy
traces.npy
```

Optional input:

```text
messages.npy
decaps_messages.npy
shared_secrets.npy
coins.npy
operation_labels.npy
segment_boundaries.npy
oracle_labels.npy
```

Output:

```text
Buffer
  Caps: sca/dataset; algorithm=kyber; backend=torch
  Payload: TorchTensorBundlePayload
    pvt_keys
    pub_keys
    ciphertexts
    traces
    optional named tensors
```

This is Kyber-specific and should not live in `leakflow_core`.

### TorchLoadSource

Reads Torch/PyTorch serialized artifacts:

```text
.pt
.pth
```

Output:

```text
TorchTensorPayload
```

or:

```text
TorchTensorBundlePayload
```

depending on what was saved.

### TorchSaveSink

Receives Torch payloads and saves them to Torch/PyTorch serialized artifacts.

Input:

```text
TorchTensorPayload
TorchTensorBundlePayload
```

Output on disk:

```text
.pt
.pth
```

Preferred extension for intermediate artifacts:

```text
.pt
```

## Paper-Informed Attack Families

The local papers imply that LeakFlow needs reusable SCA building blocks rather than one monolithic "AES attack" or "Kyber attack" implementation.

For AES and Kyber only, the future architecture should be able to express these attack families:

```text
AES classical known-plaintext SCA
  CPA / correlation-style validation.

AES blind or weakly supervised SCA
  PoI selection, MC labeling, noisy-label DNN training, key ranking.

Kyber FO / PC-oracle SCA
  chosen-ciphertext query construction, decapsulation/re-encryption trace handling,
  PC-oracle distinguisher training/inference, adaptive key recovery report.

Kyber profiled DNN PC-oracle SCA
  clone or same-device profiling data, TVLA/T-test preprocessing,
  CNN/ASPN-style distinguisher, oracle response reporting.

Kyber public-key SPA / horizontal SCA
  long-trace segmentation, envelope or feature enhancement,
  segment classification, operation-sequence or message recovery.
```

Do not add RSA, ECC, ASCON, Saber, NTRU, BIKE, SIKE, or other algorithms to LeakFlow just because they appear in the papers. They can inform generic SCA elements, but the algorithm-specific plugin target remains AES and Kyber / ML-KEM.

## Building Block Inventory

This inventory is future design vocabulary. It is intentionally broader than Phase 10.

### Generic SCA Elements

These belong in `leakflow_plugins_sca` unless a later phase finds a better split:

```text
TraceCrop
  Selects sample windows or operation intervals.

TraceNormalize
  Mean/std normalization, centering, scaling, sign handling.

TraceAlign
  Alignment by trigger, peak, correlation, dynamic time warping, or known offsets.

TraceResample
  Optional resampling/downsampling when a later phase needs it.

TTestPoiFinder
  Welch t-test / TVLA-style PoI discovery.

CorrelationPoiFinder
  Correlation-based PoI discovery for known intermediate variables.

SostPoiFinder
  Sum-of-squared t-values style leakage interval discovery.

PoiSelect
  Select top-k PoIs or merge PoI sets.

PoiGather
  Reduce traces to selected PoI columns.

Segmenter
  Produces segment boundaries from one or more traces.

EnvelopeFeature
  Produces envelope-like features for public-key SPA segmentation.

ClusterLabels
  Clustering-based labels, for example GMM or k-means.

HammingWeightLabeler
  Produces Hamming-weight labels from known intermediate values.

MCLabel
  Multi-point cluster-based labeling for blind SCA.

TheoreticalDistribution
  Precomputes key-hypothesis distributions for blind SCA.

EmpiricalDistribution
  Builds distributions from labeled or predicted traces.

DistinguisherTrainer
  Trains a statistical or neural side-channel distinguisher.

DistinguisherInference
  Runs a trained distinguisher on attack traces.

OracleAdapter
  Converts distinguisher outputs into oracle responses.

KeyRanker
  Ranks key hypotheses from scores or likelihoods.

GuessingEntropyReport
  Produces key-rank and guessing-entropy metrics.

DatasetSummarySink
  Reports shapes, caps, metadata, and basic sanity checks.
```

### Optional Deep-Learning Elements

These belong in `leakflow_plugins_dl` or another explicit optional layer that depends on the selected DNN backend:

```text
TorchModelTrain
TorchModelInfer
CnnDistinguisher
AspnDistinguisher
NoisyLabelTrainer
TrainValidationSplit
ModelMetricReport
```

The design should allow these elements to use Torch later, but `leakflow_core` must not depend on Torch.

### AES Plugin Elements

These belong in `leakflow_plugins_aes`:

```text
AesNpyToTorchSource
AesSyncNpyDatasetSource
AesSbox
AesSboxHwLabeler
AesRoundOneIntermediate
AesCpaAttack
AesKnownPlaintextKeyRanker
AesBlindDistribution
AesAttackReport
```

AES is the primary validation target. The first AES attack path should stay simple: known plaintext, first-round S-box intermediate, Hamming-weight leakage model, CPA/key ranking, and a report.

### Kyber Plugin Elements

These belong in `leakflow_plugins_kyber`:

```text
KyberNpyToTorchSource
KyberParams
KyberCiphertextBuilder
KyberReferenceCiphertextSource
KyberChosenCiphertextGenerator
KyberDecapsulationTraceCrop
KyberReencryptionTraceCrop
KyberPcOracleDatasetBuilder
KyberPcOracleAdapter
KyberDistinguisherAttack
KyberAdaptiveKeyRecovery
KyberPolyFromMsgTraceCrop
KyberMessageSegmenter
KyberOperationSequenceClassifier
KyberMessageRecovery
KyberAttackReport
```

Kyber-specific elements may know ML-KEM/Kyber parameter sets, ciphertext structure, message bits, decapsulation phases, and helper logic needed to map oracle or operation outputs back to message/key-recovery state.

They must not live in `leakflow_core`.

### Plot And Hardware Elements

These are useful but should remain optional:

```text
TracePlotSink
PoiPlotSink
SegmentPlotSink
OperationSequencePlotSink
ChipWhispererCaptureSource
OscilloscopeCaptureSource
```

Plotting and live capture are not required for Phase 10 and should not block payload work.

## Full Example: Future AES CPA Flow

This should be the first classical SCA validation path because AES is the primary validation target and the local AES dataset already exists.

Input root:

```text
traces/aes/sync/aes_sync_attack/
```

Logical pipeline:

```text
AesSyncNpyDatasetSource
  -> TraceNormalize
  -> AesSboxHwLabeler
  -> CorrelationPoiFinder
  -> AesCpaAttack
  -> GuessingEntropyReport
```

Expected payload flow:

```text
DatasetBundlePayload
  key
  plaintexts
  traces

LabelPayload
  hamming_weight_labels

PoiSetPayload
  selected sample indexes

AttackReportPayload
  best_key_guess
  key_rank
  guessing_entropy
  traces_used
```

This flow needs known plaintexts and should not require DNNs.

## Full Example: Future AES MC Labeling Flow

The motivating future pipeline is an AES side-channel flow using real ChipWhisperer `.npy` data.

### Dataset

Input folder:

```text
key_01/
  key.npy
  plain_texts.npy
  traces.npy
```

The source reads:

```text
key.npy          uint8   [16]
plain_texts.npy  uint8   [trace_count, 16]
traces.npy       float32 [trace_count, sample_count]
```

For the observed dataset, `sample_count = 5000`.

### Intended logical pipeline

```text
AesNpyToTorchSource
  -> Tee(dataset)
       -> PoIFinder
            -> Tee(poi-set)
                 -> TorchSaveSink(poi-set)
                 -> MCLabel.poi
       -> MCLabel.dataset

MCLabel
  -> Tee(label-probabilities)
       -> TorchSaveSink(label-probabilities)
       -> GuessingEntropyReport
```

### More explicit pad view

```text
AesNpyToTorchSource:dataset
  -> dataset_tee:in

dataset_tee:out0
  -> poi_finder:dataset

dataset_tee:out1
  -> mc_label:dataset

poi_finder:poi_set
  -> poi_tee:in

poi_tee:out0
  -> poi_file_sink:in

poi_tee:out1
  -> mc_label:poi_set

mc_label:label_probabilities
  -> probabilities_tee:in

probabilities_tee:out0
  -> probabilities_file_sink:in

probabilities_tee:out1
  -> guessing_entropy:label_probabilities
```

### What PoIFinder does

`PoIFinder` receives the dataset or traces tensor.

It may select, for example:

```text
100 sample indexes out of 5000 samples
```

Output payload:

```text
PoiSetPayload
  indices -> int64 [100]
  scores  -> float32 [100]
```

Caps:

```text
sca/poi-set; index_type=sample; count=100
```

### What MCLabel does

`MCLabel` is a multi-input element.

It receives:

```text
dataset
poi_set
```

It outputs:

```text
label_probabilities
```

Possible payload:

```text
LabelProbabilityPayload
  probabilities -> float32 [trace_count, class_count]
```

or a richer hypothesis-aware shape if needed later.

Caps:

```text
sca/label-probabilities; method=mc
```

### What GuessingEntropyReport does

`GuessingEntropyReport` receives label probabilities and any needed key/plaintext metadata.

It outputs:

```text
AttackReportPayload
```

Possible contents:

```text
best_key_guess
key_rank
guessing_entropy
traces_used
```

Caps:

```text
sca/attack-report; metric=guessing-entropy
```

## Full Example: Future Kyber PC-Oracle Flow

This flow is motivated by FO / re-encryption attacks on KEM decapsulation.

Future input root:

```text
traces/kyber/<capture-kind>/<dataset-name>/
```

Logical training/profile pipeline:

```text
KyberNpyToTorchSource
  -> KyberReencryptionTraceCrop
  -> TTestPoiFinder
  -> PoiGather
  -> DistinguisherTrainer
  -> TorchSaveSink(distinguisher)
```

Logical attack pipeline:

```text
KyberReferenceCiphertextSource
  -> KyberChosenCiphertextGenerator
  -> KyberNpyToTorchSource or live capture source
  -> KyberReencryptionTraceCrop
  -> DistinguisherInference
  -> KyberPcOracleAdapter
  -> KyberAdaptiveKeyRecovery
  -> KyberAttackReport
```

Expected payload flow:

```text
DatasetBundlePayload
  pub_keys
  pvt_keys
  ciphertexts
  traces
  optional messages or decaps_messages

PoiSetPayload
  selected re-encryption leakage sample indexes

DistinguisherPayload
  PC-oracle classifier

OracleResponsePayload
  plaintext-checking responses

AttackReportPayload
  recovered state, key rank, query count, success metrics
```

Notes:

* The generic SCA layer supplies PoI finding, trace cropping, distinguishers, oracle responses, and reports.
* The Kyber plugin supplies ciphertext construction, Kyber parameter knowledge, decapsulation/re-encryption phase mapping, and adaptive recovery logic.
* CNN, ASPN, reduced-template, or other distinguishers are interchangeable elements, not hard-coded pipeline behavior.

## Full Example: Future Kyber SPA Segmentation Flow

This flow is motivated by public-key SPA work on operation segmentation and Kyber message recovery.

Logical pipeline:

```text
KyberNpyToTorchSource
  -> KyberPolyFromMsgTraceCrop
  -> EnvelopeFeature
  -> Segmenter
  -> SegmentClassifier
  -> KyberOperationSequenceClassifier
  -> KyberMessageRecovery
  -> KyberAttackReport
```

Expected payload flow:

```text
DatasetBundlePayload
  traces
  optional message or operation labels for validation

SegmentSetPayload
  operation intervals

OperationSequencePayload
  recovered operation classes

AttackReportPayload
  recovered_message
  operation_accuracy
  traces_used
```

Notes:

* A DQN/RL segmenter can be one possible `Segmenter` implementation in an optional DL or research plugin.
* The core design only needs to carry payloads and route buffers; it does not need to know reinforcement learning, Kyber, or message recovery.
* LLM-assisted SPA should be treated as an optional external analysis aid, not a dependency of LeakFlow's core attack flows.

## Tee and Branching Decision

Normal pad links are one-to-one.

Branching should not be hidden in the link layer.

To branch one output to multiple consumers, use an explicit future `Tee` element.

A future `Tee` element should:

* receive one input buffer,
* produce multiple output buffers,
* share the same payload pointer when possible,
* avoid copying large tensors.

Because payloads are held through smart pointers, tee-style branching can be efficient:

```text
one payload object
multiple buffers
shared payload pointer
```

## Multi-Input Elements

The current linked linear pipeline is not enough for elements like `MCLabel`, which need multiple inputs.

Future multi-input elements will need a processing model closer to:

```cpp
using BufferMap = std::map<std::string, Buffer>;

virtual BufferMap process(BufferMap inputs);
```

or another named-input/named-output mechanism.

This should be a later design phase.

Do not force multi-input graph execution into the immediate property/descriptor phase.

## Graph Observation, Control, and Incremental Rerun

LeakFlow uses two separate planes:

```text
Observation plane
  copied, SCA-safe events for graph/UI display
  no mutable element handles
  no mutable buffer handles
  no payload pointers or raw trace/key/plaintext values

Control plane
  explicit validated commands
  property changes and future lifecycle actions
  accepted/rejected/applied observations
  safe-point application
```

`PipelineGraphRuntime` belongs to the observation plane. It consumes copied
`PipelineObserver` events and stores display state.

`PipelineControlRuntime` is the current UI-side control surface. It binds weakly
to live elements and renders controls from `PropertySpec` / `PropertyValue`.
This is intentionally separate from graph observation. A future
`PipelineController` or `PipelineSession` should become the owner of command
application so UI code does not mutate elements directly.

All property-backed controls should converge on the same source of truth:

```text
Element property value
```

The graph gear control, a standalone control panel, and element-local widgets
such as a future property-backed `TracePlot.trace_index` control should stay in
sync by routing through the same controller/session path.

Future synchronous incremental rerun should use:

* latest accepted input buffer cache per element input pad,
* latest output buffer cache per element output pad,
* dirty output-pad tracking,
* downstream link walking,
* caps revalidation before caps-output reruns,
* copied observations after each accepted/rejected/applied command.

Display-only changes do not produce buffers. Caps, metadata, payload, and
lifecycle changes are dataflow changes. Metadata-output changes should produce a
new buffer envelope with updated metadata even when the payload pointer can be
reused.

Future live pipelines use epochs, not offline-style rerun:

```text
control command
  -> safe-point application
  -> new stream/config epoch
  -> future buffers carry the new epoch
  -> queues apply explicit drain/flush/keep-mixed/drop policy
```

Silent mixing of old and new live epochs should not be the default for
reproducible SCA experiments.

## Near-Term Phase Direction

After Phase 11, the next phase should be:

```text
Phase 12: Core CLI, Inspect Tool, Core Plugin Library, and Tee
```

Scope:

* add a basic `leakflow` CLI runner,
* support setting element properties from the `leakflow` CLI,
* add `leakflow-ls`,
* list linked plugin/element descriptors,
* inspect a linked plugin descriptor,
* inspect a linked element descriptor,
* support `leakflow-ls --no-colors`,
* support `leakflow-ls --color` and `leakflow-ls -C`,
* add `leakflow_plugins_core` as a shared library target,
* add a descriptor catalog for `leakflow_plugins_core`,
* store plugin author and license metadata,
* put core plugin element headers and sources under the core plugin target
  layout,
* keep each core plugin element in its own header/source pair,
* add `FileSrc`,
* add `FileSink`,
* add `FakeSrc`,
* add `FakeSink`,
* add `Summary`,
* add `Tee`,
* add `Queue`,
* add minimal branching execution needed for `Tee`,
* add payload uniqueness helpers for safe mutation decisions.

Design decisions:

* descriptors are linked into the CLI and inspect tool in this phase,
* `leakflow_plugins_core` is one plugin shared library with multiple elements,
* the core element names are element types, not separate plugins,
* `leakflow-ls` follows the broad `gst-inspect-1.0` shape,
* `leakflow-ls` follows the broad `gst-inspect-1.0` color scheme,
* dynamic plugin loading is still postponed,
* the plugin target is `leakflow_plugins_core`,
* the Unix-like shared library file should be `libleakflow_plugins_core.so`,
* the current plugin/element author metadata is `Zer0Leak <edgard.lima@gmail.com>`,
* the current plugin license metadata is `Apache-2.0`,
* `FileSrc` and `FileSink` are raw-byte elements, not dataset or tensor format loaders,
* `Summary` renders a structured Buffer description using a summary level,
* `Queue` stores `Buffer` objects synchronously,
* `Tee` copies `Buffer` envelopes,
* `Tee` shares payload pointers,
* copying buffers protects branch-local caps and metadata,
* sharing payloads avoids expensive data copies,
* in-place payload mutation is allowed only when the payload is uniquely owned,
* use `std::shared_ptr` ownership count as the first conservative uniqueness check,
* do not add generic payload clone/deep-copy yet,
* `Mux` is postponed because mux semantics depend on concrete data shapes.

Out of scope for Phase 12:

* Torch,
* LibTorch,
* NumPy `.npy`,
* Torch `.pt` / `.pth`,
* tensor payloads,
* file-format parsing beyond generic raw bytes,
* AES,
* Kyber,
* DNNs,
* plugin registry,
* dynamic plugin loading,
* `Mux`,
* multi-input execution beyond Tee branching,
* YAML,
* plotting,
* hardware capture,
* async queues or threaded scheduling,
* CUDA changes,
* external dependencies.

Possible later phase sequence, not a binding roadmap:

```text
Phase 13: Source layout and CLI pipeline syntax
Phase 14: Base tensor and tensor-bundle payload layer
Phase 15: NumPy/Torch I/O sources and sinks
Phase 16-20: NumPy/Torch conversion, payload summaries, and explicit pipeline conversion
Phase 21: Logging foundation
Phase 22: ImGui/ImPlot TracePlot plugin foundation
Phase 23: libleakflow_aes helpers
Phase 24: leakflow_plugins_aes source and Pearson PoI finder
Phase 25: PipelineController/session layer, cached buffers,
  downstream-only rerun, and live-stream epochs
Phase 26: First full AES PoI plotting pipeline
Phase 27: Multi-input execution model
Phase 28: AES CPA/report refinement
Phase 29: Additional plot elements such as correlation and PoI overlay plots
Phase 30: Kyber dataset source once traces exist
Phase 31+: Kyber PC-oracle or SPA segmentation flow
Phase 31+: Optional DNN, hardware capture, and GUI/lab frontend
Future low-priority: Generic Convert, conversion registry, and dynamic pads
```

Future numbering should be refined one phase at a time as each previous phase is
completed.

The Phase 13 CLI syntax contract lives in `docs/reference/CLI_SYNTAX.md`.

## Design Summary

Adopt this model:

```text
Disk input:
  .npy NumPy files under traces/
  .pt/.pth Torch artifacts
  future Kyber trace folders

Core transport:
  Buffer + Caps + Metadata + shared Payload

Main numeric backend:
  Torch tensors in leakflow_torch_base

Extras file-format bridge:
  cnpy++-backed NumpyPayload in leakflow_extras

SCA semantic payloads:
  DatasetBundlePayload
  PoiSetPayload
  SegmentSetPayload
  LabelPayload
  LabelProbabilityPayload
  DistinguisherPayload
  OracleResponsePayload
  OperationSequencePayload
  AttackReportPayload
  TensorBundlePayload

Plugin families:
  leakflow_plugins_torch_io
  leakflow_plugins_sca
  leakflow_plugins_dl
  leakflow_plugins_aes
  leakflow_plugins_kyber
  leakflow_plugins_plot
  leakflow_plugins_hardware

Branching:
  Phase 12 Tee element copies Buffer envelopes and shares Payload pointers

Multi-input:
  future named-input processing model

Core:
  no Torch
  no NumPy
  no AES
  no Kyber
  no DNN
  no plotting
  no hardware capture
  no file format dependency
```
