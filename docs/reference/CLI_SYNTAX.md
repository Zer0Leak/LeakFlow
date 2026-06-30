# LeakFlow CLI Pipeline Syntax

## Status

This document defines the `leakflow run` pipeline syntax implemented in Phase 13.

The Phase 12 preset-style CLI remains available as a compatibility path, but
`leakflow run` is the preferred manual pipeline runner.

## Basic Shape

Run a pipeline with one quoted expression:

```bash
leakflow run 'FakeSrc ! Summary'
```

`leakflow run` also accepts frontend options before the expression:

```bash
leakflow run --graph --auto-start --no-telemetry 'FakeLiveSrc ! Queue ! Summary'
```

Runtime telemetry defaults off for headless runs and on for `--graph`. Use
`--telemetry` to force it on, or `--no-telemetry` to keep the graph cheaper when
you do not need live telemetry fields such as `Queue.size`.

The expression is made of statements separated by semicolons:

```text
statement; statement; statement
```

A statement may:

- create or configure an element,
- connect elements with `!`,
- annotate a pad with caps,
- annotate a pad with metadata,
- combine creation, configuration, caps annotations, metadata annotations, and links.

## Elements

Create an element by writing its type name:

```text
FakeSrc
Summary
FileSink
```

Element type matching should be forgiving. These names refer to the same element
type:

```text
FakeSrc
fakesrc
fake-src
fake_src
```

### Named Elements

Give an element an instance name with `@name`:

```text
FakeSrc@src
Summary@s
Tee@t
```

The name is useful when another statement needs to refer to the same element,
and later it can also be useful for logging and diagnostics.

Example:

```bash
leakflow run 'FakeSrc@src; Summary@s; @src ! @s'
```

This means:

```text
Create FakeSrc named src.
Create Summary named s.
Connect src to s.
```

For short linear pipelines, names are optional:

```bash
leakflow run 'FakeSrc ! Summary'
```

Every element has a readable/writable string `name` property. When an element is
created without an explicit name, the CLI generates one from the lower-case
alphanumeric type name plus a zero-based index:

```text
Tee       -> tee0
Tee; Tee  -> tee0, tee1
FileSrc   -> filesrc0
```

The generated name is the real instance name and can be referenced later:

```bash
leakflow run 'FakeSrc; Summary; @fakesrc0 ! @summary0'
```

Creation-time `name=...` is accepted as the element instance name:

```bash
leakflow run 'FakeSrc(name=src); Summary(name=s); @src ! @s'
```

`Element@name(...)` and `Element(name=...)` must agree when both are present.
Duplicate instance names fail instead of being silently renamed. After an
element has been created and indexed in the pipeline, the CLI does not allow
renaming it through a later `@name(...)` reference.

## References

Refer to an existing named element with `@name`:

```text
@src
@summary
@sink
```

Refer to a specific pad with `@name.pad`:

```text
@t.src_0
@t.src_1
@source.src
@sink.sink
```

Use pad references when a link would otherwise be ambiguous.

## Properties

Set element properties with parentheses:

```text
Element(property=value)
Element@name(property=value, other=value)
@name(property=value)
```

Examples:

```bash
leakflow run 'FakeSrc(caps_type=sca/test) ! Summary(level=2)'
```

```bash
leakflow run 'FakeSrc@src(metadata_key=kind, metadata_value=demo); @src ! Summary'
```

Properties are element properties, not pad caps or metadata. Pad caps use square
brackets, and metadata uses braces.

```text
Summary(level=2)        element property
Tee(name=t)             creation-time instance name property
@source.src[caps=...]   pad caps annotation
@source.src{key=value}  metadata annotation
```

## Property Values

Property values are parsed according to the target element's `PropertySpec`.

Supported value syntax:

```text
bool              true, false, 1, 0
integer           50
double            0.25
string            pearson
quoted string     "hello leakflow"
integer list      [0,1,2,15]
double list       [0.1,0.25,0.5]
string list       [traces,plaintexts,key]
quoted string list ["trace data","plain texts","key"]
integer interval  1000..2500
double interval   0.0..2.5
```

Examples:

```bash
leakflow run 'FakeSrc(metadata_value="hello leakflow") ! Summary'
```

```bash
leakflow run 'PoiFinder(method=pearson, poi_count=50, sample_window=1000..2500)'
```

```bash
leakflow run 'PoiGather(selected_bytes=[0,1,2,15], columns=[traces,plaintexts,key])'
```

Rules:

- Bare strings are allowed when they do not contain syntax delimiters.
- Use double quotes for strings with spaces, punctuation, or delimiters.
- Lists are bracketed with comma-separated values.
- Empty lists are valid when the property type accepts a list.
- Nested lists, maps, JSON objects, and YAML are out of scope.

## Shell Quoting

Prefer single quotes when the pipeline expression should be passed literally:

```bash
leakflow run 'FakeSrc(caps_type=sca/test) ! Summary'
```

Use double quotes when you want shell expansion such as `$HOME`:

```bash
leakflow run "FileSrc(path=$HOME/input.bin) ! FileSink(path=$HOME/output.bin)"
```

With double quotes, the shell expands `$HOME` before LeakFlow sees the pipeline.

For paths or strings with spaces inside a double-quoted pipeline expression,
escape the inner quotes:

```bash
leakflow run "FileSrc(path=\"$HOME/my files/input.bin\") ! Summary"
```

The pipeline expression may span multiple lines. When a backslash followed by
optional spaces and a newline appears inside the expression, LeakFlow treats it
as whitespace. This makes pretty-printed examples copyable even when the whole
pipeline is inside one quoted shell argument:

```bash
leakflow run 'FakeSrc@src \
  (caps_type=sca/test) \
  {dataset=smoke}; \
  @src ! Summary(level=3)'
```

A backslash that is not followed by a newline is parsed as an ordinary token and
will be rejected with the element or reference name in the error message.

The `!` operator is inspired by GStreamer. In shells with history expansion,
single quotes are the safest form for interactive use.

## Links

Use `!` to connect elements:

```bash
leakflow run 'FakeSrc ! Summary ! FakeSink'
```

This means:

```text
FakeSrc output -> Summary input -> FakeSink input
```

The CLI infers pads only when inference is unambiguous.

Inference rules:

- If the left side has exactly one output pad, that output pad may be inferred.
- If the right side has exactly one input pad, that input pad may be inferred.
- If the left side has zero or multiple output pads, specify the output pad.
- If the right side has zero or multiple input pads, specify the input pad.
- If both sides are ambiguous, specify both pads.

Examples:

```bash
leakflow run 'FakeSrc ! Summary'
```

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary; @t.src_1 ! FakeSink'
```

```bash
leakflow run '@source.traces ! @join.traces'
```

```bash
leakflow run '@a.out0 ! @b.in0'
```

## Statements

Semicolons separate statements. Each statement is finished before the next one is
interpreted.

Declaration statement:

```bash
leakflow run 'FakeSrc@src'
```

Property update statement:

```bash
leakflow run 'FakeSrc@src; @src(caps_type=sca/test)'
```

Connection statement:

```bash
leakflow run 'FakeSrc@src; Summary@s; @src ! @s'
```

Mixed creation and connection statement:

```bash
leakflow run 'FakeSrc(caps_type=sca/test) ! Summary(level=2)'
```

Branching statements:

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! FakeSink'
```

## Tee Branching

`Tee` has more than one output pad, so downstream links from a tee must name the
output pad explicitly.

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary; @t.src_1 ! FakeSink'
```

This is intentionally explicit:

```text
FakeSrc -> Tee
Tee src_0 -> Summary
Tee src_1 -> FakeSink
```

If a future element has multiple input pads, links into that element must also
name the input pad explicitly:

```bash
leakflow run '@traces.src ! @join.traces; @labels.src ! @join.labels'
```

## Pad Caps

Set or constrain pad caps with square brackets:

```text
Element[caps=TYPE; key=value]
@element[caps=TYPE; key=value]
@element.pad[caps=TYPE; key=value; key=value]
```

Examples:

```bash
leakflow run 'FakeSrc[caps=sca/fake] ! Summary'
```

```bash
leakflow run 'FakeSrc@src; @src.src[caps=sca/fake] ! Summary@s'
```

```bash
leakflow run '@source.traces[caps=sca/traces; dtype=float32; layout=trace,sample] ! @sink.sink'
```

```bash
leakflow run '@reader.src[caps=tensor/torch; dtype=float32; rank=2] ! Summary'
```

Element-level caps annotations infer the only output pad:

```text
FakeSrc[caps=sca/fake] means FakeSrc.src[caps=sca/fake]
```

If an element has multiple output pads, element-level caps are ambiguous and must
be written with an explicit pad reference:

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0[caps=sca/fake] ! Summary; @t.src_1[caps=sca/fake] ! FakeSink'
```

Phase 13 parses and validates pad caps annotations, but it does not apply them
to mutable element pad declarations yet. Full caps negotiation, wildcard caps,
and format-specific caps rules belong to later phases.

## Metadata

Set metadata annotations with braces:

```text
Element{key=value; key=value}
@element{key=value; key=value}
@element.pad{key=value; key=value}
@element.pad_%u{key=value; key=value}
```

Metadata is buffer context, not element configuration and not caps. Use it for
small provenance and reproducibility facts:

```text
dataset=aes_sync
key_id=01
capture=husky
sample_rate_hz=29454545
```

Examples:

```bash
leakflow run 'FakeSrc{dataset=smoke; source=fake} ! Summary(level=3)'
```

```bash
leakflow run 'FileSrc(path=$HOME/input.bin){dataset=aes_sync; key_id=01; capture=husky} ! Summary(level=3)'
```

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0{branch=summary} ! Summary; @t.src_1{branch=sink} ! FakeSink'
```

Element-level metadata annotations apply to all output pads:

```text
FileSrc(...){dataset=aes_sync} applies to every FileSrc output buffer.
Tee@t{dataset=aes_sync} applies to every routed Tee output buffer.
```

Pad-level metadata annotations apply only to one output pad:

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0{branch=summary} ! Summary; @t.src_1{branch=sink} ! FakeSink'
```

Pad-template metadata annotations apply to every existing output pad that
matches a declared request-style template. The `%u` placeholder matches an
unsigned decimal pad index. This is not a general regular-expression language:

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_%u{branch_family=tee-output}; @t.src_0{branch=summary} ! Summary; @t.src_1{branch=sink} ! FakeSink'
```

Metadata keys and values are strings. Quoted values are allowed:

```bash
leakflow run 'FakeSrc{note="hello leakflow"; dataset=smoke} ! Summary'
```

Metadata annotations on output pads stamp routed buffers before delivery to the
linked sink. If the same key is provided at more than one specificity level, the
more specific target wins:

```text
all output pads < pad template < exact pad
```

## Combined Annotations

Properties, caps, and metadata can appear together:

```bash
leakflow run 'FakeSrc(caps_type=sca/test)[caps=sca/fake; dtype=float32]{dataset=smoke; note="combined"} ! Summary(level=3)'
```

The meanings stay separate:

```text
(caps_type=sca/test)               element property
[caps=sca/fake; dtype=float32]      pad caps annotation
{dataset=smoke; note="combined"}    metadata annotation
```

## File Pipelines

Read raw bytes and summarize the resulting buffer:

```bash
leakflow run "FileSrc(path=$HOME/input.bin) ! Summary(level=2)"
```

Copy one raw byte file to another:

```bash
leakflow run "FileSrc(path=$HOME/input.bin) ! FileSink(path=$HOME/output.bin)"
```

Append to an output file:

```bash
leakflow run "FileSrc(path=$HOME/input.bin) ! FileSink(path=$HOME/output.bin, append=true)"
```

## Inspecting Available Elements

Use `leakflow-ls` to inspect linked plugin and element descriptors:

```bash
leakflow-ls
leakflow-ls leakflow_plugins_core
leakflow-ls FakeSrc
leakflow-ls Tee
```

The CLI parser should validate element names, pads, and property values against
the linked descriptors and runtime element declarations.

## Library API

Applications can build a `Pipeline` from the same expression syntax used
by `leakflow run` through the `leakflow_cli` helper library:

```cpp
auto built = leakflow::cli::build_builtin_pipeline_from_expression(
    "FakeSrc ! Tee; @tee0.src_0 ! Summary; @tee0.src_1 ! FakeSink");

auto tee = built.pipeline.element("tee0");
auto tee_elements = built.pipeline.elements_by_type("Tee");
```

Apps that want a smaller linked element set can create an
`ElementFactoryRegistry`, register only the plugin factories they need, and call
`build_pipeline_from_expression(expression, registry)`.

`Pipeline` owns the element name map. Adding a duplicate instance name is
an error, and accepted elements cannot be renamed directly through their `name`
property.

## Errors

The CLI should reject:

- unknown element types,
- duplicate element instance names,
- references to missing named elements,
- references to missing pads,
- ambiguous links where a pad must be specified,
- unknown properties,
- property values that do not match the target `PropertySpec`,
- malformed caps annotations,
- malformed metadata annotations,
- ambiguous element-level caps annotations,
- metadata pad-template annotations that match no output pads,
- pad-template selectors used as link endpoints,
- links with incompatible caps type,
- duplicate links or illegal reuse of a one-to-one pad link.

Example ambiguous link:

```bash
leakflow run 'FakeSrc ! Tee@t; @t ! Summary'
```

The second statement is invalid because `Tee` has multiple output pads. Use:

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary'
```

## Longer Examples

Minimal smoke pipeline:

```bash
leakflow run 'FakeSrc ! Summary'
```

Named smoke pipeline:

```bash
leakflow run 'FakeSrc@src ! Summary@s(level=2)'
```

Configure a source and summarize it:

```bash
leakflow run 'FakeSrc(caps_type=sca/test, metadata_key=kind, metadata_value=demo) ! Summary(level=3)'
```

Declare first, connect later:

```bash
leakflow run 'FakeSrc@src; Summary@s(level=2); @src ! @s'
```

Branch with `Tee`:

```bash
leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! FakeSink'
```

Branch with named endpoints:

```bash
leakflow run 'FakeSrc@src ! Tee@t; @t.src_0 ! Summary@summary(level=2); @t.src_1 ! FakeSink@sink'
```

Raw byte copy:

```bash
leakflow run "FileSrc(path=$HOME/input.bin) ! FileSink(path=$HOME/output.bin)"
```

Raw byte copy with declaration statements:

```bash
leakflow run "FileSrc@reader(path=$HOME/input.bin); FileSink@writer(path=$HOME/output.bin); @reader ! @writer"
```

## Torch Tensor Pipelines

Load one `.pt` tensor file as a `TorchTensorPayload` and summarize it:

```bash
leakflow run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Summary ! FakeSink'
```

Pad caps annotation:

```bash
leakflow run 'FakeSrc@src; @src.src[caps=sca/fake] ! Summary@s'
```

Metadata annotation:

```bash
leakflow run 'FileSrc(path=$HOME/input.bin){dataset=aes_sync; key_id=01} ! Summary(level=3)'
```

Combined properties, caps, and metadata:

```bash
leakflow run 'FakeSrc(caps_type=sca/test)[caps=sca/fake; dtype=float32]{dataset=smoke} ! Summary(level=3)'
```

Future tensor-like example:

```bash
leakflow run '@reader.src[caps=tensor/torch; dtype=float32; rank=2] ! Summary'
```

Future SCA-style example:

```bash
leakflow run 'TraceSource@traces(path=$HOME/traces.pt) ! Normalize@norm ! PoiFinder@poi(method=pearson, poi_count=50, sample_window=1000..2500)'
```

Future multi-input-style example:

```bash
leakflow run '@trace_source.traces ! @poi.traces; @label_source.labels ! @poi.labels'
```

The future examples document intended syntax only. The corresponding elements
and multi-input execution model appear in later roadmap phases.
