# Dataflow, Buffer Provenance, and Synchronization Model

Status: design of record for the dataflow execution model — **offline (Phase 27)
and the live phase, both now implemented**.

This document captures the dataflow execution model, the per-pad buffer routing
contract, and the **vector-clock buffer provenance** mechanism that replaces
`Buffer::epoch()` for all synchronization.

It is the authoritative design for:

- **Offline / one-shot (Phase 27, implemented):** DAG executor, per-pad outputs,
  Tee, the vector clock, and epoch removal.
- **The live phase (implemented):** threaded segments (one thread per segment, cut
  at every `Queue`), `Queue`-owned `BufferQueue` boundary runtime, backpressure /
  drop policies, the aggregator (Barrier fold-match + realign + Held + Latest), the
  `Sync` element, cooperative stop, the safe-point control plane, and the
  `Stopped/Running/Paused/Idle` player state machine.

> **Implementation map.** Sections 10–11 are the design; **§12 is the
> implementation status** (what was built, where, and the tests); **§13 is the
> runtime control / player model**; **§14 is a CLI cookbook** to drive every
> feature from the command line. If you only want to *use* the system, read §13–§14.

Companion docs: **`dataflow_sync_walkthroughs.md`** (ten worked pipeline
examples that exercise this model end-to-end — threads, join behavior, and the
vector-clock indexes step by step), `architecture.md`, `pipeline_controller.md`
(Phase 25 control layer), `runtime_execution.md`, `metadata_klass_taxonomy.md`.

> Note: this doc mentions bundle/`Mux`/`Demux` in Sections 4–9; those are
> **dropped** (superseded by the vector clock and the Sync element) — see the
> Section 4 status note and §9. The finalized model is Sections 10–12.

---

## 1. Motivation

The original executor is named `Pipeline`, but it already routes buffers
through a small DAG (multi-input gather + Tee fan-out). It carried three implicit
simplifications that this design removes or makes explicit:

1. execution order was element **insertion order** (valid only because `link()`
   enforces source-before-sink);
2. an element returned **one** `optional<Buffer>`, **broadcast** to all output
   pads (no distinct per-pad output, so no demux/splitter was possible);
3. synchronization across branches relied on a global `epoch` stamp plus
   "latest-valid-per-pad", which cannot express per-lineage generations or live
   stream matching.

The model below fixes all three and unifies batch-id, the matching half of join
policy, and the global epoch into a single mechanism: a **vector clock per
buffer**.

---

## 2. Per-pad output contract (Decision: Option B)

- An element produces **`ElementOutputs` = map<output-pad-name → Buffer>**, the
  mirror of `ElementInputs`. A single-output element may return one buffer as
  sugar mapped to its sole output pad.
- The engine routes **each pad's buffer to that pad's link**, uniformly, applying
  per-pad metadata annotations (unchanged from today).
- **Fan-out is a Tee behavior, not an engine rule.** The implicit "one returned
  buffer → all pads" broadcast is removed. Tee explicitly emits a copy per
  declared output pad (copy the envelope, share the payload `shared_ptr`).
- This unlocks **Demux** (distinct buffers per pad) and multi-output sources, and
  keeps the engine's routing a single rule.

Rationale: broadcast as an engine rule would force the engine to own per-branch
rate/backpressure for every multi-output element in live mode. As a Tee behavior,
the engine stays rate-agnostic and uniform.

---

## 3. DAG executor

- Execution order is **element insertion order**, which `link()` guarantees is a
  valid topological order: a link is rejected unless the source element's index is
  strictly less than the sink's, so every link goes low→high index. The same
  invariant makes **cycles impossible** (a cycle would need a high→low link). A
  separate Kahn pass / cycle check is therefore unnecessary — topological validity
  and acyclicity are structural properties of the link rule.
- For each element in order: gather one buffer per connected input pad
  (collect-then-fire), run `process_pads(...)`, route its per-pad outputs.
- Offline remains **synchronous, single-threaded, one buffer per pad per sweep**.
  The live phase introduces threaded segments (Section 10).

---

## 4. Bundles: coupled data on one pad

> **Status (post-27.3): DEFERRED to the live phase.** Bundles, `Mux`, and `Demux`
> were originally Phase 27 (27.4/27.5) to keep co-acquired data from desyncing. The
> **vector clock (Section 6) subsumes that correctness role**: separate trace /
> plaintext / key pads into `AesLeakage` are now safe because the fold-match
> verifies they share a generation (offline, one buffer per pad makes the pairing
> unambiguous anyway). So `AesLeakage` keeps its separate pads, and bundles/Mux/
> Demux were **not built in Phase 27** — they move to the live phase, where a real
> dataset/capture source emitting one coupled record (or combining genuinely
> independent streams) gives them a concrete, testable purpose. The remaining gap
> the clock cannot close — **pairing genuinely independent sources** (Section 8
> residual #2) — is exactly the live-phase problem bundles/Mux solve. The design
> below is the intended shape for that phase. `TorchTensorBundlePayload` already
> exists (Phase 15) for code that wants a tensor collection.

Row-aligned, co-acquired data (e.g. `{trace, plaintext, key}` of one acquisition)
travels as **one Buffer carrying a bundle Payload on one pad**, never as separate
desync-prone pads.

- Core defines an abstract **`BundlePayload`** interface: `keys()` and
  `get(key) -> shared_ptr<Payload>`. It is torch-free, so it lives in
  `leakflow_core`.
- `TorchTensorBundlePayload` (in `leakflow_base`, already exists) implements it.
- **AesLeakage** takes a single bundle input pad (`record`), reads `traces`,
  `plaintexts`, `keys` by name (keeping single-key broadcast), and its output is
  unchanged.
- Bundles are reserved strictly for **co-acquired fields** — never for forcing
  sync between independently-derived branches (that is what the vector clock and
  Mux are for). This discipline prevents "bundle converter" proliferation.

### Mux and Demux (generic structural elements, in `leakflow_plugins_core`)

- **Demux** (1 → N): splits a `BundlePayload` into per-key output streams. It is
  generic over the core `BundlePayload` interface. Output pads are **created on
  connect, named by key** (`@demux.traces`); the key is named at graph-build time
  (link-time pad creation), validated against the bundle at runtime. Unrequested
  keys are simply not emitted. A "default expose all keys" mode exists for
  inspection.
- **Mux** (N → 1): packs N input streams into one `BundlePayload`. It is the
  **only synchronizer** — the join/retention policy lives here, not in the engine.
  Mux also merges `capture.*`/`origin.*` metadata per the klass taxonomy.

Tee/Demux are the fan-out duals; Mux is the fan-in; the engine just routes.

---

## 5. Synchronization taxonomy

Two structural axes — **provenance** (common origin vs independent sources) and
**convergence** (streams reconverge vs end at separate sinks):

| | Converge (1 element combines) | Separate sinks (rendered apart) |
|---|---|---|
| Common origin | ① split→join (diamond) | ② split→render |
| Independent origins | ③ join independents | ④ render independents |

Two overlays decide difficulty:

- **Provenance decides whether alignment is free.** Common-origin branches are
  row-aligned by construction (same source row `n`); the sync key can be stamped
  at the origin and propagated. Independent sources have no free alignment — they
  align only if they share a key space.
- **Convergence decides whether desync is fatal or cosmetic.** Converge = wrong
  numbers (correctness). Separate sinks = at worst a presentation glitch — and
  **LeakFlow has no playback clock**, so the entire "separate sinks" column (②④)
  is easy: each sink just draws its current data; x-axis alignment is intrinsic to
  the sample index.

Inside "converge" there are two strengths:

- **Row-aligned combine** (PearsonCorrelator: `features[n]` ↔ `targets[n]`) — needs
  row sync.
- **Reference / overlay combine** (TracePlot: traces + annotations that index into
  them by sample) — needs only **provenance/epoch consistency**, not row sync. So
  TracePlot must **never epoch-gate**; it renders the latest valid traces + latest
  valid annotations.

Conclusion: the real sync problem reduces to the **converge column**, and the only
machinery-needing cells are **① across a Queue** and **③ independent join** — both
**live-only**.

---

## 6. Vector-clock buffer provenance (replaces epoch)

Each buffer carries a **vector clock**: a per-element production count that records
causal lineage. This is the single mechanism for batch identity, the matching half
of join policy, and per-element generation (replacing the global epoch).

### Slot allocation

- The pipeline holds a global monotonic `uint32` "next index" counter; **index 0
  is reserved** ("buffer is not downstream of this element / wildcard").
- When an element is **added**, it atomically does `base = counter.fetch_add(k)`
  where `k` is the number of slots it requests (declared in its descriptor):
  - **Tee, sinks → 0 slots** (produce no new causality; Tee copies the vector
    verbatim to every branch).
  - **One src pad → 1 slot.** Most multi-src-pad elements → **1 slot** too
    (their pads are causally related).
  - **Independent src pads → 1 slot per independent pad** (none today).
- `max_id` = final counter value = sum of allocated slots (not element count).
- Slot allocation, increment, and merge are **engine concerns**; elements stay
  vector-unaware (they only declare `slots` in the descriptor).

### Storage

- Each buffer carries a **dense `uint32` array sized to `max_id`** for O(1)
  offset access (`array[base]`), plus a companion **set/bitset of non-zero
  indexes** so joins iterate only live slots (skip zeros) while still using direct
  offset access. (Implementation note: Phase 27 lands the dense array + the
  fold-and-detect match; the non-zero-set is a performance optimization deferred
  to the live phase, where `max_id` and buffer rates make it matter. Offline
  `max_id` is tiny, so the dense scan is already cheap.)
- `array[i] == 0` ⇒ buffer is not downstream of element `i` (wildcard in matching).
- On counter wrap, the next value is **1, not 0** (0 stays reserved).

### Runtime rules

- **On src-pad emission**, the engine increments the producing element's slot
  (`array[base]++`) on the emitted buffer. A fan-out (Tee) stamps **all branches
  identically** (single logical increment / verbatim copy).
- **On input gather at a join**, the engine **folds** all input vectors into the
  merged output (component-wise max) and **detects conflict in the same pass**:

  ```
  merged[] = 0
  for each input buffer b (walk its non-zero set):
      for each index i in b.set (direct offset into merged):
          if   merged[i] == 0    : merged[i] = b[i]      // first claims i
          elif merged[i] == b[i] : continue              // agrees
          else                   : CONFLICT -> wait      // both non-zero, differ
  ```

  - **Matched** (all both-non-zero overlaps equal) → fire; `merged` is the output
    vector; the producing element then increments its own slot; output set = union
    of input sets (+ own slot).
  - **Conflict** → the element does nothing and waits (no-drop regime).
- This fold is **correct for any N inputs** (anchoring on one buffer is wrong: it
  misses conflicts among non-first buffers where the anchor is zero) and runs in
  **O(total non-zero entries)**. Sink-only joins use a **generation-stamped
  scratch array** for O(1) reset.

### Worked example (`max_id = 3`; Src=1, B=2, P=3)

- features X: `[_,5,0,0]` set `{1}`; targets Y: `[_,5,7,0]` set `{1,2}`.
  Overlap `{1}`: 5==5 → fire; merged `[_,5,7,0]` then `++`P=3 → `[_,5,7,1]`.
- If Y were `[_,6,7,0]`: index 1 → 5≠6 → wait (different source batch).

---

## 7. Epoch removal and partial re-run

- **`Buffer::epoch()` is removed.** An element's **own slot** is its per-element
  generation: a reprocess increments it, so the new buffer supersedes the old via
  "highest compatible vector". This is richer than a global epoch — it
  distinguishes *new upstream data* (ancestor slot ↑) from *reconfigured here*
  (own slot ↑), which a single global number cannot.
- **Partial re-run is unchanged in orchestration** (cache + replay-set +
  downstream walk from Phase 25). Only the consistency test upgrades: instead of
  "no epoch gate / latest-valid", the join **matches by vector**. Trace: change
  AesLeakage → cached traces carry `{Src:k}`; reprocessed leakage carries
  `{Src:k, Aes:m+1}`; they match on `Src:k`; TracePlot combines cached traces +
  new annotations matched on `Src:k`. Provenance becomes **checked**, not assumed.
- The Phase 25 control-layer *mechanisms* (command queue, safe-point application,
  transactional caps validation, downstream-only rerun) **stay**; they simply no
  longer carry a global epoch number.

---

## 8. Residual knobs — RESOLVED by the live-phase model (Section 11)

The three residuals first flagged here all have a concrete resolution in the
finalized synchronization & liveness model (Section 11). Summary:

1. **Retention (hold vs consume)** → a per-input join *mode* (`Held` / `Latest` /
   `Zip` / `Barrier` / `AllRequiredOnce`, Section 11.1). For ordinary pipelines the
   default barrier covers it; non-default modes live **inside a Sync element**, not
   on arbitrary pads.
2. **Independent-source pairing (case ③)** → the **Sync element** (Section 11.4):
   it "injects a common ancestor," converting independent streams into
   common-origin streams so the default barrier handles everything downstream.
3. **Drops / realignment** → a live-phase concern handled at the `Queue` boundary
   and inside the Sync element (Section 11, "no-drop on rejoining branches").

The key simplifying decisions (Section 11) are: **default sync = per-slot barrier
(the fold-match), single counter per slot (no global version), liveness-aware
property changes, and the Sync element as the sole front door for custom
cross-source pairing.**

---

## 9. What Phase 27 implements vs defers

Implemented now (offline / one-shot):

- **27.1** per-pad `ElementOutputs`; uniform engine routing; `Tee` explicit
  fan-out (the implicit "one buffer → all pads" broadcast removed); topological
  validity + acyclicity guaranteed by the existing `link()` low→high invariant;
- **27.2** the **vector clock** (descriptor `provenance_slots`, pipeline slot
  allocation, dense `provenance` array on `Buffer`, engine increment on emission,
  conflict-detecting `merge_provenance` fold-match at joins), wired through the
  offline executor and partial re-run, with `tests/core/vector_clock_test.cpp`;
- **27.3** **removal of `Buffer::epoch()`**; the Phase 25 control layer migrated
  to a session `generation` counter, and per-buffer generation derived from the
  vector clock (`provenance_generation`) for observer/plot change detection;
- **27.6** class rename `LinearPipeline` → `Pipeline` (files
  `pipeline.{hpp,cpp}`).

Deferred to the **live phase** (Section 11 is the design of record; Section 12 is
the phase brief):

- threaded segments and the real `Queue`; backpressure; drop handling;
- **liveness model** (live vs one-run sources; liveness-aware property changes);
- the **Sync element** and the join modes it encapsulates;
- a **fake live-capture source** to drive and test all of the above.

Superseded / dropped:

- **bundles, `Mux`, `Demux`, the `BundlePayload` interface** — superseded by the
  vector clock (offline) and the **Sync element** (live). The Sync element keeps
  streams *separate* (N→N) and synchronizes them by injecting a common ancestor,
  which is strictly more flexible than packing them into one bundle. `Mux`/`Demux`
  are **not planned**. (`TorchTensorBundlePayload` still exists for genuine
  data-bundling needs, but it is not the synchronization mechanism.)
- **Dynamic plot sink pads** (old Phase 27 half two): dropped. Multiple annotation
  layers, if ever needed, are handled at Phase 29 (overlay plots).
- **`AesLeakage` bundle pad** (old 27.5): not done; the vector clock makes its
  separate trace/plaintext/key pads safe.

---

## 10. Live and Queue model (IMPLEMENTED — see §12 for the build status)

### Segment-threaded execution

- **Not one thread per element.** The current runner introduces one streaming
  thread per segment, cut at **Queue-like thread boundaries**. A maximal run of
  elements with no Queue between them is a *segment* and runs on one streaming
  thread.
- **Thread count = connected components after cutting at every Queue.** Two queues
  into the *same* join collapse into one downstream thread; two queues into
  *different* downstream elements give two threads. It is *where the queues lead*,
  not how many queues, that sets the count.
- Most elements stay **synchronous and thread-unaware**. Boundary elements own
  their local runtime handoff via `ThreadBoundaryRuntime`; the pipeline remains the
  lifecycle/routing/provenance/error authority. LeakFlow's heavy compute is inside
  Torch (already parallel), so live realistically needs few runtime threads (e.g.
  capture-source → Queue → analysis → UI).

### Queue = one primitive, three roles

A `Queue` is simultaneously:

1. **rate decouple / backpressure** — a bounded buffer; the only place producer and
   consumer firing counts may differ;
2. **generation boundary** — drain / flush / keep-mixed policy (the
   `QueueEpochPolicy` enum, now expressed against the relevant element's slot,
   not a global epoch);
3. **thread boundary** — the thread-safe handoff between two segments.

The vector clock **rides through the Queue untouched**; the Queue never inspects
it.

Implementation note: `Queue` now owns the underlying `BufferQueue` and exposes
first-class telemetry fields (`size`, `peak_size`, `received`, `emitted`,
`dropped`) separately from properties. The pipeline no longer creates a separate
queue object; it calls the boundary through `ThreadBoundaryRuntime`. The threaded
runner samples generic boundary telemetry snapshots at UI cadence and publishes
only changed values as copied `TelemetryChanged` events, so the graph hover/pinned
topology view can update live telemetry without per-buffer UI events. Runtime
telemetry can be disabled per run; then the sampler/publisher is not started and
framework telemetry primitives no-op, so elements such as `Queue` do not own
the enable/disable policy. `Queue` also implements
`ActiveElement`: when a consumer segment is fed only by active-capable boundaries
and has no internal source-like producer, those boundaries own the source-side
tasks and push downstream through `RuntimeContext`. If the segment also contains
a passive source such as `TorchFileSrc`, the regular consumer segment thread is
kept so the source can run. The runner waits for active tasks via
`ActiveElement::wait_active()` before teardown.
Queue claims zero provenance slots
because it is a pass-through boundary; the vector clock rides through it untouched.

### Diamonds and aggregation in live

- A rejoining diamond (`A→B; A→C; B→D; C→D`) **without a Queue on at least one
  branch deadlocks** in a push-threaded model (the tee blocks pushing branch 1
  while D waits for branch 2). With queues on the branches, A's branches run
  decoupled and D becomes a real **aggregator** pulling matched-vector buffers
  from its input queues. This is why "put a queue on each branch after a tee when
  they rejoin."
- D's aggregation is exactly the **fold-match** of Section 6, now across queue
  heads. It **matches by vector, never by arrival order**. The asymmetric diamond
  (raw branch + processed branch) needs a Queue on the **fast** branch so the
  early raw buffer can wait for the slow processed one.

### 1:N (held) in live

- One input buffer → many outputs (B reads hardware repeatedly for one A-record,
  or an online refiner). B's outputs carry a compound vector (ancestor slot fixed,
  B's own slot incrementing). The raw input is `Held` (its slot is a wildcard, so
  it matches every B output); the driving input is consumed per buffer. At the
  generation boundary (A advances), the Queue's drain/flush policy decides the fate
  of in-flight old-generation buffers.

### Correctness rules carried into live

- **No-drop on rejoining branches** (or the aggregator must detect a vector gap and
  realign) — otherwise a drop pairs mismatched generations.
- **Per-index monotonicity** — a join must not regress to a lower slot value than
  one already consumed on that index.

### Mapping to the offline model (why offline is a faithful subset)

- The **rerun cache is the offline twin of the live Held/Queue**: hold the
  unchanged branch (in cache vs in a queue), recompute the changed branch, match by
  vector. Same shape; the live phase replaces "cache" with "bounded threaded
  queue."

---

## 11. Synchronization & Liveness — final model (live-phase design of record)

This section is the authoritative, finalized synchronization design. It is the
result of the Phase-27 design conversation and supersedes the looser language in
Sections 5–8 and 10 where they conflict.

### 11.0 Design goals (the north star)

1. **99%+ of LeakFlow API users never tune synchronization. It just works.** A
   normal SCA pipeline (file sources, a leakage model, a PoI finder, a plot) has
   zero sync configuration.
2. Only genuine **edge cases** — logically pairing *unrelated* streams — need an
   explicit, *visible* knob, and that knob is a single element (the **Sync
   element**), not scattered per-pad flags.
3. **One coherent model across one-run (offline) and live streaming.** Offline is
   a faithful subset of live.

The model rests on four decisions: **(A) default sync = per-slot Barrier; (B) one
counter per slot, no global version; (C) liveness-aware property changes; (D) the
Sync element is the sole front door for custom cross-source pairing.**

### 11.1 The five join modes (vocabulary)

A multi-input element pairs buffers across its input pads by one of these modes.
**For ordinary elements the mode is always `AllRequiredOnce` / the default barrier
— the other modes appear only *inside a Sync element* (11.4).** They only differ
in live/streaming; offline (one buffer per pad) they all fire once.

| Mode | Behavior | SCA example |
|---|---|---|
| **AllRequiredOnce** | Each connected input delivers exactly one buffer; fire once when all present. The whole offline/one-shot model. | The current AES pipeline: one trace file, one plaintext file → fire once. |
| **Zip** (lockstep 1:1) | Each fire consumes **one fresh buffer from each** input, paired by position (`A[n]`↔`B[n]`), advancing both in lockstep; if one input has not produced its next buffer, **wait**. | Two synchronized capture channels combined per acquisition. |
| **Held** (constant / broadcast) | One input is **held and reused** across many fires of the driving input; replaced only when it emits anew. A wildcard that matches every driving buffer. | One AES **key** broadcast across many trace batches. |
| **Latest** (sample-and-hold a stream) | Fire when the **primary** input advances; pair with the **most recent** buffer on the secondary input, **dropping** intermediates. Never blocks on the secondary. | `TracePlot`: traces stream; **annotations** update occasionally — use the newest, don't wait. |
| **Barrier** (wait-for-all-by-identity) | Fire only when all inputs reach a **matching identity/generation**. For **common-origin** reconvergence this is *free* (the fold-match). | The diamond: the fold-match makes the join wait for matching generations automatically. |

Relationships: `AllRequiredOnce` is the offline degenerate case of `Zip`; `Zip`
pairs by *arrival position*, `Barrier` by *identity*; `Held` is "a constant I
reuse," `Latest` is "a stream I sample the newest of and may skip."

### 11.2 Decision A — default sync = per-slot Barrier (the fold-match)

The vector-clock fold-match (Section 6) **is** a per-slot barrier, and it is the
default for every join, with **no tuning**:

- **Common-origin reconvergence** → branches share an ancestor slot → the values
  must match → the join is barriered (waits for matching generations). Free.
- **Independent inputs** → no shared non-zero slot → the fold is vacuous → the join
  **fires freely**. This is correct: with no common source there is no causal
  relationship, so there is nothing to synchronize.

This covers **99.99% of SCA**: if two inputs share a source they are kept in sync;
if they do not, they are genuinely independent and need no sync.

### 11.3 Decision B — one counter per slot; NO global version

Each slot holds a **single monotonic counter** ("generation"): it increments on
every emission of the owning element (a new batch *or* a reconfigure). We
**deliberately do not** split it into a global `(position, version)` pair.

Rationale — the version was only ever needed to stop a property-change rerun from
looking like a *new batch* to a position-`Zip` (the "conflation"). That conflation
only arises in the intersection of **offline rerun × streaming Zip**, which
**does not exist** once property changes are liveness-aware (11.5):

- a **live** source reconfigured → applies *forward* to its next genuine batch (a
  real advance, not a spurious bump);
- a **one-run** source reconfigured → offline replace-in-place (one batch, no
  stream → no streaming Zip to stall).

Therefore the single counter suffices everywhere. The **agreed constraint** that
keeps this true: **custom position-pairing must go through a Sync element** (11.4),
never a bare per-input-pad policy — so it never has to survive a raw offline
reconfigure of one of its inputs. If a Sync element's internal policy ever needs a
position-vs-version distinction, it tracks that **internally**; it never leaks into
the global per-slot clock.

### 11.4 Decision D — the Sync element ("inject a common ancestor")

The **Sync element** is the single, explicit, opt-in mechanism for logically
pairing **independent** (differently-clocked) streams. It is the front door for
*all* custom synchronization.

- **Shape:** **N input pads (≥2) → N output pads.** Configured with a **tuned sync
  policy** (`Zip` / `Barrier`-by-key / `Latest` / `Held` / `AllRequiredOnce`).
- **Mechanism — inject a common ancestor:**
  - it **claims its own provenance slot** (like a source/transform; unlike `Tee`);
  - it pairs its N inputs by the tuned policy;
  - it **stamps all N outputs of a given sync-fire with the same value on its own
    slot** (exactly as `Tee` stamps fan-out branches identically).
- **Effect:** the N outputs now **share the Sync element's slot** → downstream they
  are **common-origin (case ①)** → every downstream join uses the **default
  per-slot barrier and just works.** No downstream element knows or cares that the
  inputs were originally unrelated.

So the Sync element **converts case ③ (independent, no sync) into case ① (common
origin, auto-synced).** It is a generalization of `Tee`:

```text
Tee  : 1 input  -> N identical outputs            (copies the clock; claims no slot)
Sync : N inputs -> N aligned outputs              (claims a slot; stamps all N alike)
```

**All** custom-sync complexity — the policy, arrival-ordinal tracking, drop
detection, any optional shared-key matching, any internal position/version — lives
*inside* this one element. 99% of pipelines never insert one. Bare per-input-pad
policy tuning remains a **deep escape hatch, not the front door**; prefer the Sync
element.

### 11.5 Decision C — liveness-aware property changes

A property change must behave differently in one-run vs live, and this is
**derived, never tuned**:

- **Source liveness:** every source declares **live** (capture/streaming) or
  **one-run** (file/static). **Default is one-run.**
- **Propagation:** liveness flows downstream by **OR** — an element is
  **live-driven** if *any* input traces back to a live source. The pipeline
  computes this reachability at link/start time and stamps each element.
- **On a `SetProperty` command, the session checks the changed element:**
  - **one-run-driven** → **re-emit from cache** (`rerun_from`): recompute/replace
    the current batch in place, because no future buffer is coming and you must
    recompute to see the effect. (This is exactly today's Phase-25 behavior.)
  - **live-driven while Running/Paused** → **do not emit**; just update the config.
    The change applies **forward**, to the **next** buffer the live source pushes.
    Past streamed batches are gone; reconfiguration is forward-only.
  - **live-driven while Idle** → there is no next live buffer, only held cached
    data. The edit therefore behaves like a cached replay: recompute if the
    replay-set is replay-safe, otherwise reject the command before committing the
    property.
- **Mixed inputs** (one live + one static into the same element): the element is
  live-driven; the static inputs are simply **Held** (reused) when the live input
  re-triggers the element.

This is the piece that dissolves the version problem (11.3) and makes "change a
property mid-stream" correct without any user configuration.

### 11.6 The cross collapses (why it stays simple)

The apparent cross of {live, one-run} × {sync policy} × {mixed live/static paths}
collapses because every axis is **derived or localized**, not tuned:

| Axis | How it's handled | User tuning? |
|---|---|---|
| live vs one-run | derived from source type, propagated by OR | none (pick the source) |
| default sync | automatic fold-match (per-slot barrier) | none |
| custom sync | one explicit **Sync** node | only when inserted |
| property change | liveness + session state decide cache replay/reject vs apply-forward | none |
| mixed live/static into a join | live-driven + static `Held` | none |

User-facing story: **"pick live or file sources; the rest synchronizes itself; drop
in a Sync node only to pair genuinely unrelated streams."**

### 11.7 Worked examples

- **Diamond (common origin):** default barrier; the cached branch and the reran
  branch match on the shared ancestor slot (the reran element's own slot is a
  wildcard on the cached branch). Already works offline today.
- **`A→Z; B→Z` independent, change a property of `A`:**
  - *one-run `A`* → replace-in-place: `Z` re-fires `(a', cached b)`; no advance, no
    stall (session-orchestrated, `b` held from cache).
  - *live `A`* → forward: `A`'s next batch uses the new config; `Z` zips
    `(a_{n+1}, b_{n+1})` as normal.
- **Zip semantics:** pair by position, consume one from each per fire, advance both
  in lockstep, wait if one side has not produced its next buffer. A property change
  never advances the Zip position (forward in live, replace in one-run).

### 11.8 Execution: one `run()`, the 3-state stream result, and cooperative stop

**One `run()` — a pump loop, not a mode.** `run()` is
`start_all → pump-until-end → stop_all`, where the pump repeatedly sweeps the graph
and each source produces its next result. **Live is not a distinct run method** — a
one-run source simply ends after one buffer; a live source ends after its stream.
`Pipeline::run()` *is* this pump loop, and the current one-shot is just its
degenerate single-iteration case. There is **no `run_live()`**, and the
`OneShotDrive`/`StreamingDrive` distinction (earlier draft) **collapses into the one
loop**. The liveness flag (11.5) is **control-plane only** — it governs
property-change behavior, never how to run. Running therefore needs **no** mode
detection; the loop discovers end-of-stream from the sources.

**The 3-state stream result.** "No buffer" is ambiguous in a stream, so a streaming
`process` distinguishes three outcomes per source / output pad:

- **`Data(Buffer)`** — a buffer produced this tick.
- **`NoData`** — nothing *this tick*: a **transient** absence (a poll timeout, an
  empty `Queue`, an aggregator still waiting for a matching buffer). The pump
  **retries**; the stream is not over. `std::nullopt` / an empty `ElementOutputs`
  means **`NoData` and nothing else** — it is reserved for this transient case.
- **`EndOfStream` (EOS)** — this stream is **finished** (terminal). The pump stops
  pumping that source; EOS **propagates downstream** as elements drain, and `run()`
  returns once EOS reaches the sinks.

A **one-run** source returns `Data` **once**, then **EOS** → offline is a single
pump iteration (today's behavior, unchanged). A **live** source returns `Data` per
batch, then **EOS** at end-of-stream (a finite fake source) or never (hardware,
until stopped). An **aggregator/join** returns `NoData` while it is still waiting
for a matching buffer on some input, and `EOS` once all its inputs are EOS and it
has flushed.

**Cooperative stop (graceful, as soon as possible).** `run()` owns a
`std::stop_source`. `stop()`, **Ctrl+C** (SIGINT), and **closing the `--graph`
window** all call `request_stop()`. Any **blocking** operation — a live source
waiting for a sample, a `Queue` pull — observes the `std::stop_token` and must
return **promptly** when stop is requested (return `NoData`/`EOS` and unwind)
rather than block indefinitely. The recommended pattern is **poll-with-timeout**: a
blocking source waits with a short timeout, returns **`NoData`** if nothing arrived,
and loops — which both (a) yields control so the pump can check the stop token, and
(b) makes the source interruptible **without OS-level signals**. On stop, the pump
stops feeding new buffers, drains/flushes in-flight buffers per the `Queue` policy,
runs `stop_all` in **reverse order**, and `run()` returns. So a half-finished
`process` finishes smoothly the next time it checks the token — bounded by the poll
timeout, not by the source's data rate.

This also means **`NoData` and cooperative stop are the same mechanism seen twice**:
a source that polls-with-timeout naturally yields `NoData` ticks, and those ticks
are exactly where the stop token is checked.

---

## 12. Live phase — implementation status (what was built, where, and the tests)

The live-streaming machinery in Sections 10–11 is **implemented and tested**.
Offline behavior is unchanged (every offline test stayed green throughout). The
table below maps each design piece to its code and its test; file paths are
relative to the repo root.

| Design piece | Where it lives | Test |
|---|---|---|
| **Unified `run()` pump loop (11.8)** — no `run_live()`; live auto-detected via `has_live_source()` | `Pipeline::run`, `src/core/pipeline.cpp` | `tests/plugins/base/fake_live_src_test.cpp` |
| **3-state stream result** (`Data` / `NoData` / `EndOfStream`); `nullopt` reserved for NoData | `Element::at_end_of_stream`, `Pipeline::all_live_sources_at_eos` | as above |
| **Cooperative stop** — `std::stop_token` through element → pipeline → session; interruptible source wait; CLI **SIGINT** bridge; `--graph` window-close | `Element::set_stop_token`, `Pipeline::set_stop_token`, `InterruptScope` in `src/apps/leakflow/leakflow_cli.cpp` | `fake_live_src_test.cpp` (pre-stop + mid-stream interrupt) |
| **Fake live source** — Torch `.pt`, one `Buffer` per axis-0 row `[1, M]`; `trace_rate` paces fake-live replay (one trace per `1/rate` s); `capture.sample_rate_hz` is explicit user metadata only; honors the stop token | `FakeLiveSrc`, `src/plugins/base/fake_live_src.cpp` | `fake_live_src_test.cpp` |
| **Liveness model (11.5)** — `live`/`one-run` declaration (default one-run); downstream OR-propagation; forward-apply on live-driven Running/Paused, cache replay/reject in Idle | `Element::is_live`, `Pipeline::is_live_driven`, `PipelineSession::apply_command` | `tests/core/pipeline_session_test.cpp` |
| **Segment decomposition** — cut at every `Queue`; one segment per connected component | `decompose_into_segments`, `src/core/pipeline_segments.cpp` | `tests/core/pipeline_segments_test.cpp` |
| **Threaded runner** — one `std::jthread` per segment; boundary handoff through `ThreadBoundaryRuntime`; EOS propagation; per-segment safe point | `Pipeline::run_threaded` / `run_source_segment` / `run_consumer_segment` / `execute_segment`; `include/leakflow/core/thread_boundary_runtime.hpp` | `tests/core/threaded_runner_test.cpp` |
| **Active source runtime path** — source-like `ActiveElement` owns a task and pushes through `RuntimeContext` into downstream boundaries; `safe_point(element)` preserves pause/property behavior; `wait_active()` gives the runner an explicit natural-completion point; provenance/metadata/observer routing stays centralized | `RuntimeContext`, `ActiveElement`, `Pipeline::PipelineRuntimeContext` | `tests/core/threaded_runner_test.cpp`, `tests/core/threaded_pause_test.cpp`, `tests/core/threaded_safe_point_test.cpp` |
| **Runtime push collector** — active pushes can feed passive chains, downstream Queues, and multi-input Barrier/Held/Latest joins; mixed segments with internal passive sources stay on the pull-side consumer runner | `Pipeline::route_runtime_push`, `route_runtime_passive_input`, `route_runtime_outputs`, `Pipeline::is_active_boundary_consumer_segment` | `tests/core/threaded_runner_test.cpp`, `tests/core/aggregator_latest_test.cpp`, `tests/plugins/core/sync_threaded_test.cpp` |
| **`Queue` owns `BufferQueue`** (3 roles) — bounded FIFO; **Block** (backpressure) vs **DropOldest**; separate first-class telemetry (`size`, `peak_size`, `received`, `emitted`, `dropped`); runtime telemetry can be disabled per run; thread boundary; generation boundary; 0 provenance slots | `BufferQueue`, `src/core/buffer_queue.cpp`; `Queue` (`thread_boundary = true`, implements `ThreadBoundaryRuntime` and `ActiveElement`) | `tests/core/buffer_queue_test.cpp`, `tests/plugins/core/core_elements_test.cpp`, `tests/core/threaded_runner_test.cpp`, `tests/plugins/core/sync_threaded_test.cpp` |
| **Aggregator (11.1)** — peek/hold queue heads; **Barrier** fold-match by vector clock; **realign-on-drop** (discard orphans); **Held** reuse (auto from liveness); **Latest** sample-newest; **Zip/AllRequiredOnce** map to the default | `Pipeline::run_consumer_segment` (the `Head` modes) | `threaded_runner_test.cpp` (realign + Held), `tests/core/aggregator_latest_test.cpp` (Latest) |
| **`Sync` element (11.4)** — generic `N→N`; `policy` property; claims a slot; stamps all N outputs alike (common-ancestor injection) | `Sync`, `src/plugins/core/sync.cpp` | `tests/plugins/core/sync_threaded_test.cpp` (two independent live streams → Sync → downstream Barrier) |
| **`Sync.policy` dispatch (11.1)** — aggregator reads the join's `policy`; `latest` uses `ThreadBoundaryRuntime::boundary_try_pull` | `run_consumer_segment` policy block | `aggregator_latest_test.cpp` |
| **Safe-point control plane** — segment threads apply pending edits at a between-buffer safe point (forward-apply); UI-read serialization via the element mutex | `Pipeline::SegmentSafePoint`, `PipelineSession::apply_commands_for` | `tests/core/threaded_safe_point_test.cpp` |
| **Pause / resume** — segments park at the safe point; the source stops, the pipeline freezes | `PipelineSession::request_pause/resume/wait_while_paused` | `tests/core/threaded_pause_test.cpp` |

**Where `run_threaded` engages.** `Pipeline::should_run_threaded()` is true iff
there is a **live source AND ≥1 `Queue`** (so the graph decomposes into >1 segment).
Offline pipelines and Queue-free live pipelines stay on the single-threaded sweep.
The CLI headless `run` and the session both honor this gate.

**Generation boundary (drain/flush) — intentionally not separately implemented.**
The vector clock subsumes it: a join matches by **ancestor identity**, which is
config-independent, so a mid-run property change forward-applies without mixing
generations. The `QueueEpochPolicy` *drain/flush-old-config* knob is therefore an
optional policy (the safe default — let in-flight buffers through — is correct live
behavior), not a correctness requirement. See §11 and the runtime notes in §13.

Out of scope (unchanged): bundles / `Mux` / `Demux` (dropped — superseded by the
`Sync` element); CPA/report (Phase 28); overlay plots (Phase 29); Kyber (Phase 30+).

Design of record: Sections 10, 11, 13 of this document.

---

## 13. Runtime control — the player state machine and the safe-point control plane

This section describes how a live pipeline is **driven and edited at runtime** (the
`--graph` GUI and, partially, the headless CLI). It is GStreamer-inspired but
simplified to what LeakFlow needs.

### 13.1 The four states

A session is in exactly one **streaming-lifecycle** state. `Paused` and `Idle`
differ *only* by whether there is more to stream.

| State | Producing? | Resumable? | Cache held? | An edit… |
|---|---|---|---|---|
| **Stopped** | no | — | no (torn down) | only stages for the next Start; nothing fires |
| **Running** | yes (streaming) | — | yes | forward-applies on the next buffer |
| **Paused** | no (frozen) | **yes** — a live source still has more | yes | is set on the element immediately (Auto) or staged (Manual); the *stream* is frozen, so the rendered result updates on Resume |
| **Idle** | no | **no** — single run done, or the live source hit EOS | yes | recomputes downstream from cache if replay-safe; otherwise rejects |

`Stopped` is "dead." `Paused` and `Idle` are both "alive" (resources up, cache
held); the only difference is whether **Resume** is available.

### 13.2 Transitions and buttons

```
Stopped --Start--> Running --Pause--> Paused --Resume--> Running
                      |                                     |
                      └----------- EOS / sweep done --------┴--> Idle
   Running/Paused/Idle --Stop--> Stopped       Idle --Start--> Running (re-run from sources)
```

| Button | Enabled in | Effect |
|---|---|---|
| **Start** | Stopped, Idle | run, or (in Idle) re-run from sources — replaces the old "Re-run from sources" |
| **Stop** | Running, Paused, Idle | full teardown → Stopped |
| **Pause** | Running | freeze (park the segments) → Paused |
| **Resume** | Paused | continue streaming → Running |

The graph opens in **Stopped** (the topology is drawn immediately so it isn't
blank); pass **`--auto-start`** to begin Running on open. A finite live stream
reaching EOS lands in **Idle** (held, inspectable), not Stopped.

Mapping to GStreamer: `Stopped ≈ NULL`, `Running ≈ PLAYING`, `Paused ≈ PAUSED`
(clock stopped — here the source's pacing), `Idle ≈ a held PAUSED with no preroll`.
"Start after EOS = re-run" is GStreamer's seek-to-0 (we already replay via the
source resetting its cursor).

### 13.3 The safe-point control plane (how edits apply during a threaded run)

Segments stream continuously, so there is no "between sweeps" boundary. Instead each
segment thread reaches a **between-buffer safe point** (`Pipeline::SegmentSafePoint`,
called by `run_source_segment` / `run_consumer_segment`) where it:

1. **parks while paused** (`wait_while_paused` / `wait_paused_tick`), and
2. **applies pending property edits for its own elements** on its own thread
   (`PipelineSession::apply_commands_for`).

Because the write happens on the *same thread* that reads the property in
`process()`, there is **no data race and no per-element lock**. A session-level
`control_mutex_` serializes the session counters across segment threads; the graph
also passes its `element_mutex` so control-window *reads* serialize against these
*writes*.

Property changes are **forward-only** during a threaded run (`forward_only_apply_`):
the change lands on the next buffer; there is no rerun. While **Paused**, the safe
point keeps applying edits (so in Auto mode the value is set immediately even though
the stream is frozen).

### 13.4 Auto-apply vs. Manual — orthogonal to the state

A single toggle (UI checkbox **"Auto-apply edits"**) controls *when* an edit is
applied, in **every** state — it is orthogonal to Stopped/Running/Paused/Idle:

- **Auto** (default): a submitted edit goes straight to the live queue and applies
  immediately (the next buffer in Running; the held frame in Idle; the element value
  in Paused). In Idle, applying to the held frame requires replay-safety.
- **Manual**: a submitted edit is **staged** (`PipelineSession::set_manual_apply` →
  `submit` routes to `staged_`); the **Apply** button flushes staging into the live
  queue (`apply_staged`). This lets you batch several edits and apply them together,
  in any state.

The toggle therefore lives at *submit time*, not at draining time — draining is
uniform everywhere.

### 13.5 Property effect by state (summary)

| Element kind / state | Running | Paused | Idle | Stopped |
|---|---|---|---|---|
| live-driven element (e.g. source `trace_rate`) | forward → next buffer | set now (Auto) / staged (Manual); effect on Resume | **recompute from cache if replay-safe; otherwise reject** | stage for next Start |
| one-run / offline element | n/a (forward) | n/a | **recompute from cache if replay-safe; otherwise reject** | stage for next Start |
| **restart-scoped structural** — `Queue.max_size`, `Sync.policy` | value sticks, but effect only at the **next Start** | value sticks; effect at next Start | takes effect at next Start | takes effect at next Start |

These are intentionally **restart-scoped** (read once when the run starts):

- **`Queue.max_size`** — the Queue-owned `BufferQueue` capacity is fixed per run;
  resizing a live queue would drop/reorder in-flight buffers.
- **`Sync.policy`** — the aggregator reads the join's policy once and fixes each
  input's mode (Driving / Held / Latest) before streaming. Re-pairing mid-stream
  would mishandle buffers already held at the queue heads (a `Held`/`Latest` wildcard
  becoming `Driving`, etc.), and a pairing change *is* a new output generation. So
  edit it in Stopped/Idle, then **Start**. (You may type a new value while
  Running/Paused — it sticks — but it bites on the next Start.)

### 13.6 Implementation pointers

- State + pause primitive: `PipelineSession` (`state()`, `set_state`,
  `request_pause/resume`, `wait_while_paused`, `wait_paused_tick`).
- Request-driven worker + buttons: the session worker and toolbar in
  `src/plot/pipeline_graph.cpp` / `src/plot/imgui_plot_loop.cpp`;
  `PipelineControlRuntime` (`request_start/stop/pause/resume/apply`,
  `set_run_stopper`, `auto_recompute`).
- Numeric editors commit on Enter/focus-out (never per keystroke); int properties
  have explicit `-`/`+` step buttons.

---

## 14. CLI cookbook — drive every feature from the command line

Recipes to *experiment* with sync, queues, threads, states, drops, and provenance.
All use the checked-in fixtures (no hardware):

- `tests/fixtures/aes/sync/key_01/traces_first_50.pt` — a `[50, 5000]` trace file.
- `tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt` — a `[50, 16]` file.

Build first: `cmake --build build -j`. Add `--log-level info` to watch buffers flow
(`--log-level warn` is quieter). Add `--graph` for the interactive window.
Runtime telemetry defaults off for headless runs and on for `--graph`; override
with `--telemetry` or `--no-telemetry`.

### 14.1 Offline (no live source) — one sweep

```bash
./build/leakflow run 'FakeSrc ! Summary ! FakeSink'
```

### 14.2 A live source — pacing and EOS

`FakeLiveSrc` streams one buffer per axis-0 row, then EOS. `trace_rate` paces fake-live
replay (one trace per `1/rate` seconds). It does not stamp `capture.sample_rate_hz`;
add that as explicit source metadata when the trace sampling rate matters:

```bash
# 50 rows, ~2 traces/sec -> ~25 s, visible in the log
./build/leakflow --log-level info run \
  'FakeLiveSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt, trace_rate=2.0) ! Summary ! FakeSink'

# no rate -> as fast as possible
./build/leakflow run \
  'FakeLiveSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Summary ! FakeSink'
```

### 14.3 Live + Queue = threaded segments (the core live path)

A `Queue` cuts the pipeline into segments and owns the boundary `BufferQueue`
exposed by `ThreadBoundaryRuntime`. If the downstream consumer segment is fed only
by active-capable boundaries and has no internal passive source, those Queue
source tasks drive it through the runtime push collector. Otherwise the older
pull-side consumer segment remains a fallback; this includes CPA-style graphs
where live traces arrive through Queues but `TorchFileSrc@key_src` feeds
`AttackStats.truth` directly inside the same downstream segment.
Graph/session runs use the same active path; active tasks call safe points so
pause and property changes remain explicit.

```bash
# Block queue (drop_oldest=false) = lossless backpressure: all 50 rows arrive
./build/leakflow --log-level info run \
  'FakeLiveSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Queue(max_size=8, drop_oldest=false) ! Summary ! FakeSink'
```

**Lossy capture** (drop oldest under backpressure) — for hardware you cannot slow:

```bash
# small, fast, dropping -> far fewer than 50 survive (capture-style)
./build/leakflow --log-level warn run \
  'FakeLiveSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt, trace_rate=200.0) ! Queue(max_size=1, drop_oldest=true) ! Summary ! FakeSink'
```

### 14.4 Cooperative stop (Ctrl+C)

Start a paced live+Queue run and press **Ctrl+C**: it drains and exits in a fraction
of a second (it does not wait out the remaining traces, nor hard-kill):

```bash
./build/leakflow --log-level info run \
  'FakeLiveSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt, trace_rate=1.0) ! Queue(drop_oldest=false) ! Summary ! FakeSink'
# ...press Ctrl+C mid-stream
```

### 14.5 The graph — player controls and live editing

```bash
# opens in Stopped; click Start. Try Pause/Resume, edit trace_rate live, toggle Auto-apply.
./build/leakflow run --graph \
  'FakeLiveSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt, trace_rate=1.0) ! Queue(drop_oldest=false) ! Summary ! FakeSink'

# begin running immediately on open
./build/leakflow run --graph --auto-start \
  'FakeLiveSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt, trace_rate=1.0) ! Queue(drop_oldest=false) ! Summary ! FakeSink'
```

In the graph: change `trace_rate` while **Running** → re-paces on the next
buffer; **Pause** → the stream freezes; in **Idle** (after EOS) edits to downstream
elements recompute from cache only when the replay-set is replay-safe; otherwise
the command is rejected and the old held view stays consistent. The **Vector
Clock** and **Buffer Clocks** panels show provenance per slot and per link.

### 14.6 Sync — pairing two independent live streams (fan-in / fan-out)

Two unrelated sources have no shared clock slot; `Sync` injects a **common ancestor**
so a downstream Barrier join pairs them for free. CLI fan-in needs the inputs
declared **before** the join (source-before-sink), then `element.pad` addressing:

```bash
./build/leakflow --log-level info run \
  'FakeLiveSrc@a(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Queue@qa(drop_oldest=false) ;
   FakeLiveSrc@b(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt) ! Queue@qb(drop_oldest=false) ;
   @qa.src ! Sync@s.in_0 ;
   @qb.src ! @s.in_1 ;
   @s.out_0 ! Summary ! FakeSink ;
   @s.out_1 ! Summary ! FakeSink'
```

`Sync@s.in_0` *creates and addresses* a pad inline; `@s.in_1` / `@s.out_0` reference
the same instance. Note the idiom: **declare each input chain first, then the join.**

### 14.7 Sync policy — Latest (sample-newest)

Set `Sync(policy=...)` to choose how it pairs. The five policies:

- **symmetric** (all inputs treated alike — no per-pad config): `barrier` (default,
  match by vector-clock identity), `zip` (pair by arrival position, 1:1), and
  `all-required-once` (fire once when all present).
- **asymmetric** (one input drives, the rest are reused/sampled): `held` (reuse a
  static input as a wildcard) and `latest` (sample the **newest**, dropping
  intermediates, never waiting).

For the asymmetric policies the **driver is `in_0`** (the lowest-numbered input pad)
by convention; `in_1`, `in_2`, … are the held/sampled inputs. There is **one policy
per `Sync`**, not per pad, and it is **restart-scoped** (read once per run — see
§13.5). `latest` example (traces drive `in_0`, plaintexts sampled on `in_1`):

```bash
./build/leakflow run \
  'FakeLiveSrc@a(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! Queue@qa(drop_oldest=false) ;
   FakeLiveSrc@b(path=tests/fixtures/aes/sync/key_01/plain_texts_first_50.pt) ! Queue@qb(drop_oldest=false) ;
   @qa.src ! Sync@s(policy=latest).in_0 ;
   @qb.src ! @s.in_1 ;
   @s.out_0 ! Summary ! FakeSink ;
   @s.out_1 ! Summary ! FakeSink'
```

### 14.8 Inspecting threads and provenance

Run any of the above with `--graph` and open the **Buffer Clocks (per link)** panel:
each buffer's vector clock is decoded as `element=count` for every non-zero slot, so
you can watch the clock ride through the `Queue` untouched and see the `Sync` slot
appear on `out_0`/`out_1`. The **Vector Clock (provenance counts)** panel lists the
running production count per slot. The graph's element tooltips show each element's
claimed clock slot(s).

> Tip: the number of segment threads = connected components after cutting at every
> `Queue`. One `Queue` → two threads; two queues into one join → three threads (two
> producers + the shared consumer). See §10.
