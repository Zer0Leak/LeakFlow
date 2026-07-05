# Pipeline Controller, Session, and Incremental Rerun (Phase 25)

This document is the authoritative design for the Phase 25 control/session layer.
It records the decisions (labelled `D1`..`D12`) made during phase planning, the
rationale behind each, and the contracts implementations must preserve.

`ROADMAP.md` defines the phase goal and exit criteria. This file is the design
source of truth; `docs/context/ARCHITECTURE_CONTRACTS.md` carries the compact
day-to-day version.

## 1. Problem

Before Phase 25, live property control lived in `leakflow_plot`
(`PipelineControlRuntime`) and mutated live `Element` objects directly through a
`weak_ptr`. The synchronous executor (`Pipeline::run()`) was self-contained
(start all → one sweep → stop all), and `leakflow run --graph` ran it once on a
worker thread while the UI was forced to *pause edits while the worker ran*.

That gives no safe way to:

- apply a property change without racing the executor,
- recompute only the part of the graph affected by a change,
- distinguish results produced before and after a change,
- prepare for future live/streaming sources.

Phase 25 introduces a control/session layer that owns safe command application,
cached buffers, downstream-only rerun for synchronous pipelines, and the epoch
semantics that future live mode will need — without turning `leakflow_core` into
a GUI or a general async scheduler.

## 2. Two planes

The design keeps two planes strictly separate:

- **Observation plane** — copied, SCA-safe snapshots and events
  (`PipelineObserver` / `PipelineEvent`). Never exposes mutable `Element`
  handles, mutable `Buffer` handles, payload pointers, raw traces, keys, or
  plaintext. Read-only UI state transfer.
- **Control plane** — an explicit command API. The UI submits validated
  commands; the session applies them at safe points and reports
  accepted/rejected/applied through the observation plane.

## 3. Decisions

### D1 — One executor, two entry points

The executor is a **single engine**. "Full run" and "partial rerun" are the same
per-element step engine seeded differently, never two divergent code paths.

Grounding: the real engine is `Pipeline::run_linked_graph()` — a single
forward sweep over elements in insertion order (which `link()` guarantees is
topological because it enforces `source_index < sink_index`). Per element it
gathers `ElementInputs` (a `map<pad_name, optional<Buffer>>`), calls
`process_inputs(...)`, then routes the single output to each outgoing link. The
old recursive `run_from()` was **dead code** and is removed.

Refactor:

- Extract the per-element step (gather inputs → `process_inputs` → route outputs →
  annotate / log / emit / cache) into a reusable primitive.
- `start_all()` / `run_sweep()` / `rerun_from(element)` / `stop_all()` are public
  primitives.
- `run()` becomes thin sugar: `start_all(); auto o = run_sweep(); stop_all(); return o;`
  Kept only as a convenience for engine unit tests and trivial callers; the app
  uses the session.

### D2 — `PipelineSession` lives in `leakflow_core`

The controller/session is a new type in `leakflow_core`, sitting **above/around**
`Pipeline`. The cache and the rerun primitive live **inside**
`Pipeline` (they need the private link/pad/step internals); the session is
thin on top (command queue, `PropertyEffect` dispatch, epoch counter, state
machine, copied events).

Rejected alternatives:

- Methods directly on `Pipeline` (fuses graph + executor + command queue +
  cache + epoch + state into one class).
- A type *above* core in `leakflow_plot` / a new target. Rejected because the
  rerun engine needs `Pipeline`'s internals, and because Phase 25's exit
  criteria require **headless** `tests/core` tests of rerun/validation that must
  not link a GUI.

`PipelineControlRuntime` in `leakflow_plot` becomes a UI client of the session;
it keeps its widget state but routes edits to `session.submit(SetProperty{...})`
instead of mutating elements.

### D3 — Element rerunnability contract + `can_replay()`

Two-clause contract on every `Element`:

1. **Replay purity** — between `start()` and `stop()`, `process_inputs(inputs)`
   must be a deterministic function of `(inputs, current property values)`.
   External side effects are allowed only if they are idempotent on replay
   (overwrite, not append — e.g. plot snapshots keyed by element name, file sinks
   truncating).
2. **Lifecycle reset** — `start()` must (re)initialize all per-run mutable state
   to a clean baseline, so a `stop_all → start_all → run_sweep` cycle is a true
   reset.

Signal: `virtual bool Element::can_replay() const { return true; }`. Stateful
elements override to `false`. `Queue` is the only current `false` (it accumulates
`buffers_`). The value is mirrored into `ElementDescriptor` so `leakflow-ls` can
show it; the runtime decision queries the live element.

Effective behaviour of a cached-replay property change:

```
rerun_from(X) is allowed  ⇔  every element in X's replay-set has can_replay()==true
otherwise                 ⇒  reject the command before committing the property
```

Restart-scoped lifecycle/structural changes are separate: they are not cached
replays, so they are applied for the next start/restart path according to their
declared `PropertyEffect`.

`Element` is the right home for `can_replay()` (not `PropertyEffect`) because
replayability is a property of the element, independent of which upstream
property changed.

### D4 — caps-output changes are validated transactionally

A `caps-output` property change may make a downstream link type-incompatible.
The session **validates the new caps against downstream links before committing**
the property. If a link would break, the command is **rejected**
(`CommandRejected{which link}`) and the pipeline stays consistent. No half-applied
invalid state.

### D5 — Epochs, `Buffer::epoch()`

An **epoch** is a version stamp on the pipeline's configuration generation: a
monotonic counter answering "which configuration produced this buffer?". It
exists so old and new data are never silently confused — critical for
reproducible SCA.

Decisions:

- The session owns a monotonic epoch counter, bumped on each accepted **dataflow**
  command (`metadata-output` / `payload-output` / `caps-output`) and on full
  restarts. UI-only / sink-display changes do not bump it. The counter is
  **monotonic for the whole session lifetime** — even a stop-start cycle keeps
  climbing, so a generation number is never reused.
- `Buffer` carries the epoch: `Buffer::epoch()` / `Buffer::set_epoch(...)`. The
  **executor is the single writer** — it stamps `current_epoch` on every routed /
  produced buffer; elements never set it. Default `0` means "unversioned"
  (the plain `run()` path and freshly constructed buffers).
- Routed-buffer observations carry the epoch too, so the UI can drop stale
  observations.
- `QueueEpochPolicy` is defined as a documented enum
  (`Drain`, `Flush`, `KeepMixed`, `Block`, `DropOldest`, `DropNewest`) as a
  **contract only**. `Queue` stays synchronous; no threaded draining is wired.

`Buffer::epoch()` (rather than side-band bookkeeping) is the definitive home:
future live/threaded `Queue` must read the epoch off in-flight buffers, which
side-band cannot serve. It is also cleaner now — the cache is `map<PadKey, Buffer>`
with no wrapper, and observations read `buffer.epoch()` directly.

### D6 — Safe points and command/event plumbing

The hazard: properties are read by the worker thread inside `process_inputs()`.
Mutating them from the UI thread mid-execution is a data race.

A **safe point** is a quiescent boundary in the worker's loop — *between units of
work*. Offline that boundary is *between sweeps*; live (future) it is *between
buffers*. Same mechanism, two cadences.

Mechanism:

- The UI thread never touches elements. It enqueues commands onto a
  **thread-safe queue** owned by the session, with **last-wins coalescing per
  `(element, property)`** (so a dragged slider applies only its final value).
- The worker drains and applies commands **at the safe point** — the only place
  live elements are mutated.
- Results flow back through the **single existing `PipelineObserver` stream** as
  new `PipelineEvent` variants: `CommandAccepted`, `CommandRejected`,
  `CommandApplied`. One UI drain point; one ordering. Command events are copied
  and SCA-safe (identity, property, stringified old/new, effect, epoch).

Invariant:

```
commands : UI thread  ── enqueue ──▶ session command queue   (only write path down)
events   : worker      ── emit ────▶ observer stream → UI     (only read path up)

Only the worker thread ever touches live elements or runs the pipeline.
The UI thread only enqueues commands and drains copied events.
```

The worker is the background `std::jthread` that runs pipeline execution so the
main thread can keep drawing the GUI. Phase 25 promotes it from a one-shot runner
into the **persistent session loop**:

```
worker thread (lives for the whole interactive session):
    session.start_all();                 // once
    loop until window closes / stopped:
        drain_commands_at_safe_point();  // apply queued SetProperty here
        drive.produce_next_unit();       // offline: one sweep; live: one buffer
        emit observations;
    session.stop_all();                  // once, at teardown
```

#### StreamingDrive seams (reserved, not wired in Phase 25)

The persistent loop uses a pluggable **drive policy** so live mode slots in
without reshaping the command/event/epoch machinery:

- `OneShotDrive` (built now): sources emit once, one sweep, finish; the worker
  then idles at the safe point servicing commands until the window closes.
- `StreamingDrive` (reserved): loop pulling from a live source until stopped.

Reserved but defined: `StreamingDrive`, `QueueEpochPolicy` enforcement in a
threaded queue, and a **cooperative cancel** that honours the worker's
`std::stop_token` so a blocking source read can unwind to a safe point. The
token already exists in the worker signature, currently unused.

### D7 — Per-input-pad and per-output-pad cache

When caching is enabled, the executor caches the latest accepted **input buffer
per element input pad** and the latest **output buffer per element output pad**.

Per-input-pad granularity is **mandatory**, not optional: multi-input elements
(e.g. `PearsonCorrelator` with `features` + `targets`) reprocessing during a
partial rerun must take the unaffected pad's buffer from the cache while the
affected pad's buffer comes fresh from the rerun frontier. The executor already
gathers inputs keyed by pad (`inputs_by_element`) and asserts
`inputs.size() == incoming.size()`, so a cached input is required to satisfy that
assertion on the frontier boundary.

Caching is a **global on/off toggle** (`session.set_caching_enabled(bool)`,
applied at a safe point), default **ON**:

- **ON** — partial rerun is available; payloads of cached buffers are pinned (see
  §4).
- **OFF** — nothing retained after a sweep; partial rerun is disabled and a
  dataflow change **falls back to a full re-sweep from sources**. Lower memory.

The contract is shaped so a future **per-pad / per-element opt-out** can be added
without breaking callers.

### D8 — Command surface

The only **queued** command in Phase 25 is `SetProperty{element, property, value}`.
No add/remove/relink commands (they touch topology and pull toward dynamic pads,
which are out of scope).

Everything else is a **direct session control applied at a safe point**, not a
queued element command: full restart, re-run-from-sources, the caching toggle,
and live start/stop. UI buttons submit these session controls; the worker applies
them at the next safe point with the same discipline as commands.

### D9 — `PipelineSession` owns the `Pipeline`

The session **owns** the pipeline by move-in:

```cpp
auto pipeline = cli::build_builtin_pipeline_from_expression(expr, ...);
PipelineSession session(std::move(pipeline));
```

The session is the single observe-and-control handle for the application; it
drives lifecycle, caching, the state machine, and epochs, so split ownership
would invite lifetime hazards. App UI entry points take `PipelineSession&`
instead of `Pipeline&`. The bare `Pipeline` stays independently
constructible for engine unit tests.

### D10 — Lifecycle ownership (Option A)

The session owns lifecycle across reruns: `start_all()` **once** at session
begin, `run_sweep()` / `rerun_from()` **many** times with no intervening
`stop_all()`, and `stop_all()` once at teardown. This is required because partial
rerun reprocesses already-`start()`-ed elements; re-bracketing each rerun with
stop/start (Option B) was rejected as error-prone.

`Pipeline::run()` keeps its self-contained start→sweep→stop semantics for
one-shot callers and tests, implemented over the same primitives (D1).

### D11 — Session state machine, graceful stop, button set

The lifecycle state machine lives on the **session, not on every `Element`**.
Elements stay simple (`start` / `process_inputs` / `stop` + the D3 reset
contract).

```
Stopped   — constructed, no resources
Started   — start() done, resources acquired, ready but not streaming
Running   — actively executing units
[Paused]  — started, resources held, stream suspended (reserved, live-only)
```

Built now: `Stopped ↔ Started ↔ Running`. Reserved (designed into the enum,
wired with StreamingDrive): `Paused` + pause/resume. Preroll is a future
refinement of entering `Paused` (optionally pull one buffer so sinks show first
data) — not built.

**Graceful stop is two-step** so an element can stop nicely (close a connection,
flush):

1. Request stop — signal the worker `std::stop_token` so a blocked source read
   returns and the loop unwinds to a safe point.
2. `stop_all()` in reverse order — each `Element::stop()` releases resources.
   Elements must remain re-`start()`-able afterward.

Graph UI control buttons map to primitives:

| Button | Maps to | Transition |
|---|---|---|
| re-run from sources | `run_sweep()`, elements stay alive | Running→Running, new epoch |
| stop-start cycle | `stop_all()` → `start_all()` → `run_sweep()` | full reset (D3 escalation path) |
| start (livestream) | `start_all()` then begin `StreamingDrive` | Stopped→Running |
| stop (livestream) | cooperative cancel → join → `stop_all()` | Running→Stopped, graceful |

Automatic downstream-only partial rerun (after a property edit) sits underneath
these; `re-run from sources` is the manual full-pass override.

### D12 — Error handling during a rerun

A partial rerun can throw mid-path (e.g. changing `AesLeakage.channels` reshapes a
tensor and a downstream element throws). The session must **not** tear down the
interactive session over one bad edit.

- The exception is **caught inside the rerun**; `stop_all()` is **not** called —
  Option A lifecycle stays intact.
- A `CommandApplied{status=failed}` (carrying the error text) is emitted through
  the observer stream so the UI shows what broke.
- The affected output pads' cache entries are **marked invalid/stale**; no
  half-written new buffers masquerade as good. The last good downstream
  observations remain what the UI shows.
- The epoch **still bumps on accept** (the configuration generation did change);
  the failed path simply has no valid epoch-N output.
- The user can change the property again to recover; no restart is required.

This contrasts with `Pipeline::run()`, which (correctly for a one-shot run)
emits `Error`, stops all elements, and rethrows.

## 4. Buffer ownership and the cache

`Buffer` is a value type: it owns `Caps` and `metadata` by value and holds the
payload as `std::shared_ptr<Payload>`. `payload_is_unique()` is
`payload_ != nullptr && payload_.use_count() == 1`. The mutability signal lives on
the **payload**, not the buffer.

Consequence for caching: the cache stores `Buffer` by value, so it holds an extra
`shared_ptr<Payload>`. That raises the payload `use_count`, so any downstream
element calling `mutable_payload_if_unique<T>()` sees a shared payload and takes
the copy-on-write / replace path. This is **the cache-safety mechanism**, not a
side effect: a cached input buffer must never be mutated in place or the cache is
corrupted on rerun.

Rules:

- The cache stores `Buffer` by value and **hands out copies on rerun** — it never
  moves its stored buffer out. Repeated reruns stay valid; consumers always see a
  non-unique payload and replace rather than mutate.
- Uniqueness is payload-granular and conservative (two cached buffers may share
  one payload, as Tee already does). Caching makes this conservatism
  by-construction.
- Caching pins payloads (large trace tensors) for the session lifetime. This is
  the accepted memory cost of incremental rerun (D7).

At a join during offline rerun, combining a fresh current-epoch output with a
still-valid older-epoch cached input is **correct** — the unchanged branch's data
is still good; the join's output takes the new (max) epoch. This is distinct from
live "silent mixing", which `QueueEpochPolicy` governs.

## 5. Replay-set and escalation

A rerun is triggered only by a `SetProperty` whose `PropertyEffect` is a dataflow
effect. For a change on element `X`:

```
replay-set = X  +  every element reachable downstream of X's affected output
             pad(s), following links forward (through Tee forks, into joins)
```

`PropertyEffect.output_pads` narrows the starting pads (e.g. `AesLeakage.channels`
affects only `leakage`).

```
partial rerun_from(X) is allowed  ⇔  every element in the replay-set has can_replay()==true
otherwise                         ⇒  reject the property change transactionally
```

A non-replayable element **upstream** of `X` never forces escalation (it is never
in the replay-set; its cached output is reused). It rejects the cached replay only
when it is `X` itself or reachable downstream of `X`.

Rejected replay = the property is not committed, no element is reprocessed, and
the configuration generation is not bumped. A user can still stop/restart or
change the relevant element to a replayable mode when that is available.

## 6. What is built vs reserved

Built in Phase 25:

- `start_all` / `run_sweep` / `rerun_from` / `stop_all` primitives; `run()` as
  sugar; dead `run_from()` removed.
- `Buffer::epoch()` + executor single-writer stamping.
- `Element::can_replay()` (default true) + descriptor mirror; `Queue` returns
  false and resets `buffers_` in `start()`.
- Per-input-pad / per-output-pad buffer cache; global caching toggle (default ON).
- `PipelineSession` in core: command queue with last-wins coalescing, safe-point
  application, `SetProperty` validation, transactional caps validation,
  downstream-only rerun with escalation, rerun error handling, monotonic epoch,
  `Stopped/Started/Running` state machine, session controls (restart,
  re-run-from-sources, caching toggle), copied `CommandAccepted/Rejected/Applied`
  events.
- `OneShotDrive`; `QueueEpochPolicy` enum (definition only).
- `PipelineControlRuntime` becomes a session UI client; graph control buttons;
  `run_pipeline_graph_until_closed(PipelineSession&, ...)`.
- Headless `tests/core` for command validation, property-change events,
  downstream invalidation, metadata-only rerun, payload-output rerun, caps-output
  rejection, source-output rerun with cached buffers, and epoch behaviour.

Reserved (defined, not wired):

- `StreamingDrive`, threaded `QueueEpochPolicy` enforcement, cooperative cancel,
  `Paused` + pause/resume, preroll.

Out of scope (unchanged from the roadmap): a general async scheduler, dynamic
pads, graph-wide negotiation beyond caps revalidation, dynamic plugin loading,
new AES/Kyber/plot algorithms, hardware capture, headless GUI rendering tests.
