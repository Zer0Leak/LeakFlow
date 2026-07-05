# Dataflow & Synchronization — Worked Walkthroughs (validation)

Status: stakeholder-facing validation of the dataflow/sync/liveness design. This
document *exercises* the model on concrete pipelines and shows, step by step, the
threads, the join behavior, and the vector-clock indexes. The **rules** live in
`dataflow_sync_model.md` (the design of record); this document proves they work.

If a reviewer can follow these ten walkthroughs and agree each one behaves
correctly, the whole model is validated. It was the implementation reference for the
live phase, which is **now implemented** — see `dataflow_sync_model.md` §12
(implementation status + tests), §13 (the runtime player/control model), and §14 (a
CLI cookbook to run these scenarios yourself).

---

## 0. How to read this

**Notation**

- A buffer's **vector clock** is written `{slot:count, …}`. Slot 0 is reserved;
  zero slots are omitted (they are wildcards in matching). Example: `{1:3, 4:3}`
  means element-of-slot-1 has emitted 3 and element-of-slot-4 has emitted 3 in
  this buffer's lineage.
- **Slot allocation:** every element that produces gets one slot at `add()` time
  (Tee and sinks get **0 slots**). We label each element with its slot in the
  topology.
- **Fold-match (the default join):** at a multi-input element, for every slot that
  is non-zero in **two or more** inputs, the values must be **equal**; otherwise
  the element waits. No shared non-zero slot ⇒ nothing to match ⇒ fires freely.
  On firing, the output clock is the component-wise **max** of the inputs, then the
  element **increments its own slot**.
- **Threads / segments:** cut the graph at every `Queue`; each remaining connected
  component is one **segment** = one **thread**. Sources and Queues are the only
  thread boundaries.
- **Run:** `run()` is one pump loop — `start_all → pump-until-EOS → stop_all`. A
  source returns `Data` / `NoData` (transient) / `EOS` (terminal). A one-run source
  emits `Data` once then `EOS`; a live source streams then `EOS`.

**The four rules being validated** (from `dataflow_sync_model.md` §11):

- **A.** Default sync = per-slot Barrier (the fold-match). Common-origin synced,
  independent fires free.
- **B.** One counter per slot; no global version (liveness dissolves the need).
- **C.** Liveness-aware property change: live-driven ⇒ forward; one-run-driven ⇒
  re-emit from cache.
- **D.** The Sync element is the sole front door for custom cross-source pairing
  (it injects a common ancestor).

---

## 1. Baseline — a linear one-run pipeline

```
FakeSrc(slot 1) → Summary(slot 2) → FakeSink(0 slots, sink)
```

**Threads.** No Queue ⇒ one segment `{FakeSrc, Summary, FakeSink}` ⇒ **1 thread.**

**Run (pump loop).**

| tick | FakeSrc | Summary | FakeSink | clock at Summary out |
|---|---|---|---|---|
| 1 | `Data` | processes | consumes | `{1:1, 2:1}` |
| 2 | `EOS` | — | — | run() returns |

FakeSrc emits one buffer `{1:1}`; Summary forwards and bumps slot 2 → `{1:1,2:1}`;
FakeSink consumes. On tick 2 the source is `EOS` ⇒ the pump ends ⇒ `run()` returns.
**This is exactly today's offline behavior** — a one-run source makes the pump a
single iteration.

✔ Validates: pump loop = offline one-shot when sources are one-run; basic index
increment.

---

## 2. Flagship — the AES PoI diamond (offline): common-origin barrier + partial rerun

This is the real pipeline. It proves the default barrier and partial rerun.

```
TorchFileSrc(traces)     [slot 1] ─► Tee [0 slots] ┬─► TracePlot.sink   [0 slots, sink]
                                                   ├─► PearsonCorrelator.features
                                                   └─► AesLeakage.traces
TorchFileSrc(plaintexts) [slot 2] ─► AesLeakage.plaintexts
TorchFileSrc(key)        [slot 3] ─► AesLeakage.keys
AesLeakage               [slot 4] ─► PearsonCorrelator.targets
PearsonCorrelator [slot 5] ─► PoiSelect [slot 6] ─► CorrelationPoiToPlotAnnotations [slot 7] ─► TracePlot.annotations
```

**Threads.** No Queue ⇒ **1 thread** (offline, synchronous).

**Index trace (one sweep).**

| element | inputs (clocks) | fold-match | output clock |
|---|---|---|---|
| traces_src | — | — | `{1:1}` |
| Tee | `{1:1}` | — (copies, 0 slots) | three branches each `{1:1}` |
| plain_src | — | — | `{2:1}` |
| key_src | — | — | `{3:1}` |
| AesLeakage | traces `{1:1}`, plain `{2:1}`, keys `{3:1}` | no shared non-zero slot → fires | `{1:1, 2:1, 3:1, 4:1}` |
| PearsonCorrelator | features `{1:1}`, targets `{1:1,2:1,3:1,4:1}` | **slot 1: 1==1 → match** (2,3,4 wildcard on features) | `{1:1,2:1,3:1,4:1,5:1}` |
| PoiSelect | `{1:1,…,5:1}` | single input | `{1:1,…,5:1,6:1}` |
| ann | `{1:1,…,6:1}` | single input | `{1:1,…,6:1,7:1}` |
| TracePlot | sink `{1:1}`, annotations `{1:1,…,7:1}` | **slot 1: 1==1 → match** | renders |

**Why this is the proof.** `TracePlot.sink` (raw traces) and `TracePlot.annotations`
(derived) both carry **slot 1 = traces_src's count**. The fold-match on slot 1 **is
the barrier**: it guarantees the annotations were derived from *these* traces. The
derived elements' slots (4, 5, 6, 7) are wildcards on the raw branch, so they never
interfere. No tuning, no configuration — the diamond just synchronizes.

**Partial rerun (change `AesLeakage.channels`).** `rerun_from(AesLeakage)`; replay
set `{AesLeakage, PearsonCorrelator, PoiSelect, ann, TracePlot}`. The traces branch to
`TracePlot.sink` is **not** in the replay set → seeded from cache. (Offline, so
`PearsonCorrelator` is in recompute mode and replayable; changing `PoiSelect.top_k`
alone would replay only `{PoiSelect, ann, TracePlot}` from the cached correlation.)

| element | inputs | output clock |
|---|---|---|
| AesLeakage (reprocessed, cached inputs) | `{1:1}`, `{2:1}`, `{3:1}` | `{1:1,2:1,3:1, 4:2}` ← own slot bumped |
| PearsonCorrelator | features **cached** `{1:1}`, targets new `{1:1,2:1,3:1,4:2}` | slot 1: 1==1 → match → `{…,4:2,5:2}` |
| PoiSelect | `{…,5:2}` | `{…,5:2,6:2}` |
| ann | `{…,6:2}` | `{…,6:2,7:2}` |
| TracePlot | sink **cached** `{1:1}`, annotations new `{1:1,…,7:2}` | **slot 1: 1==1 → match** → renders cached traces + new annotations |

The reconfigured elements bumped slots 4/5/6 (their generation), but **slot 1
(traces_src) is unchanged on both branches**, so the barrier still matches and
`TracePlot` overlays the **fresh annotations on the stable traces**. This is the
"provenance becomes *checked*, not assumed" property.

✔ Validates: Rule A (common-origin barrier, free), partial rerun by vector,
own-slot bump on reconfigure (Rule B/C offline).

---

## 3. Independent sources, and the Sync element (Rule D)

**3a. Default — independent fires free.**

```
A [slot 1] ─► Z ;   B [slot 2] ─► Z      (Z [slot 3], default join)
```

`A`→`{1:1}`, `B`→`{2:1}`. At `Z`: **no shared non-zero slot** → fold is vacuous →
fires. Output `{1:1, 2:1, 3:1}`. Correct: with no common source there is nothing to
synchronize, so `Z` must not block. 99.99% of SCA lands here.

**3b. Forcing a pairing — insert a Sync element.**

You *want* `A[n]` paired with `B[n]` (e.g. two synchronized capture channels).

```
A [slot 1] ─►┐
             Sync [slot 3, policy=Zip] ─► out_a ─►┐
B [slot 2] ─►┘                          └► out_b ─► D [slot 4] (joins out_a, out_b)
```

Sync zips A and B by position `n`, and **stamps both outputs of fire `n` with its
own slot 3 = n** (like Tee stamps fan-out branches identically):

- `Sync.out_a` = `{1:n, 3:n}`   (A's clock + Sync's slot)
- `Sync.out_b` = `{2:n, 3:n}`   (B's clock + Sync's slot)

At `D`: shared non-zero slot = **slot 3** (both `3:n`) → match → fire. Slots 1 and 2
are wildcards. So **`D` uses the plain default barrier** and never knows A and B
were originally unrelated — the Sync element *injected a common ancestor* (slot 3),
converting case ③ (independent) into case ① (common-origin).

✔ Validates: Rule D — all custom-sync complexity localized in one opt-in element;
downstream stays default.

---

## 4. Live streaming — the pump loop and per-row indexes

`FakeLiveSrc` reads a `[50, 5000]` Torch tensor and emits **one buffer per axis-0
row**, then `EOS`.

```
FakeLiveSrc [slot 1, live] → AesLeakage-ish analysis [slot 2] → TracePlot [sink]
```

**Threads.** No Queue ⇒ **1 thread** (yes — *live does not require threads*; this is
single-threaded streaming).

**Run (pump loop).**

| tick | FakeLiveSrc | clock emitted | analysis out |
|---|---|---|---|
| 1 | `Data` (row 0) | `{1:1}` | `{1:1, 2:1}` |
| 2 | `Data` (row 1) | `{1:2}` | `{1:2, 2:2}` |
| … | … | … | … |
| 50 | `Data` (row 49) | `{1:50}` | `{1:50, 2:50}` |
| 51 | `EOS` | — | run() returns |

Same `run()` as Example 1 — it just iterates 50 times instead of 1, discovered from
the source's `EOS`. Each row's clock is `{1:k}`; downstream elements bump their own
slot per tick.

✔ Validates: one `run()` for one-run and live; live = source emits a stream; EOS
ends the loop; live ≠ threaded.

---

## 5. Live + Queue + threads — capture/analysis overlap, backpressure, stop

```
FakeLiveSrc [slot 1, live] ─► Queue ─► Analysis [slot 2] ─► FakeSink [sink]
```

**Threads.** Cut at the Queue:
- **Segment 1 = `{FakeLiveSrc}`** → **thread T1**.
- **Segment 2 = `{Analysis, FakeSink}`** → **thread T2**.

**Flow.**
- **T1:** `FakeLiveSrc` reads row `k`, **pushes `{1:k}` into the Queue** and returns
  (non-blocking, bounded). Loops. Between rows it polls with a timeout and checks
  the **stop token**.
- **T2:** **pulls** `{1:k}` from the Queue, `Analysis` processes → `{1:k, 2:k}`,
  `FakeSink` consumes.

The **Queue owns the boundary FIFO** (`BufferQueue`) but does not yet own a
source-side task; the current segment runner drives both sides: T1 pushes, T2
pulls.

**Backpressure.** If T2 is slower than T1, the Queue fills; T1's push **blocks**
(or **drops** per the Queue's policy). So either the source throttles to the
analysis rate, or — for a hardware capture you can't back-pressure — you set
**drop-oldest** and lose old traces rather than overflow.

**Cooperative stop.** `Ctrl+C` / window-close / `stop()` → `request_stop()`. T1's
`FakeLiveSrc`, on its next poll-timeout, sees the stop token → returns `EOS` and
unwinds. The pump drains the Queue, T2 finishes its in-flight buffers,
`stop_all` runs in reverse order, `run()` returns. Bounded by the poll timeout, not
the data rate.

✔ Validates: segment threading (threads = components cut at Queues), push/pull
through a Queue-owned boundary runtime, backpressure/drop policy, cooperative stop
via stop token + poll-timeout (= the `NoData` mechanism).

---

## 6. The hybrid — run-once config + live hardware (the case you raised)

```
A (config, one-run) [slot 1] ─► B (live; reads ChipWhisperer) [slot 2] ─► Queue ─► D [slot 3]
```

**Threads.** Cut at the Queue:
- **Segment 1 = `{A, B}`** → **thread T1**.
- **Segment 2 = `{D}`** → **thread T2**.

So **B does not get a thread *separate from* A** — they share T1. But `A` is
one-run: it emits its config once and hits `EOS`, after which **T1 is effectively
B's live loop**.

**Index trace on T1.**

| tick | A | B (held config `{1:1}` + hardware row k) | pushed to Queue |
|---|---|---|---|
| 1 | `Data` config `{1:1}` | reads HW → combines | `{1:1, 2:1}` |
| 2 | `EOS` | config **Held** `{1:1}`, HW row → | `{1:1, 2:2}` |
| 3 | (done) | `{1:1}` held, HW row → | `{1:1, 2:3}` |
| … | | | `{1:1, 2:k}` |

`A`'s slot stays `1:1` forever (it's a one-run input that B **Holds**); `B`'s own
slot increments per hardware read. `B` is **live-driven** (the hardware is the live
source); `A`'s config is the **Held** input — falling straight out of the "mixed
live/static ⇒ live-driven + static Held" rule.

**Answers to the specific questions:** B shares T1 with A (and owns it after A's
EOS); **B pushes into the Queue**; the **Queue does not pull from B** (D's thread
pulls); and **no Queue is needed between A and B** (A is instant, B Holds the
config — no rate mismatch, no thread boundary). Optionally model the hardware as a
separate `ChipWhispererSrc` and, *only if* analysis should overlap capture, put a
Queue between the capture and B.

✔ Validates: mixed one-run/live in one segment; Held config; push side of a Queue;
"no pointless Queue."

---

## 7. Live diamond — Tee, queues, and the aggregator fold-match

```
FakeLiveSrc [slot 1, live] ─► Tee [0 slots] ┬─► Queue_a ─► D.a
                                            └─► Queue_b ─► D.b      (D [slot 2])
```

**Threads.** Cut at both queues. Both queues feed the **same** element D, so:
- **Segment 1 = `{FakeLiveSrc, Tee}`** → **T1**.
- **Segment 2 = `{D}`** → **T2** (both queues collapse into one downstream thread).
- **= 2 threads** (it's *where the queues lead*, not how many, that counts).

**Flow.**
- **T1:** row `k` → Tee copies `{1:k}` to both queues (Tee claims 0 slots, so both
  branches are identical). Pushes both. Returns.
- **T2:** `D` pulls a buffer from `Queue_a` and one from `Queue_b`, **fold-matches
  by vector** (slot 1: `k == k`) → fires `D(k)`. Output `{1:k, 2:k}`.

`D` matches **by vector, never by arrival order**: if `Queue_a` runs ahead, `D`
holds its `{1:k}` and waits for `Queue_b`'s `{1:k}`. A drop on one branch would be a
vector gap (a missing `k`) — which is why a rejoining branch's Queue must be
**no-drop** (or `D` must detect the gap).

**Why the queues are mandatory here.** Without a Queue on at least one branch, a
push-threaded Tee would block pushing branch 1 while `D` waits for branch 2 →
deadlock. The queues decouple the branches and give `D` a buffered place to
aggregate.

✔ Validates: live diamond, fold-match across queue heads, deadlock-free rejoin,
thread counting by "where queues lead."

---

## 8. Property change — offline (cache replace) vs live (forward)

Same element `E` in two pipelines; change a property of `E`.

**8a. Offline (E one-run-driven).** Covered in Example 2: `rerun_from(E)` recomputes
the current cached batch in place, bumps `E`'s own slot (`4:1 → 4:2`), and the join
re-matches on the unchanged ancestor slot. The non-reran branch is held from cache.
**Replace-in-place; no advance; no stall.**

**8b. Live (E live-driven).** `E` is fed by a live source. The session **does not
re-emit**; it just updates the config. The change applies to the **next** pumped
buffer:

| tick | E's input clock | E's config | E output |
|---|---|---|---|
| 5 | `{1:5}` | old | `{1:5, 4:5}` |
| — | *property changed (forward, no rerun)* | | |
| 6 | `{1:6}` | **new** | `{1:6, 4:6}` |

Batch 5 used the old config, batch 6+ use the new one. The position (`slot 1`)
advances normally; there is **no spurious bump and no stall**. Past streamed batches
are not retroactively recomputed.

✔ Validates: Rule C — liveness-aware property change; and Rule B — why no global
version is needed (the conflation only existed in offline×streaming-Zip, which does
not occur).

---

## 9. Zip + property change — handled inside the Sync element

Two live channels paired by a Sync element (Example 3b), then a property of an
upstream element on channel A changes.

- **Live:** the change applies forward; channel A's *next* batch carries it; the
  Sync element keeps zipping by position (`A[n]` ↔ `B[n]`) as normal. The Sync
  element's own slot (the sync position) advances only on a real new pair, so the
  downstream barrier never desyncs.
- **One-run inputs to a Sync element** (rare): a reconfigure is a replace-in-place
  of the current pair; the Sync element re-emits the same sync-position with the
  updated side. Because **custom position-pairing lives only inside the Sync
  element**, it controls this directly and never has to survive a raw offline
  reconfigure on a bare pad. This is exactly the agreed constraint that lets the
  global clock stay a single counter.

✔ Validates: the agreed boundary that keeps Rule B (single counter) sound — custom
pairing is encapsulated in the Sync element.

---

## 10. Cooperative stop — end-to-end

Pipeline of Example 5 (`FakeLiveSrc → Queue → Analysis → FakeSink`), streaming, then
the user closes the window.

1. Window-close handler calls `session.request_stop()`.
2. **T1:** `FakeLiveSrc` is blocked in its hardware/file read using **poll-with-
   timeout**. On the next timeout it returns `NoData`, checks the stop token, sees
   the request, and returns **`EOS`**.
3. The pump stops feeding new buffers; the Queue **drains** its in-flight buffers
   per policy (drain = finish them; flush = drop them).
4. **T2:** `Analysis`/`FakeSink` process the drained buffers, then see `EOS`.
5. `stop_all` runs **in reverse order** (sink → analysis → source).
6. `run()` returns. Total latency ≤ one poll timeout, regardless of the data rate.

✔ Validates: graceful, prompt shutdown; `NoData`/poll-timeout *is* the
cancellation-check mechanism; drain/flush at the Queue boundary.

---

## 11. Coverage matrix — every discussed concept is exercised

| Concept | Where validated |
|---|---|
| Per-pad outputs / Tee fan-out (0 slots) | 2, 7 |
| Vector clock, own-slot increment | all |
| Default fold-match = per-slot barrier (Rule A) | 1, 2, 3a |
| Common-origin diamond barrier (free) | 2, 7 |
| Independent sources fire free | 3a |
| Partial rerun by vector (offline) | 2, 8a |
| Single counter, no global version (Rule B) | 8, 9 |
| Liveness-aware property change (Rule C) | 8a vs 8b |
| Sync element = inject common ancestor (Rule D) | 3b, 9 |
| Join modes (Zip / Held / Latest / Barrier / AllRequiredOnce) | 3b (Zip), 6 (Held), 2/7 (Barrier), 1 (AllRequiredOnce) |
| One `run()` pump loop; one-run vs live | 1, 4 |
| 3-state result (Data/NoData/EOS) | 4, 5, 10 |
| Segment threading (cut at Queues) | 5, 6, 7 |
| Queue = rate + thread + generation boundary; Queue-owned boundary FIFO | 5, 6, 7 |
| Backpressure / drop policy | 5, 6 |
| Live diamond aggregator, no-drop rule | 7 |
| Cooperative stop (token + poll-timeout) | 5, 10 |

---

## 12. What this proves

1. **The common case needs zero tuning.** Examples 1, 2, 4 are ordinary SCA
   pipelines; nobody configured synchronization. The fold-match does it.
2. **The hard cases reduce to the common case.** Independent pairing (3b), live
   diamonds (7), and the hybrid capture (6) all end up using the **default barrier**
   downstream — the complexity is absorbed by a Tee, a Queue, or a single Sync node.
3. **One execution model spans offline and live.** The same `run()` pump loop,
   vector clock, and barrier serve Examples 1 (offline) through 10 (live + stop);
   offline is the one-iteration special case.
4. **Reconfiguration is correct in both regimes** (8a/8b) and needs no global
   version, because liveness routes it (forward vs replace).
5. **Shutdown is graceful and prompt** (10), bounded by a poll timeout.

If the walkthroughs hold, the design holds. This document is the validation basis
and the implementation reference for the live phase
(`dataflow_sync_model.md` §12).
