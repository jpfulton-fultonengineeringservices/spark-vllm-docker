# Probe 04: concurrent replay (`conc_probe.py`)

Status: **8/8 concurrent runs clean.** Concurrency alone does not reproduce
the loop — because this probe's geometry (8 *identical* prompts) never
exercises the actual trigger: a **partial prefix hit against a sibling's
LIVE speculative tail**. See §4 and
[06-eagle-block-drop-root-cause-and-fix.md](06-eagle-block-drop-root-cause-and-fix.md).

## 1. Purpose

Production loops correlate with **simultaneous OMP sessions**: the user's
characteristic report is two or more concurrent agent sessions, one of which
degenerates while a sibling stays coherent. Probes 01 and 03 replayed
single-shot and could not reproduce. This probe adds concurrency: fire
multiple simultaneous requests at the engine and watch for the exact cycle
under load.

## 2. Method

Script: `/tmp/conc_probe.py` inside the `vllm_node` container, direct against
`http://localhost:8000` (no gateway).

- **8 parallel workers**, all launched together.
- Every worker sends the **SAME replay prompt** (`/tmp/replay_msgs.json`,
  the genuine captured failing conversation — 100 messages, ≈18,395 tokens).
- `max_tokens` cap **1024**, streaming, exact-cycle detector on
  `delta.reasoning_content`.

## 3. Results (verbatim)

```
run=0 finish=stop reason_chars=2857 cycle=None
run=1 finish=stop reason_chars=2925 cycle=None
run=2 finish=length reason_chars=2633 cycle=None
run=3 finish=stop reason_chars=1285 cycle=None
run=4 finish=length reason_chars=3030 cycle=None
run=5 finish=stop reason_chars=1069 cycle=None
run=6 finish=stop reason_chars=408 cycle=None
run=7 finish=length reason_chars=3125 cycle=None
```

| run | finish | reason_chars | cycle |
|---|---|---|---|
| 0 | stop | 2857 | None |
| 1 | stop | 2925 | None |
| 2 | length | 2633 | None |
| 3 | stop | 1285 | None |
| 4 | length | 3030 | None |
| 5 | stop | 1069 | None |
| 6 | stop | 408 | None |
| 7 | length | 3125 | None |

## 4. Interpretation

**8/8 clean.** Three runs hit the token cap (`length`) with coherent
reasoning; none produced a cycle. Concurrency by itself is not sufficient.

### 4.1 Why identical prompts cannot hit the trigger geometry

From the root-cause investigation (doc 06): the loop trigger is a prefix-cache
hit boundary that lands on a **sibling sequence's still-live, speculative
trailing block** — the block the EAGLE-family block-drop then forces to be
recomputed, corrupting the mamba-align SSM resume boundary for the newly
admitted sequence.

Eight **identical** prompts never create that situation:

- Either they are admitted together and **compute prefill simultaneously** —
  nothing is committed to the prefix cache yet, so there is no hit at all; or
- Later arrivals hit **fully-committed, quiescent blocks** written by an
  already-finished sibling — a stable hit, no live speculative tail, no
  boundary recompute against volatile state.

Neither geometry exercises a **partial** hit against a **live** sibling's
speculative tail. Production traffic — two *different* sessions sharing a
long common system/tool prefix, one decoding while the other is admitted
after a long waiting-queue stay — does. That is why the loop correlates with
concurrency in production yet this concurrent probe is clean.

## 5. Ruled in / ruled out

| Hypothesis | Verdict |
|---|---|
| Concurrency alone reproduces the loop | **ruled out** (8/8 clean) |
| Identical-prompt fan-out exercises the production failure geometry | **ruled out** (§4.1) |
| Trigger requires partial prefix hit against a sibling's live speculative tail | **ruled in** (confirmed as the mechanism in doc 06) |
