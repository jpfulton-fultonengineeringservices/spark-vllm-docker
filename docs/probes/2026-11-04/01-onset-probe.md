# Probe 01: loop-onset measurement (`onset_probe.py`)

Status: **6/6 runs clean.** The loop is **not reproducible by cold replay of
the genuine failing conversation**, so loop onset is unmeasurable by this
method. The budget-vs-guard decision this probe was built to settle could not
be decided here; it was later resolved by the root cause in
[06-eagle-block-drop-root-cause-and-fix.md](06-eagle-block-drop-root-cause-and-fix.md).

## 1. Purpose

Measure loop **onset**: the number of tokens of legitimate reasoning emitted
before the first exact repetition cycle begins. Onset was the discriminator
between two candidate remediations:

- **Tune the thinking budget.** If onset is late (thousands of tokens of
  legitimate reasoning first), the loop is a deep-degeneration mode and a
  budget/thinking-policy change is the right lever.
- **Relax or disable the client-side guard.** If onset is at or near token 0
  (the sequence is born looping), the pi-ai exact-cycle guard is firing on a
  server-side state corruption, not on model behaviour — and tuning the guard
  or the budget is the wrong layer entirely.

## 2. Method

Script: `/tmp/onset_probe.py`, run inside the `vllm_node` container against
the engine directly at `http://localhost:8000/v1/chat/completions`,
streaming, no gateway in the path.

Prompt source, in preference order:

1. **The real captured failing conversation** —
   `/tmp/replay_msgs.json`: 100 messages, 73,580 characters (≈18,395 tokens).
   This is the genuine agent session that produced the loop in production,
   captured verbatim and replayed unmodified.
2. A 32-block near-duplicate synthetic fallback (used only if the capture was
   unavailable; not needed).

Arms: `max_tokens` caps of **512** and **2048**, **3 runs each**, streaming,
with a character-level exact-cycle detector over `delta.reasoning_content`.

### 2.1 Sampling regime note

The request body sent `temperature: 0`. The deployed mod
`mods/qwen3.8-coder390-dflash/run.sh` **forces** `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `repetition_penalty=1.10` server-side inside
`to_sampling_params`, overriding the request. All six runs therefore executed
in the **production sampling regime** — the same envelope the live agent
traffic runs under. This matters: the clean results below cannot be
attributed to greedy decoding.

## 3. Results (verbatim)

```
source=replay nmsg=100 total_chars=73580 approx_tokens=18395
cap=512 run=0 finish=stop reason_chars=1412 text_chars=0 onset=None cycle=None
cap=512 run=1 finish=stop reason_chars=821 text_chars=197 onset=None cycle=None
cap=512 run=2 finish=stop reason_chars=757 text_chars=129 onset=None cycle=None
cap=2048 run=0 finish=stop reason_chars=1764 text_chars=169 onset=None cycle=None
cap=2048 run=1 finish=stop reason_chars=6726 text_chars=0 onset=None cycle=None
cap=2048 run=2 finish=stop reason_chars=0 text_chars=2 onset=None cycle=None
```

| cap | run | finish | reason_chars | text_chars | onset | cycle |
|---|---|---|---|---|---|---|
| 512 | 0 | stop | 1412 | 0 | None | None |
| 512 | 1 | stop | 821 | 197 | None | None |
| 512 | 2 | stop | 757 | 129 | None | None |
| 2048 | 0 | stop | 1764 | 169 | None | None |
| 2048 | 1 | stop | 6726 | 0 | None | None |
| 2048 | 2 | stop | 0 | 2 | None | None |

## 4. Interpretation

- **6/6 clean.** No exact cycle detected at any cap, on the genuine failing
  conversation, in the production sampling regime.
- The loop is **not reproducible by cold replay**: re-presenting the same
  prompt to a quiescent engine does not re-create the failure, even though
  the identical prefix looped deterministically in production (see the retry
  storm evidence in doc 06, §4).
- Onset is therefore **unmeasurable this way** — there is no loop to measure.
  The decisive implication (confirmed in doc 06) is that the trigger is not
  in the prompt content but in **live engine state at admission time**: a
  prefix-cache hit boundary landing on a still-decoding sibling's volatile
  trailing block. A cold engine has no such sibling, so no amount of replay
  fidelity can reproduce the loop.

## 5. Ruled in / ruled out

| Hypothesis | Verdict |
|---|---|
| Loop onset is measurable by replaying the failing prompt | **ruled out** |
| Loop is a property of the prompt content itself | **ruled out** (same prompt, clean 6/6) |
| Greedy vs production sampling explains non-reproduction | **ruled out** (server forced the production envelope) |
| Loop requires live concurrency-dependent engine state | **ruled in** (motivates probes 04 and the doc-06 mechanism) |
