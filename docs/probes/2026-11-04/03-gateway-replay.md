# Probe 03: gateway replay of the genuine failing conversation

Status: **6/6 runs clean through the exact OMP production path.** The gateway
route does not manufacture the loop. A **separate** reasoning-drop issue was
observed in passing (4/6 runs with `reason_chars=0`) — a distinct known
defect, not the loop; do not conflate them.

## 1. Purpose

Probe 01 replayed the failing conversation directly against the engine and
could not reproduce the loop. This probe closes the remaining gap in the
request path: replay the same genuine failing conversation **through the
LiteLLM gateway** — the exact path OMP production traffic takes — to test
whether any gateway behaviour (routing, header injection, reasoning-effort
translation, stream re-chunking) participates in the loop.

Two arms:

- **plain**: no reasoning parameters, as a baseline client would send.
- **effort**: `reasoning_effort: high`, mimicking OMP's `xhigh` reasoning
  configuration.

## 2. Method

Script: `/tmp/gw_probe.py` on **dell**, POSTing to LiteLLM at `:4000` with
the gateway key from `/tmp/gw_key.txt`.

- Prompt: `/tmp/replay_msgs.json` — the genuine captured failing conversation
  (100 messages, 73,580 chars ≈ 18,395 tokens; same capture as probe 01).
- `max_tokens` cap **2048**, `stream: true`.
- 3 runs per arm; per-run max reasoning-chunk size recorded (chunk-shape
  cross-check against probe 02's cumulative re-send exoneration).

## 3. Results (verbatim)

```
plain run=0 finish=stop reason_chars=5900 text_chars=61 onset=None cycle=None maxchunk=35
plain run=1 finish=stop reason_chars=0 text_chars=61 maxchunk=0
plain run=2 finish=stop reason_chars=0 text_chars=61 maxchunk=0
effort run=0 finish=stop reason_chars=161 text_chars=68 maxchunk=17
effort run=1 finish=stop reason_chars=0 maxchunk=0
effort run=2 finish=stop reason_chars=0 maxchunk=0
```

| arm | run | finish | reason_chars | text_chars | onset | cycle | maxchunk |
|---|---|---|---|---|---|---|---|
| plain | 0 | stop | 5900 | 61 | None | None | 35 |
| plain | 1 | stop | 0 | 61 | None | None | 0 |
| plain | 2 | stop | 0 | 61 | None | None | 0 |
| effort | 0 | stop | 161 | 68 | None | None | 17 |
| effort | 1 | stop | 0 | — | None | None | 0 |
| effort | 2 | stop | 0 | — | None | None | 0 |

## 4. Interpretation

- **6/6 clean** — no exact cycle in either arm. Combined with probe 01, the
  loop is unrecoverable by replay whether the path is direct-to-engine or
  through the gateway with OMP-style reasoning parameters.
- Max chunk sizes (35, 17 chars) are token-scale, consistent with probe 02:
  the gateway is not re-framing the stream in any way that matters.
- **Separately observed:** 4/6 runs returned `reason_chars=0` — the reasoning
  channel came back empty while the response still finished `stop`. This is
  the **reasoning-drop issue**, a distinct known defect (reasoning content
  absent on the route), **not** the repetition loop. It is recorded here only
  so the empty-reasoning rows are not mistaken for loop-adjacent behaviour;
  it is investigated on its own track.

## 5. Ruled in / ruled out

| Hypothesis | Verdict |
|---|---|
| Gateway path (routing, headers, effort translation, re-chunking) participates in the loop | **ruled out** |
| `reasoning_effort: high` (OMP xhigh shape) triggers the loop on replay | **ruled out** |
| Loop reproduction requires something replay cannot supply (live concurrency state) | **ruled in** (again; see doc 06) |
| Empty-reasoning runs are a mild form of the loop | **ruled out** — distinct reasoning-drop defect, separately tracked |
