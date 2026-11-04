# Probe 02: gateway SSE delta-shape check (cumulative re-send hypothesis)

Status: **gateway EXONERATED.** LiteLLM emits true token-sized
`reasoning_content` deltas — no cumulative re-send. Client-side repetition
accumulation cannot be manufactured by the gateway's SSE framing.

## 1. Purpose

Test the **cumulative re-send hypothesis**: if the LiteLLM gateway emitted
each SSE `delta.reasoning_content` chunk as *all reasoning so far* (rather
than only the new tokens), then a client that accumulates deltas naively
would manufacture exact repetitions locally — and the pi-ai exact-cycle guard
in `thinking-loop.ts` would fire on an artifact of double-counting, not on
real model output. Under that hypothesis the "loop" would be a
gateway/client-protocol bug, not an engine defect, and the fix would live in
the gateway or the client accumulator.

## 2. Method

Raw SSE capture from the **dell LiteLLM gateway** at `:4000`, model
`cluster/qwen3.8-27b-coder390` (the production cluster route):

- Prompt: multi-step math word problem (chosen to guarantee a non-trivial
  reasoning stream).
- `max_tokens: 1500`, `stream: true`.
- For every SSE chunk, record the **byte size of
  `delta.reasoning_content`**, then analyse the size sequence:
  - Cumulative re-send signature: chunk sizes **grow monotonically**
    (chunk *n* carries *n* tokens of payload).
  - True-delta signature: chunk sizes stay **token-sized** (a few to a few
    dozen characters) for the whole stream.

## 3. Results (verbatim)

```
finish=length
reason_chunks=463 total_reason_chars=3583
first25_sizes=[2,17,20,10,5,5,6,5,23,2,5,7,7,3,15,44,16,4,27,6,8,14,13,3,4]
last25_sizes=[7,5,1,8,3,15,8,5,1,8,13,8,4,1,10,11,5,1,6,1,11,10,1,8,1]
content_chunks=42 total_content_chars=597
GROWING(cumulative-resend)=False
max chunk 44 chars
```

| Metric | Value |
|---|---|
| finish_reason | length |
| reasoning chunks | 463 |
| total reasoning chars | 3583 |
| max single chunk | 44 chars |
| content chunks | 42 |
| total content chars | 597 |
| GROWING (cumulative-resend) | **False** |

## 4. Interpretation

463 reasoning chunks carried 3,583 characters total — **~7.7 chars/chunk
mean**, with the largest single chunk 44 chars and no upward drift between
the first 25 chunks (sizes 2–44) and the last 25 (sizes 1–15). That is the
signature of **true token-sized deltas**: each SSE event carries only the
newly decoded text. A cumulative re-send would show the last chunks carrying
thousands of characters each (the whole reasoning-so-far), and `GROWING`
would be true.

## 5. Ruled in / ruled out

| Hypothesis | Verdict |
|---|---|
| Gateway emits reasoning deltas cumulatively (re-sends all reasoning so far) | **ruled out** — chunk sizes flat, token-scale |
| Client guard trips on accumulation artifacts manufactured by the gateway | **ruled out** — nothing to double-count |
| Repetition originates in the server-side stream content itself | **ruled in** (narrows the defect to the engine or below) |
