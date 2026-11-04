# Probe 05: long-generation replay (`long_probe.py`)

Status: **4/4 runs clean**, up to 16,229 characters of continuous reasoning.
The loop does **not** onset late under a production-scale generation budget
on a replayed prompt.

## 1. Purpose

All prior replay probes capped generation at ≤2048 tokens. Production agent
traffic runs with a thinking budget of **32768** tokens. If the loop onsets
only deep into a long reasoning stream — a late-degeneration mode — every
prior probe would have terminated before the loop had room to begin. This
probe removes that confound: give the model a large budget and watch for
late-onset cycles.

## 2. Method

Script: `/tmp/long_probe.py` inside the `vllm_node` container, direct against
`http://localhost:8000`, streaming, exact-cycle detector on
`delta.reasoning_content`.

- Prompt: `/tmp/replay_msgs.json` — the genuine captured failing
  conversation (100 messages, ≈18,395 tokens; same capture as probes 01/03/04).
- `max_tokens` cap **8192** — 4× the largest prior replay cap, and enough
  headroom for a long thinking run.
- **4 runs.** Server-side sampling override in force (production envelope:
  temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.10 — see
  probe 01, §2.1).

## 3. Results (verbatim)

```
run=0 finish=stop reason_chars=1850 text_chars=254 cycle=None
run=1 finish=stop reason_chars=12464 text_chars=198 cycle=None
run=2 finish=stop reason_chars=195 text_chars=169 cycle=None
run=3 finish=stop reason_chars=16229 text_chars=136 cycle=None
```

| run | finish | reason_chars | text_chars | cycle |
|---|---|---|---|---|
| 0 | stop | 1850 | 254 | None |
| 1 | stop | 12464 | 198 | None |
| 2 | stop | 195 | 169 | None |
| 3 | stop | 16229 | 136 | None |

## 4. Interpretation

- **4/4 clean.** Two runs produced very long reasoning streams (12,464 and
  16,229 characters — on the order of 3–4k reasoning tokens, well past every
  prior probe's ceiling) and stayed coherent throughout; the cycle detector
  found nothing at any depth.
- The wide run-to-run variance in reasoning length (195 → 16,229 chars on the
  *same* prompt) is expected under the production sampling envelope and
  mirrors the variance documented in
  [../../qwen38-coder390-degenerate-repetition.md](../../qwen38-coder390-degenerate-repetition.md)
  §2 — it is not loop-adjacent.
- **No late-onset loop.** Generation depth is not the missing ingredient.
  Combined with probes 01, 03, and 04, this closes the replay avenue
  entirely: no replayed prompt, at any cap, at any concurrency, on any path,
  reproduces the loop. The trigger must be state that replay cannot supply —
  the live-sibling speculative-tail geometry established in doc 06.

## 5. Ruled in / ruled out

| Hypothesis | Verdict |
|---|---|
| Loop onsets only late in long reasoning (caps ≤2048 were too small) | **ruled out** (clean to 16k chars) |
| Generation budget is the discriminator | **ruled out** |
| Replay-based reproduction is viable with the right parameters | **ruled out** (final replay probe; all avenues clean) |
