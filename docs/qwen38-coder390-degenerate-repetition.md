# Qwen3.8-27B-Coder390-DFlash2: empty content from reasoning-budget starvation

## MTP mod sampling policy

The MTP recipe has a separate existing server-side policy in
`mods/qwen3.8-coder390-mtp/run.sh`. For both chat and text completions it
overrides request values and model defaults with temperature `1.0`, top-p
`0.95`, top-k `20`, and repetition penalty `1.10`. It also replaces the
request's thinking budget: with no effective generation cap it uses 32768;
at 340 tokens or less it uses 0; otherwise it uses
`min(32768, max(max_tokens * 75 // 100, max_tokens - 256))`. The input to
that calculation is the effective `max_tokens` argument to vLLM's
`to_sampling_params`, not the raw request field. Other sampling defaults
retain vLLM's normal request-over-model precedence.

This is the existing MTP policy, not a conclusion that the repetition penalty
fixes DFlash's observed empty-content symptom. The investigation below has a
different scope and explicitly reports the limitations of that evidence.
The MTP patch targets the explicit-local protocol layout observed in image
`sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02`;
it fails at startup if either protocol cannot be found or its sampling
forwarding layout differs, rather than silently serving without the policy.

To validate both actual vLLM request classes without touching a serving
container or real weights, from the repository root on the image host run:

```sh
docker run --rm --user 0:0 --entrypoint bash \
  -v "$PWD/mods/qwen3.8-coder390-mtp:/repo/mods/qwen3.8-coder390-mtp:ro" \
  -v "$PWD/tests/test_qwen38_coder390_mtp_sampling.py:/repo/tests/test_qwen38_coder390_mtp_sampling.py:ro" \
  sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02 \
  -c 'QWEN_MTP_REAL_VLLM=1 python3 /repo/tests/test_qwen38_coder390_mtp_sampling.py'
```

The opt-in test creates a minimal disposable checkpoint, applies the mod twice
inside the temporary container, and calls both real `ChatCompletionRequest`
and `CompletionRequest` `to_sampling_params` methods with conflicting client
and model defaults and several generation caps. It does not start an engine or
change the live service. The offline regression alone runs with
`python3 tests/test_qwen38_coder390_mtp_sampling.py`.

## DFlash investigation

Status: **partially root-caused; the reported loop itself is still
unreproduced.** Two distinct things were conflated during this investigation
and must be kept separate:

1. **Empty `content` at `finish_reason: length`** — fully explained and
   reproduced. It is reasoning-budget starvation: the thinking phase consumes
   the whole `max_tokens` budget, so no answer is emitted. Fix: a larger
   output budget (or thinking off). See §2.
2. **The verbatim `ductductduct` repetition** — **not reproduced** in any
   probe (single, 8-concurrent, prefix-cache-hit, streaming, long-context,
   with and without a repetition penalty). It remains an open observation.
   See §2.2.

The `repetition_penalty` floor added in commit `87c40e8fbd` was **reverted**
(commit `088c5a9b47`): it has a real quality cost and no demonstrated benefit.

Applies to image `vllm-node-b12x` as pinned by the recipe
(`sha256:e9f6e32941ec5680a58edfbfefaad6997f0527e22344b5b505cf8267de4625f2`),
`vllm 0.32.0rc1.dev301+g1ac942555.d20261016`, recipe
`qwen3.8-27b-coder390-4x-store-l2` (TP=4, durable Mooncake store L2).

## 1. Symptom

Live agent sessions served by `cluster/qwen3.8-27b-coder390` produced
**transient degenerate output**: a single token repeated verbatim in the
reasoning channel — reported as a repeating `ductductduct` run. Intermittent,
not steady-state; correlated with long agent contexts.

Three observable properties, all confirmed live:

1. The loop is in the **reasoning** channel, not `content`.
2. The thinking block **never closes**, so the whole `max_tokens` budget is
   consumed in reasoning and `content` returns empty.
3. `finish_reason` is `length`, not an error — so nothing looks broken in the
   response envelope, and callers see a truncated or empty answer.

## 2. Root cause of the empty-content outcome: reasoning-budget starvation

The model is a reasoning model. On a long prompt it needs roughly **400-500
tokens of thinking** before it emits an answer. When `max_tokens` is at or
below that requirement, the entire budget is spent inside the thinking block,
the block never closes, and `content` is empty.

The decisive experiment — same ~36k-token prompt, same serve, only
`max_tokens` varied:

| `max_tokens` | `finish` | `content` chars | `reasoning_tokens` |
|---|---|---|---|
| 600 | `length` | **0** | 600 (entire budget) |
| 1500 | `stop` | **2928** | 410 |
| 3000 | `stop` | **4970** | 401 |
| 6000 | `stop` | **4408** | 514 |

At 600 the model never reaches its answer. At 1500+ it finishes cleanly with
~400-500 reasoning tokens and thousands of characters of coherent content.

**The threshold is not fixed.** The reasoning requirement varies run to run on
the *same* prompt: 600, 410, 401, 514 tokens in the sweep above, and later
378 and 256 on re-runs of the same prompt — where `max_tokens=600` then
**succeeded** (`content=713`, `finish=stop`). So 600 is not a hard cliff; it
sits close enough to the typical requirement that it fails intermittently.
That variance is the mechanism behind the reported transience.

A budget of 1500+ is a safe margin, not a measured minimum.

This also explains the **transience**: whether a request degenerates depends
on how much thinking that particular prompt needs versus the budget the caller
supplied. Short prompts need little thinking and always succeed; long agent
contexts need more and fall off the cliff.

### 2.1 Why budget starvation can *look* like repetition

Once the budget is exhausted mid-thought, the tail of the reasoning stream is
whatever the model was doing when it ran out. In a long repetitive context (an
agent session log full of near-identical tool entries) that tail can be a
repeated token. This is a **plausible** mechanism for the reported
`ductductduct` run, but it is **not confirmed** — see §2.2.

### 2.2 The `ductductduct` loop is still unreproduced

Every probe in this investigation failed to reproduce a verbatim token loop:

| Probe | Result |
|---|---|
| single ~60-token prompt | clean |
| 8 concurrent distinct-prefix requests | clean |
| 4 identical prompts (prefix-cache hit path) | clean |
| streaming path | clean |
| ~36k-token prompt, `max_tokens` 600-6000 | clean (empty content at 600, coherent at 1500+) |
| same, with `repetition_penalty` 1.0 / 1.10 / 1.15 | clean |

The character-level periodic-run detector (see §5) found no repeated
substring in any response. So the loop is either (a) a rarer condition than
the probes reached, (b) specific to a prompt shape not exercised here, or
(c) an artifact of how the reporting session rendered truncated reasoning.

**Do not treat budget starvation as a confirmed explanation of the loop.**
It explains the empty-content symptom, which is real and reproducible; the
loop needs its own reproduction before it can be called root-caused.

## 3. What was ruled out, with evidence

The investigation initially pursued Mooncake store corruption, then a
sampling-penalty fix. Both are **disproven**:

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Store read failures | `load_get_failed_keys=0`, `load_get_error_count=0` | no failures |
| External cache hits correlate with failure | External prefix cache hit rate flat **23.1%** across the failure window | no correlation |
| Stale pre-boot KV being served | 500 pre-boot buckets, `atime == mtime == 06:03` — never read since written | disproven |
| Cross-node bucket collision | per-node dirs distinct: `gx10-cb11`, `gx10-f1d8`, `gx10-becc`, `gx10-9273` | isolated correctly |
| Spec-decode collapse as the trigger | MAL 2.6-3.3, acceptance 22-32% on the *successful* long runs | healthy; no collapse |
| `mamba-spec-relocate-guard` free-vs-move | guard patch applied (6 MARK lines); no correlation with the failure | not implicated |
| Repetition penalty is the fix | warm-prefix sweep: floor 1.10 gave 225 then 0 chars on identical config | **not reliable** |

### 3.1 The repetition-penalty false positive

An early A/B appeared to show `repetition_penalty: 1.10` fixing the problem
(0 → 549 chars). That was **confounded by prefix-cache warmth**: the first run
was cold (341 s) and the second warm (25 s). Re-running with the prefix warmed
first, varying only the penalty:

| Run | Penalty | `content` chars |
|---|---|---|
| A | floor 1.10 | 225 |
| B | explicit 1.0 | 0 |
| C | explicit 1.10 | 0 |
| D | floor 1.10 (same as A) | 0 |

Identical configuration produced 225 then 0. The penalty is not the
discriminator; the budget is. **Lesson: warm the prefix before A/B-ing
sampling parameters, or the cache state dominates the result.**

## 4. The fix

Raise the output budget for agent traffic. The reasoning requirement is
~400-500 tokens, so any budget comfortably above that (1500+) lets the model
finish. Two options:

1. **Caller-side (preferred):** set `max_tokens >= 1500` on agent requests.
   This is a client contract, not a server change, and it preserves the
   reasoning channel.
2. **Server-side:** disable thinking for workloads that do not consume
   reasoning:

   ```yaml
   --default-chat-template-kwargs '{"enable_thinking": false}'
   ```

   Verified: `enable_thinking: false` returns content with
   `reasoning_tokens: 0`, eliminating the failure mode entirely.

### 4.1 The `repetition_penalty` floor was tried and reverted

Commit `87c40e8fbd` added `--override-generation-config
'{"repetition_penalty":1.10}'` to the recipe. It loaded correctly (startup
log: `Default vLLM sampling parameters have been overridden ...
{'repetition_penalty': 1.1, ...}`) but **did not fix the empty-content
outcome** — a warm-prefix sweep gave 225 then 0 chars on identical config.

It was **reverted** in commit `088c5a9b47`. A global 1.10 penalty is a real
quality cost (it penalises legitimate token reuse in technical prose) with no
demonstrated benefit here. The budget is the actual lever.

## 5. Reproduction

Requires the live serve. Run from any node that can reach the engine API.

```bash
# On gx10-node1 (or any host that can reach localhost:8000 there).
cat > /tmp/budget_repro.py <<'PYEOF'
import json, time, urllib.request

MODEL = "local-inference-lab/Qwen3.8-27B-Coder390-DFlash2"
URL = "http://localhost:8000/v1/chat/completions"

# ~36k-token synthetic agent-session log. Short prompts do NOT reproduce:
# they need little thinking, so they never exhaust the budget.
filler = "\n".join(
    f"## Entry {i:05d}\nTimestamp: 2026-11-03T09:{i % 60:02d}:00Z\n"
    f"tool result: read /nas-1/models/cfg/{i}.json -> 4096 bytes, keys=[a,b,c,d,e]\n"
    f"assistant: checking the config for bucket {i}, allocation ratio 0.9, eviction 0.10"
    for i in range(400)
)
PROMPT = (
    "Here is a long session log. Summarize the recurring issue and "
    "propose fixes.\n\n" + filler
)


def ask(max_tokens, repetition_penalty=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    body = json.loads(urllib.request.urlopen(req, timeout=2400).read().decode())
    choice = body["choices"][0]
    msg = choice["message"]
    detail = (body.get("usage") or {}).get("completion_tokens_details") or {}
    return {
        "max_tokens": max_tokens,
        "finish": choice.get("finish_reason"),
        "content_chars": len(msg.get("content") or ""),
        "reasoning_tokens": detail.get("reasoning_tokens", 0),
        "secs": round(time.time() - started, 1),
    }


if __name__ == "__main__":
    # Warm the prefix first: a cold run is 3-5x slower and confounds timing.
    ask(600)
    for budget in (600, 1500, 3000):
        print(json.dumps(ask(budget)))
PYEOF
python3 /tmp/budget_repro.py
```

Expected: `max_tokens: 600` → `content_chars: 0`, `finish: length`,
`reasoning_tokens: 600`. `max_tokens: 1500` and above → `finish: stop`,
`content_chars` in the thousands, `reasoning_tokens` ~400-500.

### 5.1 Observing the reasoning channel directly

Non-streaming responses can suppress `reasoning_content` while still billing
`reasoning_tokens`, so the truncated thought is easiest to see on the stream.
The reasoning text lands in `delta.reasoning`. Use the **long** prompt — a
short one will not exhaust the budget and will not show the truncation:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(python3 - <<'PY'
import json
filler = "\n".join(
    f"## Entry {i:05d}\ntool result: read /nas-1/models/cfg/{i}.json -> 4096 bytes"
    for i in range(400)
)
print(json.dumps({
    "model": "local-inference-lab/Qwen3.8-27B-Coder390-DFlash2",
    "messages": [{"role": "user", "content": "Summarize this log.\n\n" + filler}],
    "max_tokens": 600, "temperature": 0, "stream": True,
}))
PY
)" | grep -o '"reasoning":"[^"]*"' | tail -20
```

## 6. Code change summary

**No recipe change is retained.** The `repetition_penalty` floor
(`87c40e8fbd`) was reverted in `088c5a9b47`; the recipe is back to its
pre-investigation state.

Validation performed on the reverted tree:

- `yaml.safe_load` parses the recipe.
- No `repetition_penalty` / `override-generation-config` remains in the file.
- `tests/test_recipes.sh`: **68 passed, 0 failed**.

The fix for the empty-content bug is a **caller-side budget** (§4), which is
not a recipe change.

## 7. Deployment note

**A recipe change does not take effect on `docker restart`.** The vLLM argv is
baked at container create time, so a restart reuses the old command line. A
full relaunch is required:

```bash
spark-vllm-docker-serve launch --recreate
```

Confirm the flag is live:

```bash
docker exec vllm_node ps aux | grep -o -- '--override-generation-config [^ ]*'
```

## 8. Open items

1. **Caller budgets are the real fix and are not yet enforced.** Nothing
   guarantees agent clients request `max_tokens >= 1500`. A gateway-side floor
   (or a warning when `max_tokens` is below the observed reasoning
   requirement) would prevent recurrence.
2. **No tripwire for the failure shape.** Nothing detects
   `reasoning_tokens == max_tokens` with empty `content`. A cheap check in the
   gateway or eval harness would surface this before operators notice degraded
   agent output.
3. **Reasoning requirement is prompt-dependent.** ~400-500 tokens was measured
   on one 36k-token prompt. Longer or more complex contexts may need more; the
   safe floor should be measured across the real agent workload, not assumed.
4. **The `ductductduct` loop is unreproduced.** Budget starvation explains the
   empty-content symptom but has not been shown to produce a verbatim token
   loop. Reproducing it needs a prompt shape not yet exercised — capture the
   exact failing request (prompt, `max_tokens`, sampling params, route) from a
   live occurrence before theorising further.
