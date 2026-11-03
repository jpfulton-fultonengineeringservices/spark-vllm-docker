# Qwen3.8-27B-Coder390-DFlash2: reasoning-channel degeneration under greedy decoding

Status: **mitigated** in `recipes/qwen3.8-27b-coder390-4x-store-l2.yaml`
(commit `87c40e8fbd`). Root cause is a sampler/degeneration defect in the
target model, **not** KV-cache corruption. Pending live confirmation on the
redeployed serve.

Applies to image `vllm-node-b12x` as pinned by the recipe
(`sha256:e9f6e32941ec5680a58edfbfefaad6997f0527e22344b5b505cf8267de4625f2`),
`vllm 0.32.0rc1.dev301+g1ac942555.d20261016`, recipe
`qwen3.8-27b-coder390-4x-store-l2` (TP=4, durable Mooncake store L2).

## 1. Symptom

Live agent sessions served by `cluster/qwen3.8-27b-coder390` produced
**transient degenerate output**: a single token repeated verbatim in the
reasoning channel — reported as a repeating `ductductduct` run. Intermittent,
not steady-state; correlated with long agent contexts rather than with any
particular prompt.

Three observable properties, all confirmed live:

1. The loop is in the **reasoning** channel, not `content`.
2. The thinking block **never closes**, so the whole `max_tokens` budget is
   consumed in reasoning and `content` returns empty.
3. `finish_reason` is `length`, not an error — so nothing looks broken in the
   response envelope, and callers see a truncated or empty answer.

## 2. Root cause

The checkpoint ships a `generation_config.json` with **no repetition penalty**:

```json
{
  "bos_token_id": 248044,
  "do_sample": true,
  "eos_token_id": [248046, 248044],
  "pad_token_id": 248044,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95,
  "transformers_version": "5.9.0"
}
```

`do_sample: true` with `temperature: 1.0` is stochastic and self-correcting.
But agent and tool-calling clients send **`temperature: 0`** (greedy). Under
greedy decoding with no repetition penalty, once the model enters a repeated
token trajectory it cannot leave it: every step re-selects the argmax of a
distribution the repetition has already flattened. The control is
`repetition_penalty`, and it was absent.

This is a classic degenerate-repetition mode, not a hardware or transport
fault. It requires *long* context to manifest: every request under ~60 tokens
was clean across dozens of trials.

## 3. Empirical validation

All measurements against the live 4x store-l2 serve on `gx10-node1`
(`http://localhost:8000/v1/chat/completions`), `temperature: 0`,
`max_tokens: 600-900`, single request, deterministic prompt.

### 3.1 The discriminator (decisive)

Same ~36k-token prompt (a synthetic agent-session log), four variants:

| Variant | `finish` | `content` chars | `reasoning_tokens` |
|---|---|---|---|
| baseline (greedy, no penalty) | `length` | **0** | 900 (entire budget) |
| `repetition_penalty: 1.10` | `length` | **549** | — |
| `repetition_penalty: 1.15` | `length` | **732** | 418 |
| `enable_thinking: false` | `length` | **2256** | 0 |

Two independent levers each break the loop. That is the signature of a
sampler-level degeneration, not data corruption: a corrupted KV cache would
not be repaired by a rep-penalty.

A tighter sweep on the same prompt:

| Setting | `content` chars |
|---|---|
| `repetition_penalty: 1.05`, `temp: 0` | 76 |
| `repetition_penalty: 1.10`, `temp: 0` | 549 |
| no penalty, `temp: 1.0` (model default) | 566 |

1.05 is insufficient; 1.10 restores content. The model's own shipped
`temperature: 1.0` sampling also avoids the loop — greedy is the trigger.

### 3.2 Negative controls (short prompts)

Ruled out as *insufficient to reproduce*, which is why smoke tests miss this:

| Probe | Result |
|---|---|
| single ~60-token prompt | clean, no repeats |
| 8 concurrent distinct-prefix requests | all clean, `finish=stop` or `length`, no repeats |
| 4 identical prompts (prefix-cache hit path) | clean |
| streaming path | clean |

### 3.3 What was ruled out, with evidence

The investigation initially pursued Mooncake store corruption. That hypothesis
is **disproven**:

| Claim | Measurement | Verdict |
|---|---|---|
| Store read failures | `load_get_failed_keys=0`, `load_get_error_count=0` | no failures |
| External cache hits correlate with failure | External prefix cache hit rate flat **23.1%** across the entire failure window | no correlation |
| Stale pre-boot KV being served | 500 pre-boot buckets, `atime == mtime == 06:03` — never read since written | disproven |
| Cross-node bucket collision | per-node dirs distinct: `gx10-cb11`, `gx10-f1d8`, `gx10-becc`, `gx10-9273` | isolated correctly |
| Spec-decode collapse as the trigger | MAL 1.00 / 0.000 per-position windows occur **after** logits degrade | consequence, not cause |

The spec-decode acceptance collapse to `0.000` is downstream: once the target
model's distribution is degenerate, every draft token is rejected. Treating it
as the cause sends you into the connector and the
`mamba-spec-relocate-guard` mod, both innocent here.

## 4. The fix

Engine-side sampling floor, set once in the recipe rather than per-caller:

```yaml
defaults:
  # ...
  repetition_penalty: "1.10"
```

```yaml
command: |
  b12x-kcache exec vllm serve local-inference-lab/Qwen3.8-27B-Coder390-DFlash2 \
    # ...
    --override-generation-config '{"repetition_penalty":1.10}' \
```

`repetition_penalty` is declared as the **string** `"1.10"` because
`run-recipe.py` renders the command with `str.format`; a YAML float `1.1`
would render as bare `1.1`, losing the readable precision. Both produce a
valid float in the final JSON, but `"1.10"` documents intent. The recipe's
inner JSON braces are doubled (`{{` / `}}`) so `str.format` leaves them
literal — the same convention every other JSON argument in this command uses.

### 4.1 Why this reaches requests the caller did not annotate

`--override-generation-config` is **not** per-request. It merges into the
model's default sampling parameters at startup:

```
vllm/config/model.py
  ModelConfig.get_diff_sampling_param()
      config.update(self.override_generation_config)      # :1867
```

and every chat completion applies that dict:

```
vllm/entrypoints/openai/chat_completion/serving.py
  self.default_sampling_params = self.model_config.get_diff_sampling_param()   # :178
  sampling_params = request.to_sampling_params(max_tokens,
                                               self.default_sampling_params)   # :365
```

`repetition_penalty` is in that function's `available_params` allow-list. A
caller that omits the penalty inherits `1.10`; a caller that sets one still
wins (request values take precedence). This is a *floor*, not an override.

### 4.2 Why 1.10 and not 1.15

1.15 clears the loop more aggressively (732 vs 549 chars) but measurably
shortens natural prose — it penalises legitimate token reuse in technical
writing. 1.10 is the smallest value that clears the loop with content intact
on the reproducer. Repetition penalty is an *interpretation* below 1.0, so
1.10 is a mild, bounded correction.

## 5. Reproduction

Requires the live serve. Run from any node that can reach the engine API.

```bash
# On gx10-node1 (or any host that can reach localhost:8000 there).
cat > /tmp/degeneration_repro.py <<'PYEOF'
import json, time, urllib.request

MODEL = "local-inference-lab/Qwen3.8-27B-Coder390-DFlash2"
URL = "http://localhost:8000/v1/chat/completions"

# ~36k-token synthetic agent-session log. Short prompts do NOT reproduce.
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


def periodic(text, min_period=2, max_period=30, min_reps=4):
    """Find substrings repeated back-to-back, character level.

    Character-level on purpose: 'ductductduct' has no word separator, so a
    word-boundary scrape sees ONE token, not a repetition. A word-level
    detector silently reports "no repeats" on exactly this symptom.
    """
    hits = []
    n = len(text)
    for period in range(min_period, max_period + 1):
        i = 0
        while i + period <= n:
            unit = text[i:i + period]
            reps, j = 1, i + period
            while j + period <= n and text[j:j + period] == unit:
                reps += 1
                j += period
            if reps >= min_reps:
                hits.append((unit, reps))
                i = j
            else:
                i += 1
    hits.sort(key=lambda h: -h[1])
    return hits[:5]


def ask(repetition_penalty=None, temperature=0.0, max_tokens=600):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    body = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
    choice = body["choices"][0]
    msg = choice["message"]
    detail = (body.get("usage") or {}).get("completion_tokens_details") or {}
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return {
        "finish": choice.get("finish_reason"),
        "content_chars": len(msg.get("content") or ""),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "reasoning_tokens": detail.get("reasoning_tokens", 0),
        "periodic_runs": periodic(text),
        "secs": round(time.time() - started, 1),
    }


if __name__ == "__main__":
    print("BEFORE (greedy, no penalty):")
    print(json.dumps(ask(None), indent=2))
    print("\nAFTER (repetition_penalty=1.10):")
    print(json.dumps(ask(1.10), indent=2))
PYEOF
python3 /tmp/degeneration_repro.py
```

Expected with the defect present: `content_chars: 0`, `reasoning_tokens`
equal to `max_tokens`, and `periodic_runs` non-empty. Expected with the fix
live: `content_chars` in the hundreds and `periodic_runs` empty.

### 5.1 Streaming variant (shows the reasoning channel directly)

Non-streaming responses can suppress `reasoning_content` while still billing
`reasoning_tokens`, which is why the loop is easiest to see on the stream. The
reasoning text lands in `delta.reasoning`:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-inference-lab/Qwen3.8-27B-Coder390-DFlash2",
       "messages":[{"role":"user","content":"Explain the CAP theorem."}],
       "max_tokens":400,"temperature":0,"stream":true}' \
  | grep -o '"reasoning":"[^"]*"' | head -40
```

## 6. Code change summary

Single file: `recipes/qwen3.8-27b-coder390-4x-store-l2.yaml`
(commit `87c40e8fbd`, 27 insertions).

```diff
@@ defaults:
   max_cudagraph_capture_size: 256
   num_speculative_tokens: 7
   draft_tensor_parallel_size: 4
+  # ... 21-line rationale comment: the shipped generation_config has no
+  # repetition penalty; greedy decoding degenerates into an unbounded
+  # reasoning-channel loop; measurements for 1.10 / 1.15 / thinking-off.
+  repetition_penalty: "1.10"

@@ command: |
     --speculative-config '{"method":"dflash",...}' \
+    --override-generation-config '{"repetition_penalty":1.10}' \
     --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector",...}' \
```

Validation performed:

- `yaml.safe_load` parses the recipe; `repetition_penalty` resolves to `1.10`.
- The rendered flag is valid JSON: `{"repetition_penalty":1.10}` →
  `json.loads` → `{'repetition_penalty': 1.1}` (float, as vLLM expects).
- No unresolved `str.format` placeholders; a missing key would raise
  `KeyError` and abort the launch.
- `tests/test_recipes.sh`: **68 passed, 0 failed**.

## 7. Deployment note

**A recipe change does not take effect on `docker restart`.** The vLLM argv is
baked at container create time, so a restart reuses the old command line. A
full relaunch is required:

```bash
spark-vllm-docker-serve launch --recreate
```

Confirm the flag is live before re-testing:

```bash
docker inspect vllm_node --format '{{join .Config.Cmd " "}}' | tr ' ' '\n' \
  | grep -A1 override-generation-config
```

## 8. Open items

1. **Live confirmation pending.** The mitigation is validated at the sampler
   level (A/B on the live engine) but has not yet been observed end to end on
   a redeployed serve.
2. **No repetition guard in the harness.** Nothing detects an all-reasoning
   response (`reasoning_tokens == max_tokens` with empty `content`). A cheap
   tripwire in the gateway or eval harness would catch future regressions
   before operators notice degraded agent output.
3. **Callers that pin `repetition_penalty: 1.0`** explicitly will override the
   floor and can still degenerate. Nothing enforces the minimum.
4. **Thinking-off is the stronger mitigation** (2256 chars vs 549) at the cost
   of the reasoning channel. If agent workloads do not consume reasoning,
   `--default-chat-template-kwargs '{"enable_thinking":false}'` is worth
   evaluating as a follow-up.
