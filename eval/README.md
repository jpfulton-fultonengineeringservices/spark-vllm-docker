# eval/ — lm-evaluation-harness rig

Drives the GX10 GPU cluster evaluations from this workstation over the network
(no GPU needed locally). The harness itself lives in the sibling submodule
`gpu-cluster-forks/lm-evaluation-harness` (shallow clone, `main`).

## Setup

```bash
./eval/eval.sh setup      # or just run any eval; the venv bootstraps on first use
```

Creates `eval/.venv` with `uv` and installs the harness editable with the
`api`, `tasks` and `zeno` extras plus `transformers` and `pillow`. The `vllm`
extra is opt-in (`EVAL_WITH_VLLM=1 ./eval/eval.sh setup`) and only works on a
Linux/NVIDIA host — vLLM ships no macOS/arm64 wheels; workstation runs use the
HTTP backends below.

## Usage

```bash
./eval/eval.sh run eval/configs/smoke.toml                # arc_easy, limit 10
./eval/eval.sh run eval/configs/smoke.toml --dry-run      # print the lm-eval argv, exit
./eval/eval.sh run eval/configs/mimo-nvfp4-cluster-text.toml
./eval/eval.sh run eval/configs/mimo-nvfp4-cluster-mmmu.toml
LITELLM_API_KEY=... ./eval/eval.sh run eval/configs/mimo-nvfp4-litellm.toml

./eval/eval.sh run eval/configs/qwen38-coder390-cluster-code.toml
./eval/eval.sh run eval/configs/qwen38-coder390-cluster-text.toml
./eval/eval.sh run eval/configs/qwen38-coder390-cluster-mmmu.toml
LITELLM_API_KEY=... ./eval/eval.sh run eval/configs/qwen38-coder390-litellm.toml

./eval/eval.sh viz results --local --html                 # markdown + report.html
ZENO_API_KEY=... ./eval/eval.sh viz results --zeno        # upload to Zeno hub
```

## Endpoints

| target  | base_url                                              | model id                          |
|---------|-------------------------------------------------------|-----------------------------------|
| direct  | `http://10.10.10.165:8000/v1/...` (rank-0 vllm-router, MiMo) | `XiaomiMiMo/MiMo-V2.6-Flash-RL-NVFP4` |
| LiteLLM | `https://dell.fulton-home.fultonengineeringservices.com/litellm/v1/...` | `cluster/mimo-v2.6-flash-rl-nvfp4` (needs `LITELLM_API_KEY`) |
| direct  | `http://10.10.10.165:8000/v1/...` (rank-0, Qwen3.8-27B-Coder390 TP=2) | `local-inference-lab/Qwen3.8-27B-Coder390-DFlash2` |
| LiteLLM | `https://dell.fulton-home.fultonengineeringservices.com/litellm/v1/...` | `cluster/qwen3.8-27b-coder390` (needs `LITELLM_API_KEY`) |

## Task/backend constraints

- **ARC** (`arc_easy`) is `multiple_choice` → needs loglikelihood → runs on the
  direct `local-completions` (`/v1/completions`) path only. Over LiteLLM it is
  not supported via chat-completions; if needed, point a `local-completions`
  config at `…/litellm/v1/completions` after verifying LiteLLM proxies
  logprob-bearing completions.
- **HumanEval** (`humaneval_instruct`) runs code → the CLI appends
  `--confirm_run_unsafe_code` automatically.
- **MBPP** (`mbpp_instruct`, `mbpp_plus_instruct`) is also `unsafe_code: true`
  → the CLI auto-confirms it too (the check is prefix-based over
  `humaneval`/`mbpp`/`cruxeval`/`longproc`, so every such task is covered).
- **MMMU** (`mmmu_val`) is multimodal → `local-chat-completions` with
  `max_images = 3` sends base64 image content parts to the router.
- **Generation budget.** `[run].gen_kwargs` (e.g. `gen_kwargs = "max_gen_toks=4096"`)
  is passed through to `--gen_kwargs`, widening the per-task token budget for
  the Qwen3.8 thinking model. It applies only to `generate_until` tasks (not
  `arc_easy`).
- **Stop sequences must be neutralised for this reasoning model.** lm-eval
  forwards `until` as the API `stop` list *and* applies it client-side as
  post-hoc truncation (`generate_until`), so the task's default markers bite
  twice. `humaneval_instruct`'s default
  `["\nclass","\ndef","\n#","\nif","\nprint"]` matches inside the model's
  `` block — cutting generation before the closing think tag, so vLLM's
  `qwen3` reasoning parser returns `content: null` and the harness logs
  "API returned null content" and substitutes
  `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER`. Equally, any stop that can match the
  answer (e.g. `"\ndef"`) truncates the generated function at its own
  definition line, so `pass@1` stays 0 even though the full function was
  generated. The code-eval configs therefore set
  `until=<<<NO_STOP>>>` (a sentinel that never occurs) to disable stops; the
  model closes its code fence and stops on its own. The key must still be
  present — `--gen_kwargs` is merged via `set_config(update=True)`, so omitting
  it leaves the task's markers in force. Only a single stop value is
  expressible because `lm_eval.utils.handle_arg_string` does no list parsing.
- **Tokenizer wiring.** The vllm-router exposes neither `/tokenizer_info` (TGI)
  nor `/tokenize`, so `tokenizer_backend = "remote"` fails. Instead:
  - completions/loglikelihood configs (ARC) use `tokenizer_backend = "huggingface"`
    with an explicit `tokenizer` override: `XiaomiMiMo/MiMo-V2.6-Flash-RL` for
    MiMo (the served `…-NVFP4` repo is gated but its public base hosts the
    tokenizer), and `Qwen/Qwen3.8-27B` for Qwen3.8-27B-Coder390 (the served id
    is a synthetic local slug, not on the Hub; the public upstream base shares
    its `Qwen2Tokenizer`).
  - chat/multimodal configs (HumanEval, MMMU) use `tokenizer_backend = "none"`
    (the multimodal path asserts no local tokenizer).
- **Auth.** `[model].auth_token_env` names an environment variable whose value
  is sent as the request `Authorization: Bearer <token>` header. The CLI
  forwards it to `OPENAI_API_KEY` for the run (the harness builds that header
  from `OPENAI_API_KEY`, not from the `auth_token` model arg); when
  `tokenizer_backend = "remote"` it is *also* passed as `auth_token` for the
  remote tokenizer. The token is never written into `model_args`.

## Results & visualization

Results land in `eval/results/<run-name>/` (`results_*.json` + per-task
`samples_*.jsonl`). `viz --local` prints a per-task metric table; `--html`
writes a self-contained `results/report.html`. `viz --zeno` uploads one Zeno
project per task (mirroring the harness's own `scripts/zeno_visualize.py`
column-building logic) to the endpoint in the config's `[zeno]` table — set
`[zeno].enabled`/endpoint to a self-hosted zeno-hub once one exists; until
then the local report is the default.
