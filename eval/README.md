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

./eval/eval.sh viz results --local --html                 # markdown + report.html
ZENO_API_KEY=... ./eval/eval.sh viz results --zeno        # upload to Zeno hub
```

## Endpoints

| target  | base_url                                              | model id                          |
|---------|-------------------------------------------------------|-----------------------------------|
| direct  | `http://10.10.10.165:8000/v1/...` (rank-0 vllm-router)| `XiaomiMiMo/MiMo-V2.6-Flash-RL-NVFP4` |
| LiteLLM | `https://dell.fulton-home.fultonengineeringservices.com/litellm/v1/...` | `cluster/mimo-v2.6-flash-rl-nvfp4` (needs `LITELLM_API_KEY`) |

## Task/backend constraints

- **ARC** (`arc_easy`) is `multiple_choice` → needs loglikelihood → runs on the
  direct `local-completions` (`/v1/completions`) path only. Over LiteLLM it is
  not supported via chat-completions; if needed, point a `local-completions`
  config at `…/litellm/v1/completions` after verifying LiteLLM proxies
  logprob-bearing completions.
- **HumanEval** (`humaneval_instruct`) runs code → the CLI appends
  `--confirm_run_unsafe_code` automatically.
- **MMMU** (`mmmu_val`) is multimodal → `local-chat-completions` with
  `max_images = 3` sends base64 image content parts to the router.
- **Tokenizer wiring.** The vllm-router exposes neither `/tokenizer_info` (TGI)
  nor `/tokenize`, so `tokenizer_backend = "remote"` fails. Instead:
  - completions/loglikelihood configs (ARC) use `tokenizer_backend = "huggingface"`
    with an explicit `tokenizer = "XiaomiMiMo/MiMo-V2.6-Flash-RL"` override, since
    the served `…-NVFP4` repo is gated but its public base hosts the tokenizer.
  - chat/multimodal configs (HumanEval, MMMU) use `tokenizer_backend = "none"`
    (the multimodal path asserts no local tokenizer).

## Results & visualization

Results land in `eval/results/<run-name>/` (`results_*.json` + per-task
`samples_*.jsonl`). `viz --local` prints a per-task metric table; `--html`
writes a self-contained `results/report.html`. `viz --zeno` uploads one Zeno
project per task (mirroring the harness's own `scripts/zeno_visualize.py`
column-building logic) to the endpoint in the config's `[zeno]` table — set
`[zeno].enabled`/endpoint to a self-hosted zeno-hub once one exists; until
then the local report is the default.
