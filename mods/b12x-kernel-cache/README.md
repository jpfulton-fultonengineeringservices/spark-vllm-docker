# b12x kernel-cache mod

Archives the b12x/CuTe autotuned-kernel cache after a successful serve and
restores a matching archive before later serves, skipping the multi-minute
cold tune. No equivalent archive/replay mechanism exists upstream
(`docker/B12X_CACHE_INTEGRITY.md` covers atomic on-disk writes only).

## How it works

The mod installs `/usr/local/bin/b12x-kcache` (backed by `kcache.sh`, with
implementation modules under `lib/` next to it). A wired recipe wraps its
serve command:

```
b12x-kcache exec vllm serve <model> ... --tensor-parallel-size 2 ...
```

`exec` computes a version-keyed identity (below), restores a matching archive
if one exists, starts the serve, polls `/health`, and after the first healthy
poll saves a fresh archive. A serve never fails because of the wrapper: any
restore/save problem degrades to a log line and the cold path continues.

## Cache layout

`roots.txt` lists `compile` → `/root/.cache/b12x/compile` plus a sibling
`preparation` → `/root/.cache/b12x/preparation`. On current images b12x
writes the tuning-selection `preparation/` INSIDE `compile/` (not a sibling
as upstream `_cute_compile_cache_dir()` suggests), so the sibling entry is a
benign no-op: save's directory check skips it and the `compile/` archive
already carries `preparation/`. The entry is kept for a future upstream
layout change.

## Store

Candidates, in priority order:

1. `$B12X_KCACHE_STORE` — explicit operator-owned path
2. `/root/.cache/huggingface/.spark-vllm/b12x-kcache` — default; in-container
   this path sits on the `/cluster-shared` NFS filesystem (host path:
   `/cluster-shared/models/hf_cache/.spark-vllm/b12x-kcache`). The head rank
   can write it; root-squashed peers can only read.
3. `/root/.cache/b12x/_archives` — per-node fallback inside the b12x mount

`save` uses the first candidate that passes an actual write probe (peers fail
(2) and land on (3) automatically). `restore`/`verify` search all candidates
and use the first existing archive, so peers restore the head's archive from
the shared store. Every resolution logs `[b12x-kcache] store=<path>
shared=yes|no`.

Archive: `<store>/<key>.tar.zst` (zstd, gzip fallback) plus `INDEX.tsv`
(`key<TAB>created_iso<TAB>node<TAB>bytes<TAB>sha256`).

## Key inputs

`b12x-kcache key` hashes a canonical tuple joined with the unit separator:
schema literal `b12x-kcache/v1`, b12x revision (`/workspace/b12x-source-commit`
or b12x dist version), `nvidia-cutlass-dsl` version, torch version, vllm
version, sorted sha256s of vllm `_C*.so`, `$CUTE_DSL_ARCH`, and the serve
argv's `--tensor-parallel-size` (default 1), `--quantization`,
`--attention-backend`, `--linear-backend`, `--kv-cache-dtype`, `--block-size`,
`--max-model-len`, model id, plus the sorted cache-root list. Key form:
`b12x-kernels__<model-short>__tp<T>__<hash16>`.

Unresolvable hard-fail fields (`b12x_rev`, `cutlass_dsl`, `torch`, `vllm`,
`arch`, `model`, `roots`) abort with
`[b12x-kcache] key: cannot resolve <field>` and exit 3 — never a placeholder
key, which would silently restore the wrong kernel set.

## Env knobs

| Variable | Default | Meaning |
|---|---|---|
| `B12X_KCACHE_STORE` | unset | Override store path (candidate 1) |
| `B12X_KCACHE_PRUNE` | `3` | Archives kept per key family after save |
| `B12X_KCACHE_SAVE` | `1` | `0` disables the post-health save |
| `B12X_KCACHE_FORCE_AUTOTUNE_OFF` | unset | `true` exports `B12X_AUTOTUNE=0` to the child after a successful restore |
| `B12X_KCACHE_ROOTS` | `<mod>/roots.txt` | Test hook: alternate roots file (`name<TAB>path`) |

## Seeding a peer manually

If a peer tuned first (head cold) its save lands node-local (candidate 3);
copy it over and restore:

```
scp <node>:/root/.cache/b12x/_archives/<key>.tar.zst <store>/
b12x-kcache restore            # inside the peer container
```

## Wired recipes

- `recipes/mimo-v2.6-flash-exl3-2x.yaml`
- `recipes/mimo-v2.6-flash-exl3-1x.yaml`

## Intentionally left cold (decision, not omission)

Wiring any of these later is the same two-line edit (add the mod; prefix the
serve command with `b12x-kcache exec`):

- `recipes/qwen3.8-flash-next-nvfp4-cluster.yaml`
- `recipes/deepseek-v4-flash-vision-exp.yaml`
- `recipes/qwen3.8-flash-next-nvfp4-solo.yaml`
- `recipes/deepseek-v4-flash-0731.yaml`
- `recipes/mimo-v2.6-flash-2x.yaml`
- `recipes/8x-spark-cluster/glm-5.2-nvfp4.yaml`
- `recipes/mimo-v2.6-flash-4x.yaml`
- `recipes/glm-5.3-flash.yaml`
- `recipes/3x-spark-cluster/mimo-v2.6-flash-pp3.yaml`
