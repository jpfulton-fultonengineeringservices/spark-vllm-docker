# mods/mimo-v2.6-flash-rl-nvfp4/tools

Offline gate for the mod's `run.sh` patcher. It runs the patcher against a
captured **clean** copy of the image's `mimo_v2.py` in a throwaway `PYTHON_ROOT`
and `compile()`s the result — catching anchor/indent drift that would otherwise
only surface as an engine-boot crash.

## `test_patcher.sh`

```bash
tools/test_patcher.sh <mod-run.sh> <fixture.py> [relpath]
# e.g.
tools/test_patcher.sh run.sh tools/fixtures/mimo_v2_b12x.py
```

- Scaffolds the minimal `vllm/...` tree the mod touches under a temp
  `PYTHON_ROOT`/`VLLM_SITE_PACKAGES`, copies the fixture to the in-image
  relative path, runs the mod, then `compile()`s every `.py`.
- Asserts the file actually changed (anchor hit), compiles, and is idempotent
  (a second run must not re-break it).
- Exit non-zero + a printed `SyntaxError` on failure.

## `tools/fixtures/mimo_v2_b12x.py`

Byte-exact copy of the image's
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py`
from `vllm-node-b12x:latest` (sha256 `fc310f21…`, 909 lines). Refresh it when
the image is rebuilt:

```bash
docker run --rm --entrypoint bash vllm-node-b12x:latest -c \
  'cat /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py' \
  > tools/fixtures/mimo_v2_b12x.py
```

## Why `compile()`, not `ast.parse()`

Since CPython 3.8 the `'continue'`/`'break'` "not properly in loop" check runs
at the **codegen/symtable** stage, so `ast.parse()` silently accepts a patch
that dedents a `for`-loop body and orphans the following `continue`. That is
exactly the corruption an 8-space `EDIT_B` anchor produced (the 8-space string
is a substring of the real 12-space line, so `replace()` kept the line but
emitted the inserted block at 8 spaces). The mod's own in-`run.sh` acceptance
check therefore uses `compile()` too.

## Strongest check (still)

The container smoke test is authoritative and is what the fleet uses:

```bash
docker run --rm -v "$PWD/mods/mimo-v2.6-flash-rl-nvfp4/run.sh:/tmp/run.sh:ro" \
  --entrypoint bash vllm-node-b12x:latest -c \
  'bash /tmp/run.sh; python3 -c "compile(open(\"/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py\").read(),\"x\",\"exec\"); print(\"COMPILE_OK\")"'
```
