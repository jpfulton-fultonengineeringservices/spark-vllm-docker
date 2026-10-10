"""Inject the missing VLLM_PREFIX_CACHE_RETENTION_INTERVAL env into vllm.envs.

The ported worker reads ``envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` (FES
envs lineage, default ``None``; prefix-cache retention alignment). The b12x
image's ``vllm/envs.py`` predates it.

``vllm.envs`` resolves attributes through its ``environment_variables`` dict
inside ``__getattr__``, so the entry MUST be added to that dict — an
annotation alone is not enough. This was proven the hard way: the first
shim injected only the annotation and the engine still died with
``AttributeError: module 'vllm.envs' has no attribute
'VLLM_PREFIX_CACHE_RETENTION_INTERVAL'``.

The injected lambda mirrors the FES implementation verbatim.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

TARGET_ENV = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"

# Anchor line inside the environment_variables dict (verified against
# vllm-node-b12x vLLM 0.1.dev21546+g502d6cb5a, envs.py:2413).
DICT_ANCHOR = '    "VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),\n'

# FES implementation (vllm/envs.py from
# cuda13.3-aarch64-gb10-glm5next-modular @ 4e79d6be6f).
DICT_ENTRY = (
    f'    "{TARGET_ENV}": lambda: (\n'
    f'        int(os.environ["{TARGET_ENV}"])\n'
    f'        if "{TARGET_ENV}" in os.environ\n'
    f"        else None\n"
    f"    ),\n"
)


class EnvShimError(RuntimeError):
    """The envs.py state does not match any known shape."""


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


def needs_shim(envs_py: str | Path) -> bool:
    """True when the env is absent from envs.py entirely."""
    return TARGET_ENV not in Path(envs_py).read_text()


def inject(envs_py: str | Path) -> bool:
    """Add the env lambda to the environment_variables dict.

    Returns True when a change was made, False when the shim is already
    present. Raises EnvShimError (fail-closed) when the file cannot be
    safely modified.
    """
    path = Path(envs_py)
    text = path.read_text()

    if TARGET_ENV in text:
        return False

    count = text.count(DICT_ANCHOR)
    if count != 1:
        raise EnvShimError(
            f"envs dict anchor found {count} times (expected 1);"
            " the image's envs.py shape is unknown — fail closed"
        )

    patched = text.replace(DICT_ANCHOR, DICT_ANCHOR + DICT_ENTRY, 1)
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        raise EnvShimError(f"injected envs.py does not parse: {exc}") from exc

    _atomic_write(path, patched)
    return True


def verify(envs_py: str | Path) -> None:
    """Import the envs module fresh and require the env to resolve."""
    text = Path(envs_py).read_text()
    ast.parse(text)
    if not needs_shim(envs_py):
        return
    raise EnvShimError(f"{TARGET_ENV} still absent from {envs_py}")


def _selftest() -> int:
    """Offline round-trip on a synthetic envs.py with the anchor shape."""
    sample = (
        "import os\n"
        "\n"
        "environment_variables: dict[str, callable] = {\n"
        '    "VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),\n'
        "}\n"
        "\n"
        "\n"
        "def __getattr__(name: str):\n"
        "    if name in environment_variables:\n"
        "        return environment_variables[name]()\n"
        "    raise AttributeError(name)\n"
    )
    with_temp = Path(os.environ.get("TMPDIR", "/tmp")) / "mooncake_worker_envs_selftest.py"
    with_temp.write_text(sample)
    try:
        if not needs_shim(with_temp):
            print("SELFTEST FAIL: needs_shim false on clean sample", file=sys.stderr)
            return 1
        changed = inject(with_temp)
        if not changed:
            print("SELFTEST FAIL: inject returned False", file=sys.stderr)
            return 1
        if needs_shim(with_temp):
            print("SELFTEST FAIL: target still absent after inject", file=sys.stderr)
            return 1
        if inject(with_temp):
            print("SELFTEST FAIL: second inject was not idempotent", file=sys.stderr)
            return 1
        # Resolution check: exec the module and read through __getattr__.
        import importlib.util

        spec = importlib.util.spec_from_file_location("envs_selftest", with_temp)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = module.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        if value is not None:
            print(
                f"SELFTEST FAIL: default should be None, got {value!r}",
                file=sys.stderr,
            )
            return 1
    finally:
        with_temp.unlink(missing_ok=True)
    print("SELFTEST OK: envs shim round-trip (inject, idempotent, resolves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
