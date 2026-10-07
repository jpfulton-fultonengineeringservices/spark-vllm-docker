"""Shared types and atomic-IO helpers for distributed EXL3 quantization.

This module is intentionally torch-free at the top level: ``save_h_file`` and
``load_h_file`` import ``safetensors.torch`` lazily inside the function body so
that host tests and tooling can exercise the rest of the contract without
pulling torch. ``distributed.py`` (coordinator) and ``worker.py`` (per-node
worker) both depend on the public surface defined here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ShardSpec",
    "WorkerEndpoint",
    "cfg_hash",
    "sha256_file",
    "write_atomic",
    "write_done_marker",
    "read_done_marker",
    "save_h_file",
    "load_h_file",
    "coverage_assert",
]

# Scalar H-record fields persisted in safetensors metadata. Each value is
# JSON-encoded so the round-trip preserves int-vs-str and int-vs-tuple exactly.
_H_RECORD_META_KEYS: tuple[str, ...] = (
    "count",
    "num_total",
    "inf_nan",
    "first_key",
    "device",
    "finalized",
)


@dataclass(frozen=True, slots=True)
class ShardSpec:
    """Job descriptor for one (module, shard) work unit."""

    job_id: str
    module_idx: int
    module_key: str
    shard_idx: int
    linear_keys: tuple[str, ...]
    qmaps: tuple[str, ...]
    h_dir: str
    weights_source: str
    result_uri: str
    cfg_hash: str

    def to_json(self) -> str:
        """Serialize to a canonical JSON string (sorted keys, compact)."""
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "module_idx": self.module_idx,
            "module_key": self.module_key,
            "shard_idx": self.shard_idx,
            "linear_keys": list(self.linear_keys),
            "qmaps": list(self.qmaps),
            "h_dir": self.h_dir,
            "weights_source": self.weights_source,
            "result_uri": self.result_uri,
            "cfg_hash": self.cfg_hash,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> ShardSpec:
        """Parse a JSON string produced by :meth:`to_json` back into a ShardSpec."""
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError(f"ShardSpec JSON must be an object, got {type(d).__name__}")
        try:
            linear_keys_raw = d["linear_keys"]
            qmaps_raw = d["qmaps"]
        except KeyError as e:
            raise ValueError(f"ShardSpec JSON missing required field: {e.args[0]}") from None
        if not isinstance(linear_keys_raw, list) or not isinstance(qmaps_raw, list):
            raise ValueError("ShardSpec JSON fields 'linear_keys' and 'qmaps' must be lists")
        try:
            return cls(
                job_id=str(d["job_id"]),
                module_idx=int(d["module_idx"]),
                module_key=str(d["module_key"]),
                shard_idx=int(d["shard_idx"]),
                linear_keys=tuple(str(k) for k in linear_keys_raw),
                qmaps=tuple(str(q) for q in qmaps_raw),
                h_dir=str(d["h_dir"]),
                weights_source=str(d["weights_source"]),
                result_uri=str(d["result_uri"]),
                cfg_hash=str(d["cfg_hash"]),
            )
        except KeyError as e:
            raise ValueError(f"ShardSpec JSON missing required field: {e.args[0]}") from None


@dataclass(frozen=True, slots=True)
class WorkerEndpoint:
    """Routing address for a single per-node quantization worker."""

    node: str
    inbox: str
    device: int


def cfg_hash(
    *,
    exllamav3_version: str,
    wheel_sha: str,
    bits: int,
    codebook: str,
    recipe_strategy: str,
    seed_basis: str,
    devices: tuple[int, ...],
    apply_out_scales: bool,
    source_index_hash: str,
    weights_digest: str,
) -> str:
    """Return a deterministic SHA-256 hex binding every quantizer knob + input bytes.

    The canonical JSON (sorted keys, compact separators) is hashed so that a
    different value in any field produces a different digest. This is the same
    digest referenced in :class:`ShardSpec`; the coordinator refuses to accept
    a worker's result if its ``cfg_hash`` doesn't match the one it issued for
    the shard.
    """
    payload: dict[str, Any] = {
        "exllamav3_version": exllamav3_version,
        "wheel_sha": wheel_sha,
        "bits": int(bits),
        "codebook": codebook,
        "recipe_strategy": recipe_strategy,
        "seed_basis": seed_basis,
        "devices": [int(d) for d in devices],
        "apply_out_scales": bool(apply_out_scales),
        "source_index_hash": source_index_hash,
        "weights_digest": weights_digest,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Atomically replace *path* with the bytes produced by *write_fn*.

    NFS-safe protocol: the writer produces bytes at a sibling ``.tmp`` path in
    the same directory; the tmp file is fsynced, ``os.replace`` swaps it into
    place (atomic on POSIX within one filesystem), and the directory is
    fsynced so the rename itself survives a crash. The writer never touches
    *path* directly, so readers never observe a partial file. The ``.tmp``
    file is cleaned up if any step fails.
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        write_fn(tmp)
        _fsync_file(tmp)
        os.replace(tmp, path)
        _fsync_dir(parent)
    except BaseException:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems reject fsync on directory handles; the rename is
        # still atomic, so this is best-effort durability only.
        pass
    finally:
        os.close(fd)


def write_done_marker(done_path: Path, content_sha256: str) -> None:
    """Write the data-file SHA-256 hex to *done_path* atomically.

    The done-marker is the coordinator's "shard complete" signal on NFS. The
    hex string is written exactly as given (no trailing newline); the
    coordinator reads it back and refuses to commit if the recorded hash
    doesn't match the data file it just scanned.
    """
    if not isinstance(content_sha256, str) or not content_sha256:
        raise ValueError("content_sha256 must be a non-empty hex string")

    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content_sha256)

    write_atomic(done_path, _write)


def read_done_marker(done_path: Path) -> str:
    """Read the hex digest previously written by :func:`write_done_marker`.

    Raises :class:`FileNotFoundError` if *done_path* is missing, or
    :class:`ValueError` if the marker is empty / whitespace-only. The
    coordinator treats both errors as a failed shard.
    """
    text = Path(done_path).read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"done marker is empty: {done_path}")
    return stripped


def _encode_h_record_meta(record: dict[str, Any]) -> dict[str, str]:
    """JSON-encode each scalar H-record field for safetensors metadata.

    JSON (not ``str()``) preserves the distinction between ``0`` and ``"0"``,
    and between ``5`` and ``(1, 2)``, so :func:`_decode_h_record_meta` can
    rebuild the original Python types exactly.
    """
    out: dict[str, str] = {}
    for key in _H_RECORD_META_KEYS:
        if key not in record:
            raise ValueError(f"H-record missing required field: {key}")
        out[key] = json.dumps(record[key])
    return out


def _decode_h_record_meta(raw: dict[str, str]) -> dict[str, Any]:
    """Reverse :func:`_encode_h_record_meta`, restoring original Python types."""
    out: dict[str, Any] = {}
    for key in _H_RECORD_META_KEYS:
        if key not in raw:
            raise ValueError(f"H-record metadata missing field: {key}")
        val = json.loads(raw[key])
        # JSON has no tuple type; ``inf_nan`` is ``int | tuple``, so a decoded
        # list must be rebuilt as a tuple to restore the original type exactly.
        if key == "inf_nan" and isinstance(val, list):
            val = tuple(val)
        out[key] = val
    return out


def save_h_file(path: Path, record: dict[str, Any]) -> None:
    """Persist a single H-record dict to *path* atomically.

    The record must contain a concrete fp32 tensor ``H`` plus the scalar
    fields listed in the module docstring. ``H_swap_device`` (if present) is
    stripped before writing — the worker re-derives the home device on load.
    The tensor is stored via :mod:`safetensors.torch` (imported lazily); the
    scalar fields are carried in the safetensors ``metadata`` dict.
    """
    if "H" not in record:
        raise ValueError("H-record missing required key: 'H'")
    h_tensor = record["H"]
    # Lazy import so this module stays torch-free at import time.
    from safetensors.torch import save_file

    # Strip any per-worker device cache; the worker that loads the file
    # re-derives the home device.
    meta_in = {k: v for k, v in record.items() if k not in ("H", "H_swap_device")}
    meta_str = _encode_h_record_meta(meta_in)

    def _write(tmp: Path) -> None:
        save_file({"H": h_tensor}, str(tmp), metadata=meta_str)

    write_atomic(Path(path), _write)


def load_h_file(path: Path) -> dict[str, Any]:
    """Reverse :func:`save_h_file`: return the full H-record dict.

    Asserts that ``H`` is a concrete tensor (never a meta tensor). The
    returned dict is a fresh mapping with exactly the fields written by
    :func:`save_h_file`; ``H_swap_device`` is never present.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    loaded = load_file(str(path))
    if "H" not in loaded:
        raise ValueError(f"loaded H file missing 'H' tensor: {path}")
    h_tensor = loaded["H"]
    # A meta tensor here would be a worker-side bug; refuse it loudly.
    if bool(getattr(h_tensor, "is_meta", False)):
        raise ValueError(f"H tensor is a meta tensor (not concrete): {path}")
    with safe_open(str(path), framework="pt") as f:
        raw_meta: dict[str, str] = dict(f.metadata() or {})
    record: dict[str, Any] = {"H": h_tensor}
    record.update(_decode_h_record_meta(raw_meta))
    return record


def coverage_assert(planned_keys: AbstractSet[str], gathered_keys: AbstractSet[str]) -> None:
    """Raise if *gathered_keys* doesn't exactly match *planned_keys*.

    Generalized form of the merge-layer per-key disjointness/coverage check:
    a shard is "done" only if it returned every key the coordinator planned
    and nothing else. Sets can't carry duplicates, so overlap is impossible;
    both directions are still checked so the failure message is unambiguous.
    """
    missing = planned_keys - gathered_keys
    extra = gathered_keys - planned_keys
    if not missing and not extra:
        return
    parts: list[str] = []
    if missing:
        parts.append(f"missing={sorted(missing)}")
    if extra:
        parts.append(f"extra={sorted(extra)}")
    raise ValueError("coverage mismatch: " + ", ".join(parts))
