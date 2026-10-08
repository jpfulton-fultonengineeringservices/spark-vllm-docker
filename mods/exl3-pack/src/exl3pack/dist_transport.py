"""Transport / shard-planning logic for the distributed EXL3 coordinator.

Provides ``_TransportMixin`` with H publication, shard planning, dispatch,
gather, and weights-digest methods. Also houses the ``_exllamav3_version``
helper used by ``_plan_shards``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from exllamav3.modules.linear import Linear
from safetensors.torch import load_file

if TYPE_CHECKING:
    from exl3pack.cli import CoordinatorArgs
else:
    CoordinatorArgs = dict[str, Any]  # type: ignore[misc,assignment]
from exl3pack.dist_types import (
    ShardSpec,
    WorkerEndpoint,
    cfg_hash,
    read_done_marker,
    save_h_file,
    sha256_file,
    write_atomic,
)
from exl3pack.logconfig import log_event


def _exllamav3_version() -> str:
    """Installed exllamav3 distribution version ("" when not installed)."""
    try:
        return metadata.version("exllamav3")
    except metadata.PackageNotFoundError:
        return ""


# Worker heartbeats are touched every _POLL_INTERVAL_S (1s) in worker.serve.
# A stale heartbeat does NOT kill a gather: NFS attribute caching routinely
# serves mtimes minutes old, and the GIL can starve the daemon touch thread
# during multi-minute MoE shards. Staleness is a diagnostic only -- the
# gather deadline is the real failure signal. This must stay comfortably
# above NFS actimeo so it never fires on a live worker.
_HEARTBEAT_WARN_S = 60.0


class _TransportMixin:
    """Transport methods, mixed into ``Coordinator``."""

    log: logging.Logger

    # Union: pre-prepare CoordinatorArgs, post-prepare the merged in_args
    # superset (prepare() adds derived keys).
    args: CoordinatorArgs | dict[str, Any]
    work: Path
    endpoints: list[WorkerEndpoint]
    gather_timeout: float
    _module_idx: int
    _h_dir: str
    _index_sha: str | None
    _planned_keys: set[str]
    _strategy: dict[str, float]

    # ------------------------------------------------------------------ #
    #  H publication                                                      #
    # ------------------------------------------------------------------ #

    def _publish_H(
        self,
        module_key: str,
        capture_H: dict[str, dict[str, Any]],
        qmaps: set[str],
    ) -> str:
        """Write one immutable H file per distinct qmap; return the H dir."""
        h_dir = self.work / "dist" / f"mod{self._module_idx}" / "h"
        h_dir.mkdir(parents=True, exist_ok=True)
        for qmap in sorted(qmaps):
            rec = capture_H.get(qmap)
            if rec is None:
                raise ValueError(
                    f"{module_key}: no captured H for qmap {qmap!r}"
                )
            h = rec.get("H")
            if (
                not isinstance(h, torch.Tensor)
                or bool(getattr(h, "is_meta", False))
            ):
                raise ValueError(
                    f"{module_key}: H for qmap {qmap!r} is missing or meta"
                )
            if h.dtype != torch.float32:
                raise ValueError(
                    f"{module_key}: H for qmap {qmap!r} is {h.dtype}"
                )
            inf_nan_raw = rec["inf_nan"]
            inf_nan: int | tuple[int, ...] = (
                int(inf_nan_raw)
                if isinstance(inf_nan_raw, int)
                else tuple(int(v) for v in inf_nan_raw)
            )
            dev = rec["device"]
            record: dict[str, Any] = {
                "H": h.cpu(),
                "count": int(rec["count"]),
                "num_total": int(rec["num_total"]),
                "inf_nan": inf_nan,
                "first_key": str(rec["first_key"]),
                "device": dev
                if isinstance(dev, (str, int))
                else str(dev),
                "finalized": bool(rec["finalized"]),
            }
            qmap_hash = hashlib.sha256(qmap.encode("utf-8")).hexdigest()
            save_h_file(h_dir / f"{qmap_hash}.safetensors", record)
        self._h_dir = str(h_dir.resolve())
        return self._h_dir

    # ------------------------------------------------------------------ #
    #  Shard planning                                                     #
    # ------------------------------------------------------------------ #

    def _plan_shards(
        self,
        module: Any,
        groups: list[list[Linear]],
        strategy: dict[str, float],
        endpoints: Sequence[WorkerEndpoint],
    ) -> list[ShardSpec]:
        """Greedily assign whole groups to workers by ``weights_numel``."""
        if not endpoints:
            raise ValueError("no worker endpoints available")

        weighted: list[tuple[int, list[Linear]]] = []
        for group in groups:
            if not group:
                raise ValueError("empty linear group")
            weighted.append(
                (
                    sum(int(lin.weights_numel()) for lin in group),
                    group,
                )
            )
        weighted.sort(key=lambda t: (-t[0], str(t[1][0].key)))
        loads = [0] * len(endpoints)
        assigned: list[list[list[Linear]]] = [[] for _ in endpoints]
        for weight, group in weighted:
            wi = min(range(len(endpoints)), key=lambda i: (loads[i], i))
            loads[wi] += weight
            assigned[wi].append(group)

        module_key = str(getattr(module, "key", ""))
        weights_source = self._weights_source()
        out_dir = self.work / "dist" / f"mod{self._module_idx}" / "out"
        specs: list[ShardSpec] = []

        for wi, ep in enumerate(endpoints):
            linear_keys: list[str] = []
            qmaps: set[str] = set()
            for group in assigned[wi]:
                for lin in group:
                    linear_keys.append(str(lin.key))
                    if lin.qmap:
                        qmaps.add(str(lin.qmap))

            result_uri = str(
                (out_dir / ep.node / f"shard-{wi}.safetensors").resolve()
            )

            # Compute a per-shard weights digest (plan §4).
            weights_digest = self._weights_digest(linear_keys, module)

            ch = cfg_hash(
                exllamav3_version=_exllamav3_version(),
                wheel_sha=str(self.args.get("wheel_sha", "")),
                bits=int(self.args["bits"]),
                codebook=str(self.args["codebook"]),
                recipe_strategy=json.dumps(
                    strategy, sort_keys=True, separators=(",", ":")
                ),
                seed_basis=str(self.args.get("seed_basis", "module_idx")),
                devices=(int(ep.device),),
                apply_out_scales=bool(
                    self.args.get("apply_out_scales") or False
                ),
                source_index_hash=self._source_index_hash(
                    weights_source
                ),
                weights_digest=weights_digest,
            )
            specs.append(
                ShardSpec(
                    job_id=str(self.args.get("job_id", "exl3-dist")),
                    module_idx=self._module_idx,
                    module_key=module_key,
                    shard_idx=wi,
                    linear_keys=tuple(linear_keys),
                    qmaps=tuple(sorted(qmaps)),
                    h_dir=self._h_dir,
                    weights_source=weights_source,
                    result_uri=result_uri,
                    cfg_hash=ch,
                )
            )
        return specs

    # ------------------------------------------------------------------ #
    #  Dispatch / Gather                                                  #
    # ------------------------------------------------------------------ #

    def _dispatch(self, specs: list[ShardSpec]) -> None:
        """Write one ShardSpec JSON per shard into its worker inbox (atomic)."""
        for spec in specs:
            if spec.shard_idx >= len(self.endpoints):
                raise ValueError(
                    f"shard_idx {spec.shard_idx} has no endpoint"
                )
            ep = self.endpoints[spec.shard_idx]
            inbox = Path(ep.inbox)
            payload = spec.to_json()

            def _write(tmp: Path, p: str = payload) -> None:
                tmp.write_text(p, encoding="utf-8")

            write_atomic(inbox / f"shard-{spec.shard_idx}.json", _write)

    def _gather(
        self, specs: list[ShardSpec]
    ) -> dict[str, torch.Tensor]:
        """Wait for shard outputs (bounded), verify hashes, merge q_tensors.

        P1-d: per-shard gather timeout window (plan §5).
        """
        # Late import so mock.patch.object(distributed, "safe_open", ...)
        # in tests can intercept the call (plan §6 test compatibility).
        from exl3pack import distributed as _dist

        hb_dir = self.work / "dist" / "heartbeat"

        merged: dict[str, torch.Tensor] = {}
        for spec in specs:
            # P1-d: per-shard deadline
            deadline = time.monotonic() + self.gather_timeout
            data_path = Path(spec.result_uri)
            done_path = data_path.with_suffix(".done")
            spec_node = Path(spec.result_uri).parent.name
            while not done_path.exists():
                hb = hb_dir / spec_node
                try:
                    hb_age = time.time() - hb.stat().st_mtime
                except OSError:
                    hb_age = float("inf")  # no heartbeat yet: never written
                if hb_age > _HEARTBEAT_WARN_S:
                    log_event(
                        self.log,
                        logging.WARNING,
                        "coordinator.gather_heartbeat_stale",
                        fields={
                            "module": spec.module_key,
                            "shard": spec.shard_idx,
                            "node": spec_node,
                            "hb_age_s": round(hb_age, 1),
                        },
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{spec.module_key} shard {spec.shard_idx}: "
                        f"timeout after {self.gather_timeout:.0f}s"
                    )
                time.sleep(1.0)
            expected = read_done_marker(done_path)
            actual = sha256_file(data_path)
            if actual != expected:
                raise ValueError(
                    f"{spec.module_key} shard {spec.shard_idx}: output hash "
                    f"mismatch (marker {expected}, file {actual})"
                )
            with _dist.safe_open(str(data_path), framework="pt") as f:  # type: ignore[attr-defined]
                echo = (f.metadata() or {}).get("cfg_hash")
            if echo is not None and echo != spec.cfg_hash:
                raise ValueError(
                    f"{spec.module_key} shard {spec.shard_idx}: cfg_hash "
                    f"echo {echo} != issued {spec.cfg_hash}"
                )
            for key, tensor in load_file(str(data_path)).items():
                if key in merged:
                    raise ValueError(
                        f"key {key!r} returned by more than one shard"
                    )
                merged[key] = tensor
        return merged

    # ------------------------------------------------------------------ #
    #  Weights digest                                                     #
    # ------------------------------------------------------------------ #

    def _weights_source(self) -> str:
        """Path to the shared fp16 ``model.safetensors.index.json``."""
        index = (
            Path(str(self.args["in_dir"]))
            / "model.safetensors.index.json"
        )
        return str(index.resolve())

    def _source_index_hash(self, weights_source: str) -> str:
        """sha256 of the source index.json (cached)."""
        if self._index_sha is None:
            self._index_sha = sha256_file(Path(weights_source))
        return self._index_sha

    def _weights_digest(
        self,
        linear_keys: Sequence[str],
        module: Any,
    ) -> str:
        """Digest binding each shard key to its consumed fp16 weight bytes.

        For each key the sha256 of ``linear.inner.get_weight_tensor()`` is
        bound (files hashed once and cached), so a source-data edit changes
        every affected shard's cfg_hash.
        """
        # Build a key→Linear lookup from the module for weight-digest access.
        linear_map: dict[str, Linear] = {}
        for lin in module:
            if isinstance(lin, Linear):
                linear_map[str(lin.key)] = lin

        pairs: list[list[str]] = []
        for key in sorted(linear_keys):
            lin = linear_map.get(key)
            if lin is None:
                raise ValueError(
                    f"linear key {key!r} not found in module"
                )
            wt = lin.inner.get_weight_tensor()
            pairs.append(
                [key, hashlib.sha256(wt.cpu().numpy().tobytes()).hexdigest()]
            )
        blob = json.dumps(pairs, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
