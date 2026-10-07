"""Node → local-path → checkpoint resolution.

The pack builder prefers a source checkpoint on the target node's **local
NVMe** (fast) and falls back to ``/nas-1`` when the node has no local entry.
Backed by ``node-model-map.json`` (schema ``node-model-map/v1``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NAS_ROOT = Path("/nas-1")
DEFAULT_LOCAL_ROOT = Path("/opt/llm")
DEFAULT_WORK_ROOT = Path("/opt/llm/fes-projects/exl3-mimo-build")


@dataclass(frozen=True)
class Checkpoint:
    slug: str
    path: str
    role: str
    shards: int | None = None
    save_format: str | None = None
    tp_size: int | None = None
    bytes: int | None = None


@dataclass(frozen=True)
class Node:
    id: str
    alias: str
    cluster_node: str
    local_root: str
    checkpoints: dict[str, Checkpoint]


@dataclass(frozen=True)
class NodeMap:
    schema: str
    local_root: str
    nodes: dict[str, Node]

    def by_cluster_node(self, cluster_node: str) -> Node | None:
        for node in self.nodes.values():
            if node.cluster_node == cluster_node or node.alias == cluster_node:
                return node
        return None

    def by_alias(self, alias: str) -> Node | None:
        for node in self.nodes.values():
            if node.alias == alias or node.id == alias:
                return node
        return None


def load_node_map(path: Path) -> NodeMap:
    data = json.loads(path.read_text())
    if data.get("schema") != "node-model-map/v1":
        raise ValueError(f"{path}: unexpected schema {data.get('schema')!r}")
    nodes: dict[str, Node] = {}
    for nid, raw in data["nodes"].items():
        cps: dict[str, Checkpoint] = {}
        for slug, c in raw["checkpoints"].items():
            cps[slug] = Checkpoint(
                slug=slug,
                path=c["path"],
                role=c.get("role", "source"),
                shards=c.get("shards"),
                save_format=c.get("save_format"),
                tp_size=c.get("tp_size"),
                bytes=c.get("bytes"),
            )
        nodes[nid] = Node(
            id=nid,
            alias=raw["alias"],
            cluster_node=raw["cluster_node"],
            local_root=raw.get("local_root", data.get("local_root", str(DEFAULT_LOCAL_ROOT))),
            checkpoints=cps,
        )
    return NodeMap(
        schema=data["schema"],
        local_root=data.get("local_root", str(DEFAULT_LOCAL_ROOT)),
        nodes=nodes,
    )


def resolve_source(
    node_map: NodeMap,
    node: str,
    slug: str,
    *,
    nas_root: Path = DEFAULT_NAS_ROOT,
) -> Path | None:
    """Return the source checkpoint path for ``node``/``slug``.

    Order: node-local NVMe copy (if the map lists one) → NAS twin. Returns
    ``None`` when neither is configured. Existence is *not* checked here so the
    function is usable off-node (e.g. dry-run planning); callers that execute
    should verify ``.exists()``.
    """
    n = node_map.by_alias(node) or node_map.by_cluster_node(node)
    if n is not None:
        cp = n.checkpoints.get(slug)
        if cp is not None:
            return Path(cp.path)
    # NAS fallback: /nas-1/models/mimo/<slug>
    return nas_root / "models" / "mimo" / slug


def resolve_work_root(
    node_map: NodeMap,
    node: str,
    *,
    local_root: Path | None = None,
) -> Path:
    """Work/output staging root on the node (local NVMe by default)."""
    if local_root is not None:
        base = local_root
    else:
        n = node_map.by_alias(node) or node_map.by_cluster_node(node)
        base = Path(n.local_root) if n is not None else DEFAULT_LOCAL_ROOT
    return base / "fes-projects" / "exl3-mimo-build"


def resolve_destination(slug: str, out_root: Path) -> Path:
    """Final pack output directory (on the NAS by default)."""
    return out_root / f"{slug}-exl3-v1"


def default_map_path() -> Path:
    """Locate node-model-map.json next to the package or via env override."""
    env = os.environ.get("EXL3_NODE_MODEL_MAP")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "node-model-map.json"
        if candidate.exists():
            return candidate
    return here.parent / "node-model-map.json"
