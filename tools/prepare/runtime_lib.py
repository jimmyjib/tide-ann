"""Directed partitioned-runtime parsing, validation, and sealing helpers.

Runtime adjacency stores global neighbour IDs even though rows are owned by a
partition.  Keep that representation unchanged: METIS is only an input to a
new ownership map, never a replacement graph.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


class RuntimeErrorChecked(RuntimeError):
    """A validation failure that must prevent publication of an artifact."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_adj(path: Path, expected_rows: int | None = None) -> tuple[list[np.ndarray], int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise RuntimeErrorChecked(f"short adjacency header: {path}")
        file_size, max_degree, start_node, _ = struct.unpack("<QIIQ", header)
        if file_size != path.stat().st_size:
            raise RuntimeErrorChecked(f"adjacency size header mismatch: {path}")
        rows: list[np.ndarray] = []
        while handle.tell() < file_size:
            raw_degree = handle.read(4)
            if len(raw_degree) != 4:
                raise RuntimeErrorChecked(f"truncated degree: {path}")
            degree, = struct.unpack("<I", raw_degree)
            if degree > max_degree:
                raise RuntimeErrorChecked(f"degree exceeds header: {path}")
            raw = handle.read(degree * 4)
            if len(raw) != degree * 4:
                raise RuntimeErrorChecked(f"truncated neighbours: {path}")
            rows.append(np.frombuffer(raw, dtype="<u4").copy().astype(np.uint32))
        if handle.tell() != file_size:
            raise RuntimeErrorChecked(f"adjacency parse overrun: {path}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeErrorChecked(f"row count mismatch in {path}: {len(rows)} != {expected_rows}")
    if rows and start_node >= len(rows):
        raise RuntimeErrorChecked(f"start node outside partition: {path}")
    return rows, int(start_node), int(max_degree)


def write_adj(path: Path, rows: Iterable[np.ndarray], start_node: int, max_degree: int) -> int:
    normalized = [np.asarray(row, dtype="<u4") for row in rows]
    if not normalized:
        raise RuntimeErrorChecked("cannot write empty partition")
    if not 0 <= start_node < len(normalized):
        raise RuntimeErrorChecked("invalid partition start node")
    size = 24 + sum(4 + row.nbytes for row in normalized)
    with path.open("wb") as handle:
        handle.write(struct.pack("<QIIQ", size, int(max_degree), int(start_node), 0))
        for row in normalized:
            handle.write(struct.pack("<I", len(row)))
            handle.write(row.tobytes())
    return size


def load_global_adjacency(root: Path, n_nodes: int, parts: int) -> tuple[list[np.ndarray], int]:
    global_rows: list[np.ndarray | None] = [None] * n_nodes
    global_start: int | None = None
    for part in range(parts):
        directory = root / f"partition_{part}"
        l2g_path = directory / "local_to_global.npy"
        adj_path = directory / "adj"
        if not l2g_path.is_file() or not adj_path.is_file():
            raise RuntimeErrorChecked(f"missing partition payload: {directory}")
        l2g = np.load(l2g_path, mmap_mode="r")
        if l2g.ndim != 1 or not np.issubdtype(l2g.dtype, np.integer):
            raise RuntimeErrorChecked(f"invalid local_to_global: {l2g_path}")
        if len(l2g) == 0 or np.any(l2g < 0) or np.any(l2g >= n_nodes) or len(np.unique(l2g)) != len(l2g):
            raise RuntimeErrorChecked(f"invalid local_to_global IDs: {l2g_path}")
        rows, start, _ = read_adj(adj_path, len(l2g))
        for local, global_id in enumerate(l2g):
            gid = int(global_id)
            if global_rows[gid] is not None:
                raise RuntimeErrorChecked(f"duplicate global row {gid}")
            row = rows[local]
            if np.any(row >= n_nodes):
                raise RuntimeErrorChecked(f"neighbour out of range at node {gid}")
            global_rows[gid] = row
        candidate = int(l2g[start])
        if global_start is None:
            global_start = candidate
    if any(row is None for row in global_rows):
        raise RuntimeErrorChecked("partition rows do not cover all global nodes")
    return [row for row in global_rows if row is not None], int(global_start or 0)


def ordered_graph_hash(rows: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for gid, row in enumerate(rows):
        payload = np.asarray(row, dtype="<u4")
        digest.update(struct.pack("<II", gid, len(payload)))
        digest.update(payload.tobytes())
    return digest.hexdigest()


def validate_membership(membership: np.ndarray, n_nodes: int, parts: int) -> np.ndarray:
    if membership.ndim != 1 or len(membership) != n_nodes:
        raise RuntimeErrorChecked("membership length mismatch")
    if not np.issubdtype(membership.dtype, np.integer):
        raise RuntimeErrorChecked("membership must be integer")
    membership = membership.astype(np.int32, copy=False)
    if np.any(membership < 0) or np.any(membership >= parts):
        raise RuntimeErrorChecked("membership value outside partition range")
    if set(membership.tolist()) != set(range(parts)):
        raise RuntimeErrorChecked("membership has an empty partition")
    return membership


def copy_input(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    shutil.copy2(source, destination)


def partition_start(base: np.ndarray, ids: np.ndarray, preferred: int, metric: str) -> int:
    if preferred in set(ids.tolist()):
        return int(np.searchsorted(ids, preferred))
    vectors = np.asarray(base[ids], dtype=np.float32)
    centroid = vectors.mean(axis=0)
    if metric == "cosine":
        # Apple Accelerate can emit spurious overflow warnings for a large
        # float32 matrix-vector multiply even when every result is finite.
        # This only selects a deterministic local entry point, so use a small
        # float64 einsum and avoid warning-dependent behavior.
        precise_vectors = vectors.astype(np.float64, copy=False)
        precise_centroid = centroid.astype(np.float64, copy=False)
        denom = np.linalg.norm(precise_vectors, axis=1) * max(float(np.linalg.norm(precise_centroid)), 1e-12)
        dot = np.einsum("ij,j->i", precise_vectors, precise_centroid)
        score = np.divide(dot, denom, out=np.full(len(vectors), -np.inf), where=denom > 0)
        return int(np.argmax(score))
    return int(np.argmin(np.sum((vectors - centroid) ** 2, axis=1)))


def static_metrics(rows: list[np.ndarray], membership: np.ndarray, parts: int) -> dict:
    degree = np.asarray([len(row) for row in rows], dtype=np.int64)
    edge_counts = []
    cross_counts = []
    boundary_counts = []
    adjacency_bytes = []
    for part in range(parts):
        ids = np.flatnonzero(membership == part)
        edge_count = int(sum(len(rows[int(gid)]) for gid in ids))
        cross = int(sum(np.count_nonzero(membership[rows[int(gid)]] != part) for gid in ids))
        boundary = int(sum(bool(np.any(membership[rows[int(gid)]] != part)) for gid in ids))
        edge_counts.append(edge_count)
        cross_counts.append(cross)
        boundary_counts.append(boundary)
        adjacency_bytes.append(int(24 + sum(4 + 4 * len(rows[int(gid)]) for gid in ids)))
    total_edges = int(degree.sum())
    node_counts = [int(np.sum(membership == part)) for part in range(parts)]
    return {
        "nodes": node_counts,
        "node_balance_max_over_mean": float(max(node_counts) / np.mean(node_counts)),
        "directed_edges": edge_counts,
        "adjacency_bytes": adjacency_bytes,
        "cross_directed_edges": cross_counts,
        "cross_ratio": float(sum(cross_counts) / total_edges) if total_edges else 0.0,
        "boundary_nodes": boundary_counts,
        "degree": {"p50": float(np.percentile(degree, 50)), "p95": float(np.percentile(degree, 95)),
                   "p99": float(np.percentile(degree, 99)), "max": int(degree.max(initial=0))},
    }


def inventory(root: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and path_name(item) not in {"RUNTIME_MANIFEST.json", "COMPLETE"}):
        rel = str(path.relative_to(root))
        result[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def path_name(path: Path) -> str:
    return path.name


def seal_runtime(source_root: Path, membership_path: Path, base: Path, query: Path, groundtruth: Path,
                 ghost: Path, output: Path, parts: int, metric: str, link_mode: str, metadata: dict,
                 source_parts: int | None = None) -> dict:
    """``parts`` is the part count of the sealed runtime; ``source_parts`` is the part count of the
    runtime the directed graph is read from (the same as ``parts`` unless given)."""
    if output.exists():
        raise RuntimeErrorChecked(f"refusing to overwrite existing runtime: {output}")
    source_parts = parts if source_parts is None else int(source_parts)
    if parts < 1 or source_parts < 1:
        raise RuntimeErrorChecked("part counts must be positive")
    source_base = np.load(base, mmap_mode="r")
    if source_base.dtype != np.float32 or source_base.ndim != 2:
        raise RuntimeErrorChecked("base must be a float32 two-dimensional npy")
    n_nodes = len(source_base)
    membership = validate_membership(np.load(membership_path, mmap_mode="r"), n_nodes, parts)
    rows, preferred_start = load_global_adjacency(source_root, n_nodes, source_parts)
    source_hash = ordered_graph_hash(rows)
    ghost_ids = np.load(ghost, mmap_mode="r")
    if ghost_ids.ndim != 1 or np.any(ghost_ids < 0) or np.any(ghost_ids >= n_nodes):
        raise RuntimeErrorChecked("ghost IDs are invalid for runtime size")
    for required in (query, groundtruth):
        if not required.is_file():
            raise RuntimeErrorChecked(f"missing required query input: {required}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for source, name in ((base, "base.npy"), (query, "query.npy"), (groundtruth, "groundtruth.npy")):
            copy_input(source, staging / name, link_mode)
        copy_input(ghost, staging / "ghost" / "ghost_indices.npy", link_mode)
        np.save(staging / "membership.npy", membership)
        for part in range(parts):
            part_dir = staging / f"partition_{part}"
            part_dir.mkdir()
            ids = np.flatnonzero(membership == part).astype(np.uint32)
            np.save(part_dir / "local_to_global.npy", ids)
            local_start = partition_start(source_base, ids, preferred_start, metric)
            max_degree = max(len(rows[int(gid)]) for gid in ids)
            write_adj(part_dir / "adj", [rows[int(gid)] for gid in ids], local_start, max_degree)
        runtime_rows, _ = load_global_adjacency(staging, n_nodes, parts)
        runtime_hash = ordered_graph_hash(runtime_rows)
        if runtime_hash != source_hash:
            raise RuntimeErrorChecked("directed graph changed while repartitioning")
        manifest = {
            "schema": {"name": "tide-repartition-runtime", "version": 1},
            "N": n_nodes, "dimension": int(source_base.shape[1]), "parts": parts, "metric": metric,
            "directed_graph_sha256": runtime_hash,
            "membership_sha256": sha256(staging / "membership.npy"),
            "ghost_sha256": sha256(staging / "ghost" / "ghost_indices.npy"),
            "source_runtime": str(source_root.resolve()), "source_graph_sha256": source_hash,
            "source_parts": source_parts,
            "static_metrics": static_metrics(runtime_rows, membership, parts),
            "metadata": metadata,
        }
        manifest["inventory"] = inventory(staging)
        payload = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
        (staging / "RUNTIME_MANIFEST.json").write_bytes(payload)
        (staging / "COMPLETE").write_text(json.dumps({"status": "COMPLETE", "manifest_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True) + "\n")
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
