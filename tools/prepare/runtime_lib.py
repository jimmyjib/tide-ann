"""Read, write and check partitioned runtimes. A partition's adj file stores global neighbor ids."""
from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np


class PrepareError(RuntimeError):
    """Invalid input or output. The command-line tools exit with status 2."""


def read_adj(path: Path, expected_rows: int | None = None) -> tuple[list[np.ndarray], int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise PrepareError(f"short adjacency header: {path}")
        file_size, max_degree, start_node, _ = struct.unpack("<QIIQ", header)
        if file_size != path.stat().st_size:
            raise PrepareError(f"adjacency size header mismatch: {path}")
        rows: list[np.ndarray] = []
        while handle.tell() < file_size:
            raw_degree = handle.read(4)
            if len(raw_degree) != 4:
                raise PrepareError(f"truncated degree: {path}")
            degree, = struct.unpack("<I", raw_degree)
            if degree > max_degree:
                raise PrepareError(f"degree exceeds header: {path}")
            raw = handle.read(degree * 4)
            if len(raw) != degree * 4:
                raise PrepareError(f"truncated neighbors: {path}")
            rows.append(np.frombuffer(raw, dtype="<u4").copy().astype(np.uint32))
        if handle.tell() != file_size:
            raise PrepareError(f"adjacency parse overrun: {path}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise PrepareError(f"row count mismatch in {path}: {len(rows)} != {expected_rows}")
    if rows and start_node >= len(rows):
        raise PrepareError(f"start node outside partition: {path}")
    return rows, int(start_node), int(max_degree)


def load_global_adjacency(root: Path, n_nodes: int, parts: int) -> tuple[list[np.ndarray], int]:
    global_rows: list[np.ndarray | None] = [None] * n_nodes
    global_start: int | None = None
    for part in range(parts):
        directory = root / f"partition_{part}"
        l2g_path = directory / "local_to_global.npy"
        adj_path = directory / "adj"
        if not l2g_path.is_file() or not adj_path.is_file():
            raise PrepareError(f"missing partition payload: {directory}")
        l2g = np.load(l2g_path, mmap_mode="r")
        if l2g.ndim != 1 or not np.issubdtype(l2g.dtype, np.integer):
            raise PrepareError(f"invalid local_to_global: {l2g_path}")
        if len(l2g) == 0 or np.any(l2g < 0) or np.any(l2g >= n_nodes) or len(np.unique(l2g)) != len(l2g):
            raise PrepareError(f"invalid local_to_global IDs: {l2g_path}")
        rows, start, _ = read_adj(adj_path, len(l2g))
        for local, global_id in enumerate(l2g):
            gid = int(global_id)
            if global_rows[gid] is not None:
                raise PrepareError(f"duplicate global row {gid}")
            row = rows[local]
            if np.any(row >= n_nodes):
                raise PrepareError(f"neighbor out of range at node {gid}")
            global_rows[gid] = row
        candidate = int(l2g[start])
        if global_start is None:
            global_start = candidate
    if any(row is None for row in global_rows):
        raise PrepareError("partition rows do not cover all global nodes")
    return [row for row in global_rows if row is not None], int(global_start or 0)


def validate_membership(membership: np.ndarray, n_nodes: int, parts: int) -> np.ndarray:
    if membership.ndim != 1 or len(membership) != n_nodes:
        raise PrepareError("membership length mismatch")
    if not np.issubdtype(membership.dtype, np.integer):
        raise PrepareError("membership must be integer")
    membership = membership.astype(np.int32, copy=False)
    if np.any(membership < 0) or np.any(membership >= parts):
        raise PrepareError("membership value outside partition range")
    if set(membership.tolist()) != set(range(parts)):
        raise PrepareError("membership has an empty partition")
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
        # float64 einsum: Accelerate's float32 matvec can raise spurious overflow warnings
        precise_vectors = vectors.astype(np.float64, copy=False)
        precise_centroid = centroid.astype(np.float64, copy=False)
        denom = np.linalg.norm(precise_vectors, axis=1) * max(float(np.linalg.norm(precise_centroid)), 1e-12)
        dot = np.einsum("ij,j->i", precise_vectors, precise_centroid)
        score = np.divide(dot, denom, out=np.full(len(vectors), -np.inf), where=denom > 0)
        return int(np.argmax(score))
    return int(np.argmin(np.sum((vectors - centroid) ** 2, axis=1)))


def write_runtime(source_root: Path, membership_path: Path, base: Path, query: Path, groundtruth: Path,
                  ghost: Path, output: Path, parts: int, metric: str, link_mode: str,
                  source_parts: int | None = None) -> dict:
    """Write a runtime of `parts` partitions. The graph is read from a runtime of `source_parts` partitions."""
    from tools.prepare.graph_csr import csr_static_metrics, write_adj_csr  # graph_csr imports this module

    if output.exists():
        raise PrepareError(f"runtime already exists: {output}")
    source_parts = parts if source_parts is None else int(source_parts)
    if parts < 1 or source_parts < 1:
        raise PrepareError("part counts must be positive")
    source_base = np.load(base, mmap_mode="r")
    if source_base.dtype != np.float32 or source_base.ndim != 2:
        raise PrepareError("base must be a float32 two-dimensional npy")
    n_nodes = len(source_base)
    membership = validate_membership(np.load(membership_path, mmap_mode="r"), n_nodes, parts)
    rows, preferred_start = load_global_adjacency(source_root, n_nodes, source_parts)
    indptr = np.zeros(n_nodes + 1, dtype=np.uint64)
    np.cumsum([len(row) for row in rows], dtype=np.uint64, out=indptr[1:])
    indices = np.concatenate(rows).astype(np.uint32, copy=False)
    del rows
    degrees = np.diff(indptr)
    ghost_ids = np.load(ghost, mmap_mode="r")
    if ghost_ids.ndim != 1 or np.any(ghost_ids < 0) or np.any(ghost_ids >= n_nodes):
        raise PrepareError("ghost IDs are invalid for runtime size")
    for required in (query, groundtruth):
        if not required.is_file():
            raise PrepareError(f"missing required query input: {required}")

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
            write_adj_csr(part_dir / "adj", indptr, indices, ids, local_start, int(degrees[ids].max()))
        manifest = {
            "N": n_nodes, "dimension": int(source_base.shape[1]), "parts": parts, "source_parts": source_parts,
            "metric": metric, "static_metrics": csr_static_metrics(indptr, indices, membership, parts),
        }
        (staging / "RUNTIME_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
