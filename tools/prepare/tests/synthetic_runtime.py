"""Small runtimes, row-list references and scipy reference checks for the tools.prepare tests."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from tools.prepare.runtime_lib import PrepareError, load_global_adjacency, validate_membership

RUNTIME_KEYS = {"N", "dimension", "parts", "source_parts", "metric", "static_metrics"}


def random_rows(n: int, rng: np.random.Generator, low: int = 3, high: int = 7) -> list[np.ndarray]:
    """Directed rows with unique neighbors and no self loops (Vamana-like)."""
    rows = []
    for node in range(n):
        degree = int(rng.integers(low, high + 1))
        others = np.setdiff1d(np.arange(n), [node])
        rows.append(np.sort(rng.choice(others, size=min(degree, len(others)), replace=False)).astype(np.uint32))
    return rows


def balanced_membership(n: int, rng: np.random.Generator, parts: int = 4) -> np.ndarray:
    membership = np.arange(n) % parts
    rng.shuffle(membership)
    return membership.astype(np.int32)


def write_adj(path: Path, rows: Iterable[np.ndarray], start_node: int, max_degree: int) -> int:
    """Row-list reference for graph_csr.write_adj_csr."""
    normalized = [np.asarray(row, dtype="<u4") for row in rows]
    if not normalized:
        raise PrepareError("cannot write empty partition")
    if not 0 <= start_node < len(normalized):
        raise PrepareError("invalid partition start node")
    size = 24 + sum(4 + row.nbytes for row in normalized)
    with path.open("wb") as handle:
        handle.write(struct.pack("<QIIQ", size, int(max_degree), int(start_node), 0))
        for row in normalized:
            handle.write(struct.pack("<I", len(row)))
            handle.write(row.tobytes())
    return size


def static_metrics(rows: list[np.ndarray], membership: np.ndarray, parts: int) -> dict:
    """Row-list reference for graph_csr.csr_static_metrics."""
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


def write_global_graph(path: Path, rows: list[np.ndarray], start: int, max_degree: int) -> None:
    """One global DiskANN graph file, the layout ``graph_csr.read_global_graph_csr`` reads."""
    body = bytearray()
    for row in rows:
        body.extend(struct.pack("<I", len(row)))
        body.extend(np.asarray(row, dtype="<u4").tobytes())
    path.write_bytes(struct.pack("<QIIQ", 24 + len(body), int(max_degree), int(start), 0) + bytes(body))


def make_runtime(root: Path, n: int = 96, dim: int = 6, seed: int = 7, parts: int = 4, ghost_count: int = 8,
                 rows: list[np.ndarray] | None = None, membership: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if rows is None:
        rows = random_rows(n, rng)
    if membership is None:
        membership = balanced_membership(n, rng, parts)
    base = rng.normal(size=(n, dim)).astype(np.float32)
    ghost = np.sort(rng.choice(n, size=ghost_count, replace=False)).astype(np.int64)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "base.npy", base)
    np.save(root / "membership.npy", membership.astype(np.int32))
    (root / "ghost").mkdir()
    np.save(root / "ghost" / "ghost_indices.npy", ghost)
    max_degree = max(len(r) for r in rows)
    for part in range(parts):
        directory = root / f"partition_{part}"
        directory.mkdir()
        ids = np.flatnonzero(membership == part).astype(np.uint32)
        np.save(directory / "local_to_global.npy", ids)
        write_adj(directory / "adj", [rows[int(g)] for g in ids], 0, max_degree)
    manifest = {"N": n, "dimension": dim, "parts": parts, "metric": "l2"}
    (root / "RUNTIME_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"rows": rows, "base": base, "membership": membership.astype(np.int64), "ghost": ghost, "rng": rng,
            "max_degree": max_degree}


def brute_force_groundtruth(base: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    d2 = ((queries[:, None, :].astype(np.float64) - base[None, :, :].astype(np.float64)) ** 2).sum(axis=2)
    return np.argsort(d2, axis=1, kind="stable")[:, :k].astype(np.int32)


def build_undirected_csr(rows: list[np.ndarray], n_nodes: int):
    """scipy version of the symmetrized graph (reference for graph_csr.symmetrize_csr)."""
    import scipy.sparse as sp
    deg = np.fromiter((len(r) for r in rows), dtype=np.int64, count=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg)
    indices = np.concatenate(rows).astype(np.int64) if deg.sum() else np.zeros(0, dtype=np.int64)
    A = sp.csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr), shape=(n_nodes, n_nodes))
    U = (A + A.T)
    U.data[:] = 1
    U = U.tocsr()
    U.sort_indices()
    return U


def verify_runtime(root: Path, rows: list[np.ndarray]) -> dict:
    """Check a runtime written by write_runtime against the source graph rows. Raises PrepareError."""
    manifest = json.loads((root / "RUNTIME_MANIFEST.json").read_text())
    if set(manifest) != RUNTIME_KEYS:
        raise PrepareError(f"unexpected manifest keys {sorted(manifest)}")
    base = np.load(root / "base.npy", mmap_mode="r")
    if base.dtype != np.float32 or base.shape != (manifest["N"], manifest["dimension"]):
        raise PrepareError("base shape or dtype mismatch")
    membership = validate_membership(np.load(root / "membership.npy", mmap_mode="r"), len(base), manifest["parts"])
    runtime_rows, _ = load_global_adjacency(root, len(base), manifest["parts"])
    if len(runtime_rows) != len(rows) or any(not np.array_equal(a, b) for a, b in zip(runtime_rows, rows)):
        raise PrepareError("runtime graph differs from the source graph")
    ghost = np.load(root / "ghost" / "ghost_indices.npy", mmap_mode="r")
    if ghost.ndim != 1 or np.any(ghost < 0) or np.any(ghost >= len(base)):
        raise PrepareError("invalid head ids")
    return {"N": int(len(base)), "node_counts": [int(np.sum(membership == part)) for part in range(manifest["parts"])]}
