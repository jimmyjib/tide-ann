"""Synthetic sealed-runtime fixtures and reference oracles for the tools.prepare unit tests.

Two reference oracles live here: ``build_undirected_csr`` / ``edge_cut`` (scipy) and
``verify_sealed_runtime``.

The directory layout mirrors ``runtime_lib.seal_runtime`` closely enough for ``load_global_adjacency``:
``RUNTIME_MANIFEST.json``, ``base.npy``, ``membership.npy``, ``ghost/ghost_indices.npy`` and
``partition_<k>/{adj,local_to_global.npy}``.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from tools.prepare.runtime_lib import (RuntimeErrorChecked, inventory, load_global_adjacency, ordered_graph_hash,
                                       sha256, validate_membership, write_adj)


def random_rows(n: int, rng: np.random.Generator, low: int = 3, high: int = 7) -> list[np.ndarray]:
    """Directed rows with unique neighbours and no self loops (Vamana-like)."""
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


def write_global_graph(path: Path, rows: list[np.ndarray], start: int, max_degree: int) -> None:
    """One global DiskANN graph file as ``partition_graph.read_global_graph`` expects."""
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
    manifest = {"schema": {"name": "tide-repartition-runtime", "version": 1}, "N": n, "dimension": dim,
                "parts": parts, "metric": "l2"}
    (root / "RUNTIME_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"rows": rows, "base": base, "membership": membership.astype(np.int64), "ghost": ghost, "rng": rng,
            "max_degree": max_degree}


def brute_force_groundtruth(base: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    d2 = ((queries[:, None, :].astype(np.float64) - base[None, :, :].astype(np.float64)) ** 2).sum(axis=2)
    return np.argsort(d2, axis=1, kind="stable")[:, :k].astype(np.int32)


def build_undirected_csr(rows: list[np.ndarray], n_nodes: int):
    """scipy reference of the symmetrized graph (oracle for ``graph_csr.symmetrize_csr``)."""
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


def edge_cut(U, membership: np.ndarray) -> int:
    """Undirected edges whose endpoints have different owners (oracle for ``undirected_edge_cuts``)."""
    src = np.repeat(np.arange(len(membership)), np.diff(U.indptr))
    return int((membership[U.indices] != membership[src]).sum() // 2)


def verify_sealed_runtime(root: Path) -> dict:
    """Fail-closed check of a sealed runtime (oracle for ``seal_runtime``)."""
    manifest_path = root / "RUNTIME_MANIFEST.json"
    complete_path = root / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise RuntimeErrorChecked("runtime is not sealed")
    payload = manifest_path.read_bytes()
    if json.loads(complete_path.read_text()) != {"status": "COMPLETE",
                                                 "manifest_sha256": hashlib.sha256(payload).hexdigest()}:
        raise RuntimeErrorChecked("COMPLETE does not bind manifest")
    manifest = json.loads(payload)
    if manifest.get("schema") != {"name": "tide-repartition-runtime", "version": 1}:
        raise RuntimeErrorChecked("unexpected schema")
    if inventory(root) != manifest.get("inventory"):
        raise RuntimeErrorChecked("file inventory/hash mismatch")
    base = np.load(root / "base.npy", mmap_mode="r")
    if base.dtype != np.float32 or base.ndim != 2 or len(base) != manifest["N"]:
        raise RuntimeErrorChecked("base contract mismatch")
    membership = validate_membership(np.load(root / "membership.npy", mmap_mode="r"), len(base), manifest["parts"])
    rows, _ = load_global_adjacency(root, len(base), manifest["parts"])
    graph_hash = ordered_graph_hash(rows)
    if graph_hash != manifest["directed_graph_sha256"]:
        raise RuntimeErrorChecked("directed graph hash mismatch")
    if sha256(root / "membership.npy") != manifest["membership_sha256"]:
        raise RuntimeErrorChecked("membership hash mismatch")
    ghost = np.load(root / "ghost" / "ghost_indices.npy", mmap_mode="r")
    if ghost.ndim != 1 or np.any(ghost < 0) or np.any(ghost >= len(base)):
        raise RuntimeErrorChecked("invalid ghost IDs")
    return {"status": "VERIFIED", "N": int(len(base)), "directed_graph_sha256": graph_hash,
            "node_counts": [int(np.sum(membership == part)) for part in range(manifest["parts"])]}
