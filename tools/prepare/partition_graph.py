#!/usr/bin/env python3
"""METIS-partition a global directed DiskANN graph without changing its rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from tools.prepare.graph_csr import (CsrScaleError, csr_ordered_graph_hash, csr_static_metrics,
                                            metis_int32_arrays, read_global_graph_csr, symmetrize_csr,
                                            write_adj_csr)
from tools.prepare.runtime_lib import RuntimeErrorChecked, ordered_graph_hash, static_metrics, write_adj


def read_global_graph(path: Path, n_nodes: int) -> tuple[list[np.ndarray], int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise RuntimeErrorChecked("short graph header")
        file_size, max_degree, start_node, _ = struct.unpack("<QIIQ", header)
        if file_size != path.stat().st_size or start_node >= n_nodes:
            raise RuntimeErrorChecked("invalid graph header")
        rows = []
        for _ in range(n_nodes):
            raw = handle.read(4)
            if len(raw) != 4:
                raise RuntimeErrorChecked("truncated graph degree")
            degree, = struct.unpack("<I", raw)
            if degree > max_degree:
                raise RuntimeErrorChecked("degree exceeds graph max")
            neighbours = np.frombuffer(handle.read(degree * 4), dtype="<u4").copy()
            if len(neighbours) != degree or np.any(neighbours >= n_nodes):
                raise RuntimeErrorChecked("invalid graph neighbour")
            rows.append(neighbours)
        if handle.read(1):
            raise RuntimeErrorChecked("unexpected graph trailing data")
    return rows, int(start_node), int(max_degree)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=4, help="partitions, one per index server (2..64)")
    parser.add_argument("--ghost-ratio", type=float, default=0.01)
    parser.add_argument("--ghost-seed", type=int, default=42)
    parser.add_argument("--csr", action="store_true",
                        help="stream the graph into flat CSR arrays, symmetrize it in chunks (graph_csr) and "
                             "call pymetis with xadj/adjncy; same result as the default row-list path, sized for "
                             "10M-node graphs (refuses >= 2^31 undirected entries: 64-bit METIS fallback)")
    args = parser.parse_args(argv)
    # One part per index server: 2..64 servers (the repair controller's owner range).
    if not 2 <= args.parts <= 64 or not 0 < args.ghost_ratio <= 1:
        raise RuntimeErrorChecked("partitioning requires 2..64 parts and a valid ghost ratio")
    if args.output.exists():
        raise RuntimeErrorChecked("refusing to overwrite output")
    base = np.load(args.base, mmap_mode="r")
    if base.dtype != np.float32 or base.ndim != 2:
        raise RuntimeErrorChecked("invalid base")
    try:
        import pymetis
    except ImportError as error:
        raise RuntimeErrorChecked("pymetis is required (pip install pymetis)") from error
    if args.csr:
        indptr, indices, start, max_degree = read_global_graph_csr(args.graph, len(base))
        uindptr, uindices = symmetrize_csr(indptr, indices)
        try:
            xadj, adjncy, _, _ = metis_int32_arrays(uindptr, uindices)
        except CsrScaleError as error:
            raise RuntimeErrorChecked(str(error)) from error
        cuts, raw_membership = pymetis.part_graph(args.parts, xadj=xadj, adjncy=adjncy)
        del uindptr, uindices, xadj, adjncy
        rows = None
    else:
        rows, start, max_degree = read_global_graph(args.graph, len(base))
        undirected = [set(map(int, row)) for row in rows]
        for source, row in enumerate(rows):
            for target in row:
                undirected[int(target)].add(source)
        cuts, raw_membership = pymetis.part_graph(args.parts, adjacency=[sorted(row) for row in undirected])
    membership = np.asarray(raw_membership, dtype=np.int32)
    if membership.shape != (len(base),) or set(membership.tolist()) != set(range(args.parts)):
        raise RuntimeErrorChecked("METIS membership contract failure")
    args.output.mkdir(parents=True)
    np.save(args.output / "membership.npy", membership)
    for part in range(args.parts):
        directory = args.output / f"partition_{part}"; directory.mkdir()
        ids = np.flatnonzero(membership == part).astype(np.uint32)
        np.save(directory / "local_to_global.npy", ids)
        local_start = int(np.searchsorted(ids, start)) if membership[start] == part else 0
        if args.csr:
            write_adj_csr(directory / "adj", indptr, indices, ids, local_start, max_degree)
        else:
            write_adj(directory / "adj", [rows[int(gid)] for gid in ids], local_start, max_degree)
    rng = np.random.RandomState(args.ghost_seed)
    ghost = np.concatenate([rng.choice(np.flatnonzero(membership == part),
                                       max(1, int(np.sum(membership == part) * args.ghost_ratio)), replace=False)
                            for part in range(args.parts)]).astype(np.int64)
    (args.output / "ghost").mkdir()
    np.save(args.output / "ghost" / "ghost_indices.npy", np.sort(ghost))
    if args.csr:
        graph_sha256 = csr_ordered_graph_hash(indptr, indices)
        metrics = csr_static_metrics(indptr, indices, membership, args.parts)
    else:
        graph_sha256 = ordered_graph_hash(rows)
        metrics = static_metrics(rows, membership, args.parts)
    manifest = {"schema": {"name": "tide-repartition-partition", "version": 1}, "N": int(len(base)),
                "parts": args.parts, "metis_edge_cuts": int(cuts), "ghost_ratio": args.ghost_ratio,
                "ghost_seed": args.ghost_seed, "directed_graph_sha256": graph_sha256,
                "static_metrics": metrics}
    (args.output / "PARTITION_BUILD.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "cuts": int(cuts),
                      "adjacency": "csr" if args.csr else "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeErrorChecked as error:
        print(f"GRAPH PARTITION FAILED: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
