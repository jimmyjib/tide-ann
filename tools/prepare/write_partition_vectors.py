#!/usr/bin/env python3
"""Write each partition's original vectors (partition_<r>/vectors.npy) for PQ mode.

Checks the pivots and codes against the runtime and spot-checks that the codes match the vectors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.prepare import pq_files

SAMPLE_ROWS = 1024
MIN_CODE_MATCH = 0.99


def write_partition_vectors(runtime: Path, pq_dir: Path) -> dict:
    manifest = json.loads((runtime / "RUNTIME_MANIFEST.json").read_text(encoding="utf-8"))
    n_nodes, dim, parts = int(manifest["N"]), int(manifest["dimension"]), int(manifest["parts"])
    if manifest.get("metric") != "l2":
        raise pq_files.PQFileError("--pq needs an l2 runtime (normalize cosine data first)")
    base = np.load(runtime / "base.npy", mmap_mode="r", allow_pickle=False)
    membership = np.load(runtime / "membership.npy", allow_pickle=False)
    if base.dtype != np.dtype("<f4") or base.shape != (n_nodes, dim):
        raise pq_files.PQFileError(f"base.npy is not float32 {n_nodes} x {dim}")
    if membership.shape != (n_nodes,):
        raise pq_files.PQFileError("membership.npy does not have N rows")
    pair = pq_files.check_pair(pq_dir / "pq_pivots.bin", pq_dir / "pq_codes.bin", n_nodes, dim)
    _, _, centroids = pq_files.read_pivots(pq_dir / "pq_pivots.bin")
    codes = pq_files.read_codes(pq_dir / "pq_codes.bin")

    covered = np.zeros(n_nodes, dtype=bool)
    partitions = []
    for rank in range(parts):
        l2g_path = runtime / f"partition_{rank}" / "local_to_global.npy"
        l2g = np.load(l2g_path, allow_pickle=False).astype(np.int64)
        if l2g.ndim != 1 or (len(l2g) and (l2g.min() < 0 or l2g.max() >= n_nodes)):
            raise pq_files.PQFileError(f"{l2g_path}: ids outside [0, N)")
        if np.any(covered[l2g]) or len(np.unique(l2g)) != len(l2g):
            raise pq_files.PQFileError(f"{l2g_path}: an id is listed twice")
        if np.any(membership[l2g] != rank):
            raise pq_files.PQFileError(f"{l2g_path}: an id is not owned by partition {rank}")
        covered[l2g] = True
        vectors = np.ascontiguousarray(base[l2g], dtype="<f4")
        # spot-check that the codes were built from these vectors
        sample = np.unique((np.arange(min(SAMPLE_ROWS, len(l2g))) * len(l2g)) // max(1, min(SAMPLE_ROWS, len(l2g))))
        if len(sample):
            matched = np.all(pq_files.encode(vectors[sample], centroids) == codes[l2g[sample]], axis=1)
            if matched.mean() < MIN_CODE_MATCH:
                raise pq_files.PQFileError(f"partition {rank}: only {matched.mean():.3f} of the sampled rows encode "
                                           "to their stored code")
        out_dir = pq_dir / f"partition_{rank}"
        try:
            out_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise pq_files.PQFileError(f"{out_dir} already exists") from error
        np.save(out_dir / "vectors.npy", vectors, allow_pickle=False)
        partitions.append({"rank": rank, "rows": int(len(l2g))})
    if not covered.all():
        raise pq_files.PQFileError("the partitions do not cover every node")
    return {"N": n_nodes, "dimension": dim, "parts": parts, "pq": pair, "partitions": partitions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the per-partition original vectors for PQ mode.")
    parser.add_argument("--runtime", type=Path, required=True, help="runtime directory")
    parser.add_argument("--pq-dir", type=Path, required=True, help="directory with the tide-build-pq files")
    args = parser.parse_args(argv)
    record = write_partition_vectors(args.runtime, args.pq_dir)
    print(f"wrote {record['parts']} partition vector files to {args.pq_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
