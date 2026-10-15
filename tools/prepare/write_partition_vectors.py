#!/usr/bin/env python3
"""Per-partition original vectors and the PQ manifest of a sealed runtime (own-only storage).

Hybrid PQ own-only storage (server ``--pq``) loads, for server r, ``membership.npy``,
``partition_r/{adj,local_to_global.npy}`` from the runtime and, from a PQ directory,
``pq_pivots.bin``, ``pq_codes.bin`` (both written by ``tide-build-pq --codes`` from the runtime's
``base.npy``) and ``partition_r/vectors.npy``: row i is the original vector of
``local_to_global[i]``. This tool writes those vector files into the PQ directory, checks the PQ
pair against the runtime (``pq_files.check_pair``: rows = N, dimension, M, the codes' pivot hash),
encodes a sample of every partition's rows to confirm the codes belong to these vectors, and seals
the directory with ``PQ_MANIFEST.json`` (schema ``release-pq-storage`` v1: the runtime manifest's
sha256, the PQ parameters, and sha256 + bytes of every file a server reads from it).

Run with the directory that holds the ``tools`` package on ``PYTHONPATH`` as ``python -m tools.prepare.write_partition_vectors``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools.prepare import pq_files

SCHEMA = {"name": "release-pq-storage", "version": 1}
SAMPLE_ROWS = 1024
MIN_CODE_MATCH = 0.99


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_npy_fsync(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def write_partition_vectors(runtime: Path, pq_dir: Path) -> dict:
    manifest_path = runtime / "RUNTIME_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_nodes, dim, parts = int(manifest["N"]), int(manifest["dimension"]), int(manifest["parts"])
    if manifest.get("metric") != "l2":
        raise pq_files.PQFileError("own-only storage serves l2 runtimes (normalise cosine data first)")
    if (pq_dir / "PQ_MANIFEST.json").exists():
        raise pq_files.PQFileError(f"{pq_dir} is already sealed")
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
        # The codes belong to these vectors: a deterministic sample encodes to its stored code.
        sample = np.unique((np.arange(min(SAMPLE_ROWS, len(l2g))) * len(l2g)) // max(1, min(SAMPLE_ROWS, len(l2g))))
        if len(sample):
            matched = np.all(pq_files.encode(vectors[sample], centroids) == codes[l2g[sample]], axis=1)
            if matched.mean() < MIN_CODE_MATCH:
                raise pq_files.PQFileError(f"partition {rank}: only {matched.mean():.3f} of the sampled rows encode "
                                           "to their stored code")
        out_dir = pq_dir / f"partition_{rank}"
        out_dir.mkdir(parents=False, exist_ok=False)
        write_npy_fsync(out_dir / "vectors.npy", vectors)
        partitions.append({"rank": rank, "rows": int(len(l2g)),
                           "local_to_global_sha256": sha256_file(l2g_path),
                           "sampled_rows": int(len(sample))})
    if not covered.all():
        raise pq_files.PQFileError("the partitions do not cover every node")

    inventory = {}
    for relative in ["pq_pivots.bin", "pq_codes.bin"] + [f"partition_{r}/vectors.npy" for r in range(parts)]:
        path = pq_dir / relative
        inventory[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    record = {"schema": SCHEMA, "runtime_manifest_sha256": sha256_file(manifest_path),
              "membership_sha256": sha256_file(runtime / "membership.npy"), "N": n_nodes, "dimension": dim,
              "parts": parts, "pq": pair, "partitions": partitions, "inventory": inventory,
              "vector_bytes_per_server_mean": int(n_nodes * dim * 4 / parts),
              "code_bytes": n_nodes * pair["M"]}
    manifest_out = pq_dir / "PQ_MANIFEST.json"
    with manifest_out.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", type=Path, required=True, help="sealed runtime (base.npy, membership, partitions)")
    parser.add_argument("--pq-dir", type=Path, required=True, help="directory holding tide-build-pq's pivots and codes")
    args = parser.parse_args(argv)
    record = write_partition_vectors(args.runtime, args.pq_dir)
    print(json.dumps({"status": "COMPLETE", "pq_dir": str(args.pq_dir), "pq": record["pq"],
                      "partitions": record["partitions"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
