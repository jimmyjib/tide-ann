#!/usr/bin/env python3
"""Materialize a sustained-load query stream from a native query set.

Running a controller under load needs minutes of saturated serving, far more
queries than any native set holds. This tool concatenates K seeded
shuffles of the native queries into one query.npy (the coordinator loads
query.npy whole and takes a -q prefix, so no serving code changes), with
the ground-truth rows permuted identically so Recall stays measurable on
every pass. Load streams are workload material only: they repeat vectors
by construction, so evaluate recall on the native query set (the manifest's
note says so).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "tide-query-load-stream-v1"
NOTE = (
    "SUSTAINED_LOAD_WORKLOAD_ONLY_NOT_FOR_RECALL_EVALUATION"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    queries = np.load(args.query)
    groundtruth = np.load(args.groundtruth)
    if queries.ndim != 2 or groundtruth.ndim != 2:
        raise SystemExit("query and groundtruth must be 2-D arrays")
    if len(queries) != len(groundtruth):
        raise SystemExit(
            f"row mismatch: {len(queries)} queries vs "
            f"{len(groundtruth)} groundtruth rows")

    n = len(queries)
    out_queries = np.empty((n * args.repeats, queries.shape[1]),
                           dtype=queries.dtype)
    out_groundtruth = np.empty((n * args.repeats, groundtruth.shape[1]),
                               dtype=groundtruth.dtype)
    permutations = []
    for ordinal in range(args.repeats):
        rng = np.random.Generator(np.random.PCG64(args.seed + ordinal))
        perm = rng.permutation(n)
        permutations.append(hashlib.sha256(
            perm.astype("<u4").tobytes()).hexdigest())
        out_queries[ordinal * n:(ordinal + 1) * n] = queries[perm]
        out_groundtruth[ordinal * n:(ordinal + 1) * n] = groundtruth[perm]

    args.output.mkdir(parents=True)
    np.save(args.output / "query.npy", out_queries)
    np.save(args.output / "groundtruth.npy", out_groundtruth)

    manifest = {
        "schema": SCHEMA,
        "note": NOTE,
        "label": args.label,
        "seed": args.seed,
        "repeats": args.repeats,
        "native_rows": int(n),
        "stream_rows": int(n * args.repeats),
        "source": {
            "query": {"path": str(args.query),
                      "sha256": sha256_path(args.query)},
            "groundtruth": {"path": str(args.groundtruth),
                            "sha256": sha256_path(args.groundtruth)},
        },
        "per_pass_permutation_sha256": permutations,
        "output": {
            "query": {"path": str(args.output / "query.npy"),
                      "sha256": sha256_path(args.output / "query.npy")},
            "groundtruth": {
                "path": str(args.output / "groundtruth.npy"),
                "sha256": sha256_path(args.output / "groundtruth.npy")},
        },
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"label": args.label, "stream_rows": int(n * args.repeats),
                      "repeats": args.repeats,
                      "bytes": int(out_queries.nbytes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
