#!/usr/bin/env python3
"""Concatenate N shuffled copies of a query set and its ground truth into one long query stream.

Used as search load for a controller phase. The stream repeats queries, so measure recall on the original set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a query stream from shuffled copies of a query set.")
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        raise SystemExit(f"row mismatch: {len(queries)} queries vs {len(groundtruth)} groundtruth rows")

    n = len(queries)
    out_queries = np.empty((n * args.repeats, queries.shape[1]), dtype=queries.dtype)
    out_groundtruth = np.empty((n * args.repeats, groundtruth.shape[1]), dtype=groundtruth.dtype)
    for ordinal in range(args.repeats):
        rng = np.random.Generator(np.random.PCG64(args.seed + ordinal))
        perm = rng.permutation(n)
        out_queries[ordinal * n:(ordinal + 1) * n] = queries[perm]
        out_groundtruth[ordinal * n:(ordinal + 1) * n] = groundtruth[perm]

    args.output.mkdir(parents=True)
    np.save(args.output / "query.npy", out_queries)
    np.save(args.output / "groundtruth.npy", out_groundtruth)

    manifest = {"seed": args.seed, "repeats": args.repeats, "query_rows": int(n),
                "stream_rows": int(n * args.repeats)}
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} ({n * args.repeats} queries, {args.repeats} passes over {n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
