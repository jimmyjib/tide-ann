#!/usr/bin/env python3
"""Pick an insert batch from a pool of vectors: the rows nearest to one server's vectors.

A pool row whose nearest existing vector is on --server comes first. The other rows follow by how much farther the
server's nearest vector is than the overall nearest one, ties by pool row. The batch keeps pool order. Also writes
the existing vectors followed by the batch (insert row i becomes id N + i, so row r is id r) and prints a JSON
summary line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick the pool rows nearest to one server as an insert batch.")
    parser.add_argument("--points", type=Path, action="append", required=True, help="existing vectors in id order")
    parser.add_argument("--owners", type=Path, required=True, help="owner of each existing id (.npy)")
    parser.add_argument("--server", type=int, required=True)
    parser.add_argument("--servers", type=int, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--first-row", type=int, required=True)
    parser.add_argument("--pool-rows", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-plus-inserts", type=Path, required=True)
    args = parser.parse_args(argv)
    # np.save appends .npy to a name without it
    targets = [path if path.name.endswith(".npy") else path.with_name(path.name + ".npy")
               for path in (args.output, args.base_plus_inserts)]
    present = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if present:
        raise SystemExit(f"output already exists: {', '.join(present)}")
    pool = np.load(args.pool)
    end = args.first_row + args.pool_rows
    if args.first_row < 0 or not 1 <= args.rows <= args.pool_rows or end > len(pool):
        raise SystemExit(f"bad --first-row/--pool-rows/--rows for a pool of {len(pool)} rows")
    points = np.ascontiguousarray(np.concatenate([np.load(path) for path in args.points]), dtype=np.float32)
    owners = np.load(args.owners).astype(np.int64)
    if owners.shape != (len(points),) or not 0 <= args.server < args.servers or owners.max() >= args.servers:
        raise SystemExit("--owners does not match --points, or --server is out of range")
    candidates = np.ascontiguousarray(pool[args.first_row:end], dtype=np.float32)
    # squared L2 without the candidate's own norm (gap = 0 when the nearest vector is on the server)
    norms = np.einsum("ij,ij->i", points, points)
    on_server = owners == args.server
    nearest_owner = np.empty(len(candidates), dtype=np.int64)
    gap = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), 512):
        block = norms[None, :] - 2.0 * (candidates[start:start + 512] @ points.T)
        nearest = block.argmin(axis=1)
        nearest_owner[start:start + 512] = owners[nearest]
        gap[start:start + 512] = block[:, on_server].min(axis=1) - block[np.arange(len(block)), nearest]
    # rows in the server's part first, then the nearest to it (ties by pool row), kept in pool order
    take = np.sort(np.lexsort((np.arange(len(candidates)), gap))[:args.rows])
    rows = candidates[take]
    # insert row i becomes id N + i, so row r of base_plus_inserts is id r
    combined = np.concatenate([points, rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.base_plus_inserts.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, rows)
    np.save(args.base_plus_inserts, np.ascontiguousarray(combined, dtype=np.float32))
    print(json.dumps({"batch_rows": int(len(rows)), "base_plus_inserts_rows": int(len(combined)),
                      "pool_rows_on_server": int(np.sum(nearest_owner == args.server)),
                      "nearest_on_server": int(np.sum(nearest_owner[take] == args.server)),
                      "nearest_owner_counts": np.bincount(nearest_owner[take], minlength=args.servers).tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
