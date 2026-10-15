#!/usr/bin/env python3
"""Write the queries of each given cluster to <output>/query-hot-<cluster>.npy.

Reads query.npy and query_labels.npy of a make_synthetic dataset and prints the query count of each cluster as
one JSON line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.prepare.make_synthetic import parse_cluster_list


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write one query file per cluster (query-hot-<cluster>.npy).")
    parser.add_argument("--source", type=Path, required=True,
                        help="make_synthetic output (query.npy, query_labels.npy)")
    parser.add_argument("--hot-clusters", type=parse_cluster_list, required=True,
                        help="comma-separated cluster ids")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    clusters = list(dict.fromkeys(args.hot_clusters))
    present = [str(path) for path in (args.output / f"query-hot-{c}.npy" for c in clusters)
               if path.exists() or path.is_symlink()]
    if present:
        raise SystemExit(f"output already exists: {', '.join(present)}")
    query = np.load(args.source / "query.npy")
    labels = np.load(args.source / "query_labels.npy")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cluster in clusters:
        rows = np.ascontiguousarray(query[labels == cluster], dtype=np.float32)
        if not len(rows):
            raise SystemExit(f"no query comes from hot cluster {cluster}")
        np.save(args.output / f"query-hot-{cluster}.npy", rows)
        counts[str(cluster)] = int(len(rows))
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
