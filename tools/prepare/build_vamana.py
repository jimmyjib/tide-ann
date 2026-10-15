#!/usr/bin/env python3
"""Build the Vamana graph (R 64, L 100, alpha 1.2) of a float32 .npy with diskannpy."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("l2", "cosine"), required=True)
    parser.add_argument("--prefix", default="vamana")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    if not args.input.is_file():
        raise ValueError(f"--input {args.input} is not a file")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty")
    base = np.load(args.input, mmap_mode="r")
    if base.dtype != np.float32 or base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError("input must be finite two-dimensional float32")
    if args.metric == "cosine" and np.any(np.linalg.norm(base, axis=1) == 0):
        raise ValueError("cosine input contains zero-norm vector")
    args.output.mkdir(parents=True, exist_ok=True)
    import diskannpy
    diskannpy.build_memory_index(data=base, distance_metric=args.metric, index_directory=str(args.output),
                                 complexity=100, graph_degree=64, alpha=1.2, num_threads=args.threads,
                                 use_pq_build=False, index_prefix=args.prefix)
    if not any(path.is_file() for path in args.output.iterdir()):
        raise RuntimeError("diskannpy produced no graph files")
    (args.output / "GRAPH_BUILD.json").write_text(json.dumps({
        "metric": args.metric, "N": int(base.shape[0]), "dimension": int(base.shape[1]),
        "R": 64, "L_build": 100, "alpha": 1.2, "prefix": args.prefix, "threads": args.threads,
    }, sort_keys=True, indent=2) + "\n")
    print(f"wrote {args.output}/{args.prefix} ({base.shape[0]} nodes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
