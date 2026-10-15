#!/usr/bin/env python3
"""Turn source/ and index/vamana under --dataset-root into a K-server runtime (K = --parts).

Runs the partition, head id, ground truth, runtime and query stream steps in order.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent


def build_steps(root: Path, python: str, *, n_nodes: int, parts: int = 4, stream_repeats: int = 64,
                stream_seed: int = 2, stream_mode: str = "shuffle", stream_rows: int = 262144,
                hot_share: float = 0.7, regions: int = 32, hot_regions: int = 3,
                tools: Path = TOOLS) -> list[tuple[str, list[str]]]:
    src, index = root / "source", root / "index"
    gt_args = ["--metric", "l2", "--k", "10"]
    if stream_mode == "shuffle":
        stream = [python, "-B", tools / "build_query_load_stream.py", "--query", root / "runtime" / "query.npy",
                  "--groundtruth", root / "runtime" / "groundtruth.npy", "--repeats", str(stream_repeats),
                  "--seed", str(stream_seed), "--output", root / "stream"]
    elif stream_mode == "skewed":
        stream = [python, "-B", tools / "build_skewed_query_stream.py", "--query", root / "runtime" / "query.npy",
                  "--groundtruth", root / "runtime" / "groundtruth.npy",
                  "--membership", root / "runtime" / "membership.npy", "--rows", str(stream_rows),
                  "--hot-share", str(hot_share), "--regions", str(regions), "--hot-regions", str(hot_regions),
                  "--seed", str(stream_seed), "--output", root / "stream"]
    else:
        raise ValueError(f"unknown stream mode {stream_mode}")
    steps = [
        ("partition", [python, "-B", tools / "partition_graph.py", "--graph", index / "vamana",
                       "--base", src / "base.npy", "--output", root / "partition", "--parts", str(parts)]),
        ("ghost", [python, "-B", tools / "make_ghost_ids.py", "--nodes", str(n_nodes),
                   "--output", root / "ghost" / "ghost_indices.npy", "--ratio", "0.01", "--seed", "42"]),
        ("gt-base", [python, tools / "generate_groundtruth.py", "--base", src / "base.npy",
                     "--query", src / "query.npy", "--output", root / "gt" / "base" / "groundtruth.npy",
                     *gt_args]),
        ("gt-base-plus-insert", [python, tools / "generate_groundtruth.py",
                                 "--base", src / "base_plus_insert.npy", "--query", src / "query.npy",
                                 "--output", root / "gt" / "base-plus-insert" / "groundtruth.npy", *gt_args]),
        ("runtime", [python, tools / "write_runtime.py", "--source-runtime", root / "partition",
                     "--membership", root / "partition" / "membership.npy", "--base", src / "base.npy",
                     "--query", src / "query.npy", "--groundtruth", root / "gt" / "base" / "groundtruth.npy",
                     "--ghost", root / "ghost" / "ghost_indices.npy", "--output", root / "runtime",
                     "--parts", str(parts), "--metric", "l2", "--link-mode", "hardlink"]),
        ("stream", stream),
    ]
    return [(step, [str(item) for item in argv]) for step, argv in steps]


def outputs(root: Path) -> list[Path]:
    return [root / name for name in ("partition", "ghost", "gt", "runtime", "stream")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn a dataset and its Vamana graph into a K-server runtime.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stream-mode", choices=("shuffle", "skewed"), default="shuffle")
    parser.add_argument("--stream-repeats", type=int, default=64, help="shuffle mode: passes over the queries")
    parser.add_argument("--stream-seed", type=int, default=2)
    parser.add_argument("--stream-rows", type=int, default=262144, help="skewed mode: stream rows")
    parser.add_argument("--hot-share", type=float, default=0.7, help="skewed mode: share of hot rows")
    parser.add_argument("--regions", type=int, default=32, help="skewed mode: k-means regions")
    parser.add_argument("--hot-regions", type=int, default=3, help="skewed mode: hot regions")
    parser.add_argument("--parts", type=int, default=4, help="partitions, one per index server (2..64)")
    args = parser.parse_args(argv)
    root = args.dataset_root.absolute()
    required = [root / "source" / name for name in ("base.npy", "query.npy", "base_plus_insert.npy")]
    required.append(root / "index" / "vamana")
    missing = [str(path) for path in required if not path.is_file()]
    present = [str(path) for path in outputs(root) if path.exists()]
    if missing:
        print(f"error: missing inputs: {', '.join(missing)}", file=sys.stderr)
        return 2
    if present:
        print(f"error: outputs already exist: {', '.join(present)}", file=sys.stderr)
        return 2
    if args.stream_repeats < 1:
        print("error: --stream-repeats must be >= 1", file=sys.stderr)
        return 2
    n_nodes = int(np.load(root / "source" / "base.npy", mmap_mode="r").shape[0])
    steps = build_steps(root, sys.executable, n_nodes=n_nodes, stream_repeats=args.stream_repeats,
                        stream_seed=args.stream_seed, stream_mode=args.stream_mode, stream_rows=args.stream_rows,
                        hot_share=args.hot_share, regions=args.regions, hot_regions=args.hot_regions,
                        parts=args.parts)
    for name, step in steps:
        print(f"step {name}: {' '.join(step)}", flush=True)
        started = time.monotonic()
        result = subprocess.run(step, check=False)
        print(f"step {name}: exit {result.returncode} in {round(time.monotonic() - started, 3)} s", flush=True)
        if result.returncode != 0:
            print(f"error: step {name} failed", file=sys.stderr)
            return 2
    print(f"wrote {root} ({len(steps)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
