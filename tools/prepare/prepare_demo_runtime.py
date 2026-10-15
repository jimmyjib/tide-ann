#!/usr/bin/env python3
"""Turn a dataset plus its Vamana graph into a sealed K-server runtime (K = ``--parts``, default 4).

This runs, in order, the data-preparation tools of ``tools/prepare`` with demo paths:

1. ``partition``            partition_graph.py: node-balanced K-way METIS of the graph;
2. ``ghost``                make_ghost_ids.py: global head ids (1% of the base, seed 42);
3. ``gt-base``              exact top-10 ground truth of the queries on the base;
4. ``gt-base-plus-insert``  the same on base + inserts (ids N.. are the inserts);
5. ``seal``                 seal_runtime.py: the sealed runtime (base GT as the native GT);
6. ``row-bytes``            extract_directed_row_bytes.py: 4 + 4*degree per node;
7. ``stream``               the controller demo's query stream, either
                            ``shuffle`` (build_query_load_stream.py: seeded shuffles of the
                            native queries, GT permuted alike; demo-s) or ``skewed``
                            (build_skewed_query_stream.py: most rows from a few hot query
                            regions of one owner).

Expected layout under ``--dataset-root``: ``source/`` (make_synthetic.py or
slice_dataset.py output) and ``index/vamana`` (build_vamana.py output). Each step's output
must not exist yet. The driver stops at the first failing step and leaves every output for
inspection; it writes ``PREP_COMPLETE.json`` only when all steps pass.

Run with the directory that holds the ``tools`` package on ``PYTHONPATH``; the step tools inherit it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
# The directory that holds the tools package (the repository root).
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = {"name": "tide-demo-runtime-prep", "version": 2}


def build_steps(root: Path, python: str, *, n_nodes: int, producer_commit: str, parts: int = 4,
                stream_repeats: int = 64, stream_seed: int = 2, name: str = "demo-s",
                checkpoint: str = "demo-s", stream_mode: str = "shuffle",
                stream_rows: int = 262144, hot_share: float = 0.7, regions: int = 32, hot_regions: int = 3,
                tools: Path = TOOLS) -> list[tuple[str, list[str]]]:
    src, index = root / "source", root / "index"
    gt_args = ["--metric", "l2", "--k", "10", "--l2-kernel", "norm-dot", "--l2-float64"]
    if stream_mode == "shuffle":
        stream = [python, "-B", tools / "build_query_load_stream.py", "--query", root / "runtime" / "query.npy",
                  "--groundtruth", root / "runtime" / "groundtruth.npy", "--repeats", str(stream_repeats),
                  "--seed", str(stream_seed), "--output", root / "stream",
                  "--label", f"{name}-native-x{stream_repeats}"]
    elif stream_mode == "skewed":
        stream = [python, "-B", tools / "build_skewed_query_stream.py", "--query", root / "runtime" / "query.npy",
                  "--groundtruth", root / "runtime" / "groundtruth.npy",
                  "--membership", root / "runtime" / "membership.npy", "--rows", str(stream_rows),
                  "--hot-share", str(hot_share), "--regions", str(regions), "--hot-regions", str(hot_regions),
                  "--seed", str(stream_seed), "--output", root / "stream",
                  "--label", f"{name}-skewed-h{hot_share:g}-r{regions}x{hot_regions}"]
    else:
        raise ValueError(f"unknown stream mode {stream_mode}")
    steps = [
        ("partition", [python, "-B", tools / "partition_graph.py", "--graph", index / "vamana",
                       "--base", src / "base.npy", "--output", root / "partition", "--parts", str(parts)]),
        ("ghost", [python, "-B", tools / "make_ghost_ids.py", "--nodes", str(n_nodes),
                   "--output", root / "ghost" / "ghost_indices.npy", "--ratio", "0.01", "--seed", "42"]),
        ("gt-base", [python, tools / "generate_groundtruth_mkdir.py", "--base", src / "base.npy",
                     "--query", src / "query.npy", "--output", root / "gt" / "base" / "groundtruth.npy",
                     *gt_args]),
        ("gt-base-plus-insert", [python, tools / "generate_groundtruth_mkdir.py",
                                 "--base", src / "base_plus_insert.npy", "--query", src / "query.npy",
                                 "--output", root / "gt" / "base-plus-insert" / "groundtruth.npy", *gt_args]),
        ("seal", [python, tools / "seal_runtime.py", "--source-runtime", root / "partition",
                  "--membership", root / "partition" / "membership.npy", "--base", src / "base.npy",
                  "--query", src / "query.npy", "--groundtruth", root / "gt" / "base" / "groundtruth.npy",
                  "--ghost", root / "ghost" / "ghost_indices.npy", "--output", root / "runtime",
                  "--parts", str(parts), "--metric", "l2", "--link-mode", "hardlink", "--label", name,
                  "--checkpoint", checkpoint, "--producer-commit", producer_commit]),
        ("row-bytes", [python, "-B", tools / "extract_directed_row_bytes.py", "--runtime", root / "runtime",
                       "--output", root / "row-bytes"]),
        ("stream", stream),
    ]
    return [(step, [str(item) for item in argv]) for step, argv in steps]


def outputs(root: Path) -> list[Path]:
    return [root / name for name in ("partition", "ghost", "gt", "runtime", "row-bytes", "stream",
                                     "PREP_COMPLETE.json")]


def source_revision() -> str:
    """git HEAD of the tree that holds this tool, or 'unknown' when it is not a git checkout."""
    try:
        result = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "rev-parse", "--show-toplevel", "HEAD"],
                                check=False, capture_output=True, text=True)
    except OSError:
        return "unknown"
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 2:
        return "unknown"
    # Only this tree's own repository: the tool itself must be tracked there, so a repository that
    # merely contains an untracked copy of the tree is never named.
    tracked = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "ls-files", "--error-unmatch", "--",
                              "tools/prepare/prepare_demo_runtime.py"], check=False, capture_output=True, text=True)
    return lines[1] if tracked.returncode == 0 else "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--name", default="demo-s", help="dataset name used in labels")
    parser.add_argument("--checkpoint", default="demo-s", help="seal_runtime --checkpoint label")
    parser.add_argument("--stream-mode", choices=("shuffle", "skewed"), default="shuffle")
    parser.add_argument("--stream-repeats", type=int, default=64, help="shuffle mode: native passes")
    parser.add_argument("--stream-seed", type=int, default=2)
    parser.add_argument("--stream-rows", type=int, default=262144, help="skewed mode: stream rows")
    parser.add_argument("--hot-share", type=float, default=0.7, help="skewed mode: share of hot rows")
    parser.add_argument("--regions", type=int, default=32, help="skewed mode: k-means regions")
    parser.add_argument("--hot-regions", type=int, default=3, help="skewed mode: hot regions")
    parser.add_argument("--parts", type=int, default=4, help="partitions, one per index server (2..64)")
    parser.add_argument("--producer-commit", default=None,
                        help="source revision recorded in the runtime manifest (default: git HEAD of this tree, "
                             "or 'unknown' outside a git checkout)")
    args = parser.parse_args(argv)
    root = args.dataset_root.absolute()
    required = [root / "source" / name for name in ("base.npy", "query.npy", "base_plus_insert.npy",
                                                    "MANIFEST.json")] + [root / "index" / "vamana",
                                                                         root / "index" / "GRAPH_BUILD.json"]
    missing = [str(path) for path in required if not path.is_file()]
    present = [str(path) for path in outputs(root) if path.exists()]
    if missing or present or args.stream_repeats < 1:
        print(f"DEMO PREP REFUSED: missing={missing} existing_outputs={present}", file=sys.stderr)
        return 2
    n_nodes = int(np.load(root / "source" / "base.npy", mmap_mode="r").shape[0])
    commit = args.producer_commit or source_revision()
    records = []
    for name, step in build_steps(root, sys.executable, n_nodes=n_nodes, producer_commit=commit,
                                  stream_repeats=args.stream_repeats, stream_seed=args.stream_seed,
                                  name=args.name, checkpoint=args.checkpoint, stream_mode=args.stream_mode,
                                  stream_rows=args.stream_rows, hot_share=args.hot_share,
                                  regions=args.regions, hot_regions=args.hot_regions, parts=args.parts):
        print(f"=== step {name}: {' '.join(step)}", flush=True)
        started = time.monotonic()
        result = subprocess.run(step, check=False)
        seconds = round(time.monotonic() - started, 3)
        records.append({"step": name, "argv": step, "returncode": result.returncode, "seconds": seconds})
        print(f"=== step {name}: exit {result.returncode} in {seconds}s", flush=True)
        if result.returncode != 0:
            print(f"DEMO PREP FAILED at step {name}", file=sys.stderr)
            return 2
    key_files = ["runtime/RUNTIME_MANIFEST.json", "runtime/COMPLETE", "gt/base/groundtruth.npy",
                 "gt/base-plus-insert/groundtruth.npy", "row-bytes/directed_row_bytes.npy",
                 "stream/query.npy", "stream/groundtruth.npy", "ghost/ghost_indices.npy",
                 "partition/membership.npy"]
    manifest = {"schema": SCHEMA, "dataset_root": str(root), "name": args.name, "producer_commit": commit,
                "n_nodes": n_nodes, "stream_mode": args.stream_mode, "steps": records,
                "files": {name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
                          for name in key_files}}
    (root / "PREP_COMPLETE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "dataset_root": str(root), "steps": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
