#!/usr/bin/env python3
"""Extract per-node directed adjacency row bytes from a sealed runtime.

Row cost matches the row serialization of a node move (MOVE): 4 bytes for
the degree word plus 4 bytes per neighbour
(``4 + 4 * degree``), saved as int64. The manifest records the runtime
identity so planner inputs stay pinned to the graph they price.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.prepare.runtime_lib import load_global_adjacency


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    manifest = json.loads(
        (args.runtime / "RUNTIME_MANIFEST.json").read_text(encoding="utf-8"))
    node_count = manifest["N"]
    parts = manifest.get("parts", 4)
    rows, _start = load_global_adjacency(args.runtime, node_count, parts)
    if len(rows) != node_count:
        raise SystemExit(
            f"adjacency rows {len(rows)} != manifest N {node_count}")
    degrees = np.fromiter(
        (len(rows[node]) for node in range(node_count)),
        dtype=np.int64, count=node_count)
    row_bytes = 4 + 4 * degrees

    args.output.mkdir(parents=True)
    out = args.output / "directed_row_bytes.npy"
    np.save(out, row_bytes)
    record = {
        "schema": {"name": "directed-row-bytes-extract", "version": 1},
        "runtime": str(args.runtime),
        "runtime_manifest_sha256": sha256_path(
            args.runtime / "RUNTIME_MANIFEST.json"),
        "directed_graph_sha256": manifest.get("directed_graph_sha256"),
        "N": int(node_count),
        "total_bytes": int(row_bytes.sum()),
        "row_rule": "4 + 4 * degree (int64)",
        "output_sha256": sha256_path(out),
    }
    (args.output / "ROW_BYTES_MANIFEST.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(f"ROW_BYTES=PASS n={node_count} total={record['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
