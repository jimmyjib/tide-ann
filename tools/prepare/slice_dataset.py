#!/usr/bin/env python3
"""Cut base, insert and query files out of an existing float32 vector set.

The base is rows [0, --base-rows) of --source-base and the inserts are --insert-rows rows from
--insert-start. The queries are all rows of --query. The output has the layout of make_synthetic.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def slice_arrays(source: np.ndarray, query: np.ndarray, *, base_rows: int, insert_start: int,
                 insert_rows: int) -> dict[str, np.ndarray]:
    if source.dtype != np.float32 or source.ndim != 2 or query.dtype != np.float32 or query.ndim != 2:
        raise ValueError("source and query must be float32 matrices")
    if source.shape[1] != query.shape[1]:
        raise ValueError(f"dimension differs: source {source.shape[1]} query {query.shape[1]}")
    if base_rows < 1 or insert_rows < 0 or insert_start < base_rows:
        raise ValueError("insert rows must follow the base rows without overlap")
    if insert_start + insert_rows > len(source):
        raise ValueError(f"insert rows end at {insert_start + insert_rows} > {len(source)} source rows")
    base = np.ascontiguousarray(source[:base_rows])
    insert = np.ascontiguousarray(source[insert_start:insert_start + insert_rows])
    return {"base.npy": base, "insert.npy": insert, "query.npy": np.ascontiguousarray(query),
            "base_plus_insert.npy": np.ascontiguousarray(np.concatenate([base, insert]))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut a dataset out of an existing float32 vector set.")
    parser.add_argument("--source-base", type=Path, required=True, help="float32 N x D .npy")
    parser.add_argument("--query", type=Path, required=True, help="float32 Q x D .npy, disjoint from the source")
    parser.add_argument("--base-rows", type=int, required=True)
    parser.add_argument("--insert-start", type=int, required=True)
    parser.add_argument("--insert-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        print(f"error: {output} already exists", file=sys.stderr)
        return 2
    try:
        arrays = slice_arrays(np.load(args.source_base, mmap_mode="r", allow_pickle=False),
                              np.load(args.query, allow_pickle=False), base_rows=args.base_rows,
                              insert_start=args.insert_start, insert_rows=args.insert_rows)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for name, array in arrays.items():
            np.save(staging / name, array)
        manifest = {"base_rows": args.base_rows, "insert_start": args.insert_start, "insert_rows": args.insert_rows,
                    "shapes": {name: list(array.shape) for name, array in arrays.items()}}
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    base = arrays["base.npy"]
    print(f"wrote {output} (base {len(base)} x {base.shape[1]}, {len(arrays['query.npy'])} queries, "
          f"{len(arrays['insert.npy'])} inserts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
