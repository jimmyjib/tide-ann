#!/usr/bin/env python3
"""Cut a regression dataset (base, inserts, queries) out of an existing float32 vector set.

The output has the layout of ``make_synthetic.py`` (``base.npy``, ``insert.npy``,
``query.npy``, ``base_plus_insert.npy``, ``MANIFEST.json``), so ``prepare_demo_runtime.py``
turns it into a sealed runtime unchanged:

* base   = rows ``[0, base_rows)`` of ``--source-base``;
* insert = rows ``[insert_start, insert_start + insert_rows)`` of the same file, which must
  not overlap the base rows;
* query  = every row of ``--query`` (a query set disjoint from the source base);
* base_plus_insert = base rows then insert rows, so row ``r`` is id ``r`` once the insert
  file is inserted from row 0 (insert row ``i`` becomes id ``base_rows + i``).

Every input is pinned by sha256 in the manifest. Rows are copied, never transformed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = {"name": "tide-sliced-dataset", "version": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-base", type=Path, required=True, help="float32 N x D .npy")
    parser.add_argument("--query", type=Path, required=True, help="float32 Q x D .npy, disjoint from the source")
    parser.add_argument("--base-rows", type=int, required=True)
    parser.add_argument("--insert-start", type=int, required=True)
    parser.add_argument("--insert-rows", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--note", default="test data sliced from a public dataset; "
                                                    "supports no performance claim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        print(f"SLICED DATASET FAILED: refusing to overwrite {output}", file=sys.stderr)
        return 2
    try:
        arrays = slice_arrays(np.load(args.source_base, mmap_mode="r", allow_pickle=False),
                              np.load(args.query, allow_pickle=False), base_rows=args.base_rows,
                              insert_start=args.insert_start, insert_rows=args.insert_rows)
    except ValueError as error:
        print(f"SLICED DATASET FAILED: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for name, array in arrays.items():
            np.save(staging / name, array)
        manifest = {
            "schema": SCHEMA, "label": args.label, "note": args.note,
            "source": {"base": {"path": str(args.source_base.resolve()), "sha256": sha256_file(args.source_base)},
                       "query": {"path": str(args.query.resolve()), "sha256": sha256_file(args.query)}},
            "rows": {"base": [0, args.base_rows],
                     "insert": [args.insert_start, args.insert_start + args.insert_rows],
                     "query": [0, int(len(arrays["query.npy"]))]},
            "id_contract": "base ids 0..N-1; insert row i becomes id N+i when the insert file is inserted "
                           "from row 0; base_plus_insert.npy row r is id r",
            "producer_sha256": sha256_file(Path(__file__).resolve()),
            "shapes": {name: list(array.shape) for name, array in arrays.items()},
            "dtypes": {name: str(array.dtype) for name, array in arrays.items()},
            "files": {name: {"bytes": (staging / name).stat().st_size, "sha256": sha256_file(staging / name)}
                      for name in arrays},
        }
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "COMPLETE", "output": str(output), "shapes": manifest["shapes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
