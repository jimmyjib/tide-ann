#!/usr/bin/env python3
"""Normalise vectors to unit L2 length: how cosine data is served.

The servers compute squared L2 only (there is no cosine build). For unit vectors squared L2 is
``2 - 2 cos``, so the L2 order is the cosine order, and product-quantization distances (squared L2
by construction) share one scale with the exact ones. Normalise every file the system reads or
compares against: the base, the queries and the inserts, then compute ground truth on the
normalised files with ``generate_groundtruth.py --metric l2``.

``--input A --output A' [--input B --output B' ...]``: each input is a float32 (or float64)
``rows x dim`` .npy; each output is a new float32 file. A row of zero length cannot be normalised
and is refused (``--zero-rows keep`` keeps such rows at zero instead). Prints one JSON summary.

Run with the directory that holds the ``tools`` package on ``PYTHONPATH`` as ``python -m tools.prepare.normalize_vectors``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CHUNK_ROWS = 1 << 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_file(source: Path, target: Path, keep_zero_rows: bool = False) -> dict:
    data = np.load(source, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or data.dtype.kind != "f":
        raise SystemExit(f"{source}: not a 2-D float array")
    if target.exists():
        raise SystemExit(f"refusing existing output {target}")
    out = np.lib.format.open_memmap(target, mode="w+", dtype="<f4", shape=data.shape)
    zero_rows = 0
    for start in range(0, len(data), CHUNK_ROWS):
        block = np.asarray(data[start:start + CHUNK_ROWS], dtype=np.float64)
        norms = np.linalg.norm(block, axis=1)
        zero = norms == 0
        if zero.any() and not keep_zero_rows:
            first = start + int(np.flatnonzero(zero)[0])
            del out
            target.unlink()
            raise SystemExit(f"{source}: row {first} has zero length and cannot be normalised")
        zero_rows += int(zero.sum())
        norms[zero] = 1.0
        out[start:start + len(block)] = (block / norms[:, None]).astype(np.float32)
    out.flush()
    del out
    with target.open("rb+") as handle:
        os.fsync(handle.fileno())
    check = np.load(target, mmap_mode="r", allow_pickle=False)
    lengths = np.linalg.norm(np.asarray(check[: min(len(check), 4096)], dtype=np.float64), axis=1)
    return {"input": str(source), "output": str(target), "rows": int(data.shape[0]), "dim": int(data.shape[1]),
            "zero_rows_kept": zero_rows, "input_sha256": sha256_file(source), "output_sha256": sha256_file(target),
            "sampled_length_min": float(lengths.min()) if len(lengths) else None,
            "sampled_length_max": float(lengths.max()) if len(lengths) else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--zero-rows", choices=("refuse", "keep"), default="refuse")
    args = parser.parse_args(argv)
    if len(args.input) != len(args.output):
        parser.error("give one --output per --input")
    if len({path.resolve() for path in args.output}) != len(args.output):
        parser.error("outputs must be distinct")
    dims = {np.load(src, mmap_mode="r", allow_pickle=False).shape[-1] for src in args.input}
    if len(dims) != 1:
        raise SystemExit(f"inputs have different dimensions {sorted(dims)}")
    files = [normalize_file(src, dst, args.zero_rows == "keep") for src, dst in zip(args.input, args.output)]
    print(json.dumps({"status": "COMPLETE", "metric_after": "l2 on unit vectors (cosine order)",
                      "files": files}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
