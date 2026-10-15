#!/usr/bin/env python3
"""Normalize vectors to unit length, so that squared L2 ranks them like cosine similarity.

Normalize the base, query and insert files, then compute ground truth with
generate_groundtruth.py --metric l2. Each --input/--output pair writes a new float32 .npy.
A zero row is rejected unless --zero-rows keep is given.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CHUNK_ROWS = 1 << 16


def normalize_file(source: Path, target: Path, keep_zero_rows: bool = False) -> tuple[int, int]:
    """Write the unit-length rows of source to target. Returns (rows, zero rows kept)."""
    data = np.load(source, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or data.dtype.kind != "f":
        raise SystemExit(f"{source}: not a 2-D float array")
    if target.exists():
        raise SystemExit(f"{target} already exists")
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
            raise SystemExit(f"{source}: row {first} has zero norm (use --zero-rows keep)")
        zero_rows += int(zero.sum())
        norms[zero] = 1.0
        out[start:start + len(block)] = (block / norms[:, None]).astype(np.float32)
    out.flush()
    del out
    with target.open("rb+") as handle:
        os.fsync(handle.fileno())
    return int(data.shape[0]), zero_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize vectors to unit length.")
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
    for source, target in zip(args.input, args.output):
        rows, zero_rows = normalize_file(source, target, args.zero_rows == "keep")
        print(f"wrote {target} ({rows} rows, {zero_rows} zero rows kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
