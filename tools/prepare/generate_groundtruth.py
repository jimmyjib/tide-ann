#!/usr/bin/env python3
"""Exact top-k ground truth (l2 or cosine) on the CPU, as int32 ids [queries, k].

l2 is computed in float64 as |q|^2 + |v|^2 - 2 q.v. Ties are broken by the smaller vector id.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

QUERY_CHUNK = 64
BASE_CHUNK = 5000


def select_topk(scores: np.ndarray, ids: np.ndarray, k: int, largest: bool) -> tuple[np.ndarray, np.ndarray]:
    """Top k by score, ties broken by ascending vector id."""
    if scores.ndim != 2 or ids.ndim not in (1, 2) or not 0 < k <= scores.shape[1]:
        raise ValueError("scores/ids/k are incompatible")
    expanded_ids = np.broadcast_to(ids, scores.shape)
    selected_scores = np.empty((scores.shape[0], k), dtype=scores.dtype)
    selected_ids = np.empty((scores.shape[0], k), dtype=expanded_ids.dtype)
    for row in range(scores.shape[0]):
        primary = -scores[row] if largest else scores[row]
        order = np.lexsort((expanded_ids[row], primary))[:k]
        selected_scores[row] = scores[row, order]
        selected_ids[row] = expanded_ids[row, order]
    return selected_scores, selected_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute exact top-k ground truth on the CPU.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("l2", "cosine"), required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("the ground truth output already exists")
    base = np.load(args.base, mmap_mode="r", allow_pickle=False)
    query = np.load(args.query, mmap_mode="r", allow_pickle=False)
    if (base.dtype != np.float32 or query.dtype != np.float32 or base.ndim != 2 or query.ndim != 2
            or base.shape[1] != query.shape[1]):
        raise SystemExit("base/query must be compatible float32 matrices")
    if not 0 < args.k <= len(base) or len(query) == 0:
        raise SystemExit("invalid --k, or no queries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int32, shape=(len(query), args.k))
    l2 = args.metric == "l2"
    base_norm = np.sum(np.asarray(base, dtype=np.float64) ** 2, axis=1) if l2 else None
    for q_start in range(0, len(query), QUERY_CHUNK):
        q_end = min(q_start + QUERY_CHUNK, len(query))
        queries = np.asarray(query[q_start:q_end], dtype=np.float64 if l2 else np.float32)
        if l2:
            query_norm = np.sum(queries ** 2, axis=1, keepdims=True)
            best_scores = np.full((len(queries), args.k), np.inf, dtype=np.float64)
        else:
            query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
            if np.any(query_norm == 0):
                raise SystemExit("cosine query has zero norm")
            queries = queries / query_norm
            best_scores = np.full((len(queries), args.k), -np.inf, dtype=np.float32)
        best_ids = np.full((len(queries), args.k), -1, dtype=np.int32)
        for b_start in range(0, len(base), BASE_CHUNK):
            b_end = min(b_start + BASE_CHUNK, len(base))
            vectors = np.asarray(base[b_start:b_end], dtype=np.float64 if l2 else np.float32)
            if l2:
                scores = query_norm + base_norm[b_start:b_end][None, :] - 2.0 * (queries @ vectors.T)
            else:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise SystemExit("cosine base has zero norm")
                scores = queries @ (vectors / norms).T
            if not np.isfinite(scores).all():
                raise SystemExit("non-finite distance or similarity score")
            ids = np.arange(b_start, b_end, dtype=np.int32)[None, :]
            candidate_ids = np.concatenate((best_ids, np.broadcast_to(ids, scores.shape)), axis=1)
            best_scores, best_ids = select_topk(np.concatenate((best_scores, scores), axis=1), candidate_ids,
                                                args.k, largest=not l2)
        result[q_start:q_end] = best_ids
    result.flush()
    del result
    os.replace(temporary, args.output)
    print(f"wrote {args.output} ({len(query)} x {args.k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
