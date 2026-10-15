#!/usr/bin/env python3
"""Exact chunked top-k ground truth for a query range.

The producer is CPU-only and deterministic.  It supports disjoint query ranges,
so a large query set can be computed in short independent runs whose outputs
are then merged into one complete exact result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def stable_inner_products(
    queries: np.ndarray,
    vectors: np.ndarray,
    *,
    matmul=np.matmul,
) -> tuple[np.ndarray, str]:
    products = matmul(queries, vectors.T)
    if np.isfinite(products).all():
        return products, "matmul"
    products = np.einsum("qd,bd->qb", queries, vectors, optimize=False)
    if not np.isfinite(products).all():
        raise ValueError("non-finite inner product in both matmul and einsum")
    return products, "einsum-fallback"


def squared_l2_scores(
    queries: np.ndarray,
    vectors: np.ndarray,
    *,
    kernel: str,
    query_norm: np.ndarray | None = None,
    vector_norm: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if kernel == "scipy-cdist":
        try:
            from scipy.spatial.distance import cdist
        except ImportError as exc:  # pragma: no cover - environment contract
            raise ValueError("scipy-cdist requested but scipy is unavailable") from exc
        scores = cdist(queries, vectors, metric="sqeuclidean")
        if not np.isfinite(scores).all():
            raise ValueError("non-finite scipy-cdist squared-L2 score")
        return scores, "scipy-cdist-sqeuclidean-float64"
    if kernel != "norm-dot":
        raise ValueError(f"unsupported L2 kernel: {kernel}")
    if query_norm is None or vector_norm is None:
        raise ValueError("norm-dot requires query and vector norms")
    inner, inner_kernel = stable_inner_products(queries, vectors)
    scores = query_norm + vector_norm[None, :] - 2.0 * inner
    if not np.isfinite(scores).all():
        raise ValueError("non-finite norm-dot squared-L2 score")
    return scores, f"norm-dot-{inner_kernel}"


def select_topk(
    scores: np.ndarray, ids: np.ndarray, k: int, largest: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Order by score, resolving the cutoff with ascending vector ID."""
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("l2", "cosine"), required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--base-chunk", type=int, default=5000)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--verified-base-sha256")
    parser.add_argument("--verified-query-sha256")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int)
    parser.add_argument("--l2-float64", action="store_true")
    parser.add_argument(
        "--l2-kernel", choices=("norm-dot", "scipy-cdist"), default="norm-dot"
    )
    args = parser.parse_args()

    manifest_output = args.output.with_suffix(args.output.suffix + ".json")
    temporary_output = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    temporary_manifest = manifest_output.with_name(
        manifest_output.name + f".tmp-{os.getpid()}"
    )
    if args.output.exists() or manifest_output.exists():
        raise SystemExit("refusing to overwrite ground truth")
    base = np.load(args.base, mmap_mode="r", allow_pickle=False)
    query = np.load(args.query, mmap_mode="r", allow_pickle=False)
    if (
        base.dtype != np.float32
        or query.dtype != np.float32
        or base.ndim != 2
        or query.ndim != 2
        or base.shape[1] != query.shape[1]
    ):
        raise SystemExit("base/query must be compatible float32 matrices")
    if not 0 < args.k <= len(base) or args.query_chunk <= 0 or args.base_chunk <= 0:
        raise SystemExit("invalid chunk or k")
    query_end = len(query) if args.query_end is None else args.query_end
    if not 0 <= args.query_start < query_end <= len(query):
        raise SystemExit("invalid query range")
    query_range = query[args.query_start:query_end]
    result = np.lib.format.open_memmap(
        temporary_output,
        mode="w+",
        dtype=np.int32,
        shape=(len(query_range), args.k),
    )
    l2_dtype = np.float64 if args.l2_float64 else np.float32
    l2_kernel = None
    base_norm = (
        np.sum(np.asarray(base, dtype=l2_dtype) ** 2, axis=1)
        if args.metric == "l2" and args.l2_kernel == "norm-dot"
        else None
    )
    for q_start in range(0, len(query_range), args.query_chunk):
        q_end = min(q_start + args.query_chunk, len(query_range))
        queries = np.asarray(
            query_range[q_start:q_end],
            dtype=l2_dtype if args.metric == "l2" else np.float32,
        )
        if args.metric == "cosine":
            query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
            if np.any(query_norm == 0):
                raise SystemExit("cosine query has zero norm")
            queries = queries / query_norm
            best_scores = np.full((len(queries), args.k), -np.inf, dtype=np.float32)
        else:
            query_norm = (
                np.sum(queries ** 2, axis=1, keepdims=True)
                if args.l2_kernel == "norm-dot"
                else None
            )
            best_scores = np.full((len(queries), args.k), np.inf, dtype=np.float64)
        best_ids = np.full((len(queries), args.k), -1, dtype=np.int32)
        for b_start in range(0, len(base), args.base_chunk):
            b_end = min(b_start + args.base_chunk, len(base))
            vectors = np.asarray(
                base[b_start:b_end],
                dtype=l2_dtype if args.metric == "l2" else np.float32,
            )
            if args.metric == "cosine":
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise SystemExit("cosine base has zero norm")
                scores = queries @ (vectors / norms).T
            else:
                try:
                    scores, kernel = squared_l2_scores(
                        queries,
                        vectors,
                        kernel=args.l2_kernel,
                        query_norm=query_norm,
                        vector_norm=None if base_norm is None else base_norm[b_start:b_end],
                    )
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                if l2_kernel is not None and l2_kernel != kernel:
                    raise SystemExit("L2 kernel changed within one execution")
                l2_kernel = kernel
            if not np.isfinite(scores).all():
                raise SystemExit("non-finite distance/similarity score")
            ids = np.arange(b_start, b_end, dtype=np.int32)[None, :]
            best_scores, best_ids = select_topk(
                np.concatenate((best_scores, scores), axis=1),
                np.concatenate((best_ids, np.broadcast_to(ids, scores.shape)), axis=1),
                args.k,
                largest=args.metric == "cosine",
            )
        result[q_start:q_end] = best_ids
        result.flush()
        if args.progress_file:
            args.progress_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.progress_file.with_suffix(args.progress_file.suffix + ".tmp")
            temporary.write_text(
                json.dumps({
                    "completed_queries": q_end,
                    "total_queries": len(query_range),
                }) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, args.progress_file)
    if bool(args.verified_base_sha256) != bool(args.verified_query_sha256):
        raise SystemExit("verified base/query hashes must be supplied together")
    base_sha256 = args.verified_base_sha256 or digest(args.base)
    query_sha256 = args.verified_query_sha256 or digest(args.query)
    manifest = {
        "schema": {"name": "tide-exact-groundtruth", "version": 2},
        "metric": args.metric,
        "k": args.k,
        "N": int(len(base)),
        "queries": int(len(query_range)),
        "full_query_count": int(len(query)),
        "query_range": [args.query_start, query_end],
        "dimension": int(base.shape[1]),
        "base_sha256": base_sha256,
        "query_sha256": query_sha256,
        "groundtruth_sha256": digest(temporary_output),
        "query_chunk": args.query_chunk,
        "base_chunk": args.base_chunk,
        "l2_float64": args.l2_float64,
        "l2_inner_product_kernel": l2_kernel if args.metric == "l2" else None,
        "tie_break": "distance_or_negative_similarity_then_vector_id_ascending",
        "producer_sha256": digest(Path(__file__).resolve()),
    }
    temporary_manifest.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    os.replace(temporary_manifest, manifest_output)
    print(json.dumps({
        "status": "COMPLETE",
        "output": str(args.output),
        "sha256": manifest["groundtruth_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
