#!/usr/bin/env python3
"""Build a query stream that draws most rows from a few hot regions of a query set.

k-means splits the queries into --regions regions. Each region gets the server that owns most of its
queries' ground-truth top-1 neighbors. Regions are ranked by purity (then size, then index). The hot server is
the dominant server of the first region, and the hot regions are the first --hot-regions regions of that server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def kmeans(points: np.ndarray, k: int, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded Lloyd k-means with k-means++ initialization. Returns (labels, centers)."""
    if not 1 <= k <= len(points):
        raise ValueError(f"k={k} must be in [1, {len(points)}]")
    x = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    norms = (x * x).sum(axis=1)
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[rng.integers(len(x))]
    best = norms - 2.0 * x @ centers[0] + centers[0] @ centers[0]
    for c in range(1, k):
        weights = np.maximum(best, 0.0)
        total = float(weights.sum())
        index = int(rng.choice(len(x), p=weights / total)) if total > 0 else int(rng.integers(len(x)))
        centers[c] = x[index]
        best = np.minimum(best, norms - 2.0 * x @ centers[c] + centers[c] @ centers[c])
    labels = np.full(len(x), -1, dtype=np.int64)
    for _ in range(iterations):
        dist = norms[:, None] - 2.0 * x @ centers.T + (centers * centers).sum(axis=1)[None, :]
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return labels, centers


def select_hot_regions(labels: np.ndarray, top1_owner: np.ndarray, regions: int, hot_regions: int,
                       parts: int) -> tuple[int, list[int], list[dict]]:
    table = []
    for region in range(regions):
        members = top1_owner[labels == region]
        hist = np.bincount(members, minlength=parts) if len(members) else np.zeros(parts, dtype=np.int64)
        dominant = int(np.argmax(hist))
        purity = float(hist[dominant] / len(members)) if len(members) else 0.0
        table.append({"region": region, "size": int(len(members)), "dominant_owner": dominant,
                      "purity": round(purity, 6), "owner_histogram": [int(v) for v in hist]})
    ranked = sorted((row for row in table if row["size"] > 0),
                    key=lambda row: (-row["purity"], -row["size"], row["region"]))
    hot_owner = ranked[0]["dominant_owner"]
    hot = [row["region"] for row in ranked if row["dominant_owner"] == hot_owner][:hot_regions]
    if len(hot) < hot_regions:
        raise ValueError(f"only {len(hot)} regions are dominated by owner {hot_owner}, "
                         f"{hot_regions} requested")
    return hot_owner, hot, table


def cycle_permutations(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    passes = -(-count // len(pool))
    return np.concatenate([pool[rng.permutation(len(pool))] for _ in range(passes)])[:count]


def build(query: np.ndarray, truth: np.ndarray, membership: np.ndarray, *, rows: int, hot_share: float,
          regions: int, hot_regions: int, iterations: int, seed: int) -> dict:
    if query.ndim != 2 or truth.ndim != 2 or len(query) != len(truth) or truth.shape[1] < 10:
        raise ValueError("query/ground-truth rows or shape differ")
    if not 0.0 < hot_share < 1.0 or rows < 1 or not 1 <= hot_regions <= regions:
        raise ValueError("invalid rows, hot share or region counts")
    parts = int(membership.max()) + 1
    if int(truth[:, :10].max()) >= len(membership) or int(truth[:, :10].min()) < 0:
        raise ValueError("ground truth names an id outside the membership")
    top1 = membership.astype(np.int64)[np.asarray(truth[:, 0], dtype=np.int64)]
    labels, _ = kmeans(query, regions, iterations, seed)
    hot_owner, hot, table = select_hot_regions(labels, top1, regions, hot_regions, parts)
    is_hot = np.isin(labels, hot)
    hot_pool, rest_pool = np.flatnonzero(is_hot), np.flatnonzero(~is_hot)
    if not len(hot_pool) or not len(rest_pool):
        raise ValueError("empty hot or rest pool")
    rng = np.random.default_rng(seed + 1)
    n_hot = int(round(hot_share * rows))
    hot_rows = cycle_permutations(hot_pool, n_hot, rng)
    rest_rows = cycle_permutations(rest_pool, rows - n_hot, rng)
    positions = rng.permutation(rows)
    source = np.empty(rows, dtype=np.int64)
    source[positions[:n_hot]] = hot_rows
    source[positions[n_hot:]] = rest_rows
    return {"source_rows": source.astype(np.int32), "labels": labels, "hot_owner": hot_owner,
            "hot_regions": hot, "regions": table, "n_hot_rows": n_hot}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a query stream skewed towards a few hot regions.")
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=262144)
    parser.add_argument("--hot-share", type=float, default=0.7)
    parser.add_argument("--regions", type=int, default=32)
    parser.add_argument("--hot-regions", type=int, default=3)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"error: output exists: {args.output}", file=sys.stderr)
        return 2
    query = np.load(args.query, allow_pickle=False)
    truth = np.load(args.groundtruth, allow_pickle=False)
    membership = np.load(args.membership, allow_pickle=False)
    try:
        result = build(query, truth, membership, rows=args.rows, hot_share=args.hot_share,
                       regions=args.regions, hot_regions=args.hot_regions,
                       iterations=args.kmeans_iterations, seed=args.seed)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    source = result["source_rows"]
    args.output.mkdir(parents=True)
    np.save(args.output / "query.npy", np.ascontiguousarray(query[source]))
    np.save(args.output / "groundtruth.npy", np.ascontiguousarray(truth[source]))
    np.save(args.output / "source_rows.npy", source)
    np.save(args.output / "query_regions.npy", result["labels"].astype(np.int32))
    manifest = {
        "parameters": {"rows": args.rows, "hot_share": args.hot_share, "regions": args.regions,
                       "hot_regions": args.hot_regions, "kmeans_iterations": args.kmeans_iterations,
                       "seed": args.seed},
        "query_rows": int(len(query)), "stream_rows": int(len(source)), "hot_rows": result["n_hot_rows"],
        "hot_owner": result["hot_owner"], "hot_regions": result["hot_regions"], "regions": result["regions"],
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(source)} queries, {result['n_hot_rows']} from regions "
          f"{result['hot_regions']} of server {result['hot_owner']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
