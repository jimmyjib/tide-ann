#!/usr/bin/env python3
"""Build a skewed query load stream from a uniform native query set.

The controller needs a sustained stream whose work is clearly unequal across the
servers to have something to repartition. A uniform query set on a METIS partition gives nearly equal work, so this tool
draws most of the stream from a few hot regions of the query space:

1. k-means (Lloyd, k-means++ seeding, fixed seed) clusters the native queries into
   ``--regions`` regions;
2. every query is labelled with the owner of its ground-truth top-1 neighbour
   (``membership[gt[:, 0]]``), and every region with its dominant owner and purity;
3. the hot regions are chosen by one fixed rule: rank the regions by purity (then size,
   then index); the dominant owner of the first region is the hot owner, and the hot
   regions are the first ``--hot-regions`` regions of the ranking with that owner;
4. ``round(hot_share * rows)`` stream rows cycle through seeded permutations of the hot
   regions' queries, the rest cycle through seeded permutations of all other queries, and
   a seeded permutation of the positions interleaves the two.

Each stream row's ground truth is the source query's row, so recall stays measurable;
``source_rows.npy`` names the source query of every stream row. The predicted owner shares
(owner of GT top-1, and the mean over the GT top-10 owners) are a design proxy for the
server imbalance, not a measurement: only a served run measures per-server work.

The stream repeats native queries by construction and is load material only; evaluate
recall on the uniform native set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = "tide-skewed-query-stream-v1"
NOTE = "SUSTAINED_SKEWED_LOAD_WORKLOAD_ONLY_NOT_FOR_RECALL_EVALUATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def kmeans(points: np.ndarray, k: int, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Lloyd k-means with k-means++ seeding; returns (labels, centres)."""
    if not 1 <= k <= len(points):
        raise ValueError(f"k={k} must be in [1, {len(points)}]")
    x = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    norms = (x * x).sum(axis=1)
    centres = np.empty((k, x.shape[1]), dtype=np.float64)
    centres[0] = x[rng.integers(len(x))]
    best = norms - 2.0 * x @ centres[0] + centres[0] @ centres[0]
    for c in range(1, k):
        weights = np.maximum(best, 0.0)
        total = float(weights.sum())
        index = int(rng.choice(len(x), p=weights / total)) if total > 0 else int(rng.integers(len(x)))
        centres[c] = x[index]
        best = np.minimum(best, norms - 2.0 * x @ centres[c] + centres[c] @ centres[c])
    labels = np.full(len(x), -1, dtype=np.int64)
    for _ in range(iterations):
        dist = norms[:, None] - 2.0 * x @ centres.T + (centres * centres).sum(axis=1)[None, :]
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                centres[c] = members.mean(axis=0)
    return labels, centres


def owner_shares(owners: np.ndarray, rows: np.ndarray, parts: int) -> np.ndarray:
    return np.bincount(owners[rows].ravel(), minlength=parts).astype(np.float64) / owners[rows].size


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
        raise ValueError(f"only {len(hot)} regions are dominated by owner {hot_owner}; "
                         f"{hot_regions} hot regions requested")
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
    owners = membership.astype(np.int64)[np.asarray(truth[:, :10], dtype=np.int64)]
    top1 = owners[:, 0]
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
    native = np.arange(len(query))

    def shares(selection: np.ndarray) -> dict:
        top1_share = np.bincount(top1[selection], minlength=parts) / len(selection)
        top10_share = owner_shares(owners, selection, parts)
        return {"top1_owner_share": [round(float(v), 6) for v in top1_share],
                "top1_max_over_mean": round(float(top1_share.max() * parts), 6),
                "top10_owner_share": [round(float(v), 6) for v in top10_share],
                "top10_max_over_mean": round(float(top10_share.max() * parts), 6)}

    return {"source_rows": source.astype(np.int32), "labels": labels, "hot_owner": hot_owner,
            "hot_regions": hot, "regions": table, "n_hot_rows": n_hot, "hot_pool": int(len(hot_pool)),
            "rest_pool": int(len(rest_pool)), "parts": parts,
            "proxy": {"native": shares(native), "stream": shares(source),
                      "first_4096_rows": shares(source[:4096])}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"SKEWED STREAM REFUSED: output exists: {args.output}", file=sys.stderr)
        return 2
    query = np.load(args.query, allow_pickle=False)
    truth = np.load(args.groundtruth, allow_pickle=False)
    membership = np.load(args.membership, allow_pickle=False)
    try:
        result = build(query, truth, membership, rows=args.rows, hot_share=args.hot_share,
                       regions=args.regions, hot_regions=args.hot_regions,
                       iterations=args.kmeans_iterations, seed=args.seed)
    except ValueError as error:
        print(f"SKEWED STREAM FAILED: {error}", file=sys.stderr)
        return 2
    source = result["source_rows"]
    args.output.mkdir(parents=True)
    np.save(args.output / "query.npy", np.ascontiguousarray(query[source]))
    np.save(args.output / "groundtruth.npy", np.ascontiguousarray(truth[source]))
    np.save(args.output / "source_rows.npy", source)
    np.save(args.output / "query_regions.npy", result["labels"].astype(np.int32))
    manifest = {
        "schema": SCHEMA, "note": NOTE, "label": args.label,
        "parameters": {"rows": args.rows, "hot_share": args.hot_share, "regions": args.regions,
                       "hot_regions": args.hot_regions, "kmeans_iterations": args.kmeans_iterations,
                       "seed": args.seed,
                       "rule": "regions ranked by (purity desc, size desc, index asc); hot owner = dominant "
                               "owner of the first; hot regions = first hot_regions of that owner"},
        "native_rows": int(len(query)), "stream_rows": int(len(source)), "hot_rows": result["n_hot_rows"],
        "hot_owner": result["hot_owner"], "hot_regions": result["hot_regions"],
        "hot_pool_queries": result["hot_pool"], "rest_pool_queries": result["rest_pool"],
        "regions": result["regions"], "predicted_owner_share_proxy": result["proxy"],
        "source": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                   for name, path in (("query", args.query), ("groundtruth", args.groundtruth),
                                      ("membership", args.membership))},
        "files": {name: {"bytes": (args.output / name).stat().st_size,
                         "sha256": sha256_file(args.output / name)}
                  for name in ("query.npy", "groundtruth.npy", "source_rows.npy", "query_regions.npy")},
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "hot_owner": result["hot_owner"],
                      "hot_regions": result["hot_regions"],
                      "stream_top1_max_over_mean": result["proxy"]["stream"]["top1_max_over_mean"],
                      "stream_top10_max_over_mean": result["proxy"]["stream"]["top10_max_over_mean"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
