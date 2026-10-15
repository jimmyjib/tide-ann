#!/usr/bin/env python3
"""A biased insert set: inserts concentrated in the hot query region, on the
boundary between the hot owner and the other servers, so a controller phase after the inserts has
inserted ids to move.

Inputs: a float32 source vector set (``--source``, e.g. the larger vector set a sliced base was
cut from) and a candidate row range of it that overlaps neither the runtime's base rows nor any
other insert set (``--candidate-start`` >= ``--exclude-below``); the runtime's ``base.npy`` and
``membership.npy`` (load-time owners); the uniform queries and their k-means regions together with
the skewed stream's manifest (its ``hot_owner`` and ``hot_regions``, ``build_skewed_query_stream``).

Selection (deterministic; ties by source row):

1. hot centroids: the mean of the queries of each hot region; every candidate's squared L2 distance
   to its nearest hot centroid is its hot distance;
2. the ``--prefilter`` candidates with the smallest hot distance;
3. their exact ``--neighbours`` nearest base nodes (squared L2, ties by id) and the share of those
   owned by the hot owner;
4. ``boundary`` candidates: a hot-owner share in [``--min-hot-share``, ``--max-hot-share``] (some of
   their nearest nodes live on other servers), taken in hot-distance order up to ``--max-boundary``
   (default ``--count``); the rest is ``fill``: the nearest-to-hot candidates not taken, whatever
   their share (counted in the manifest). A mix keeps the set concentrated on the hot region while
   giving the controller boundary nodes whose rows can favour another server;
5. the selected rows in a seeded random order (``--seed``), so the first and the second half of the
   file (e.g. two consecutive insert phases) are drawn alike.

Outputs (a fresh ``--output`` directory): ``insert.npy`` (``--count`` x D float32, rows copied
unchanged), ``base_plus_insert.npy`` (the runtime base then the inserts: row r is id r once the
file is inserted from row 0), ``source_rows.npy`` (the source row of each insert) and
``MANIFEST.json`` (inputs pinned by sha256, the parameters, the selection's numbers, and the
predicted home owner of each insert: the owner of its nearest base node).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = {"name": "tide-biased-insert-set", "version": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hot_centroids(query: np.ndarray, regions: np.ndarray, hot_regions: list[int]) -> np.ndarray:
    """The mean query vector of each hot region (float64), in the order hot_regions lists them."""
    if len(regions) != len(query):
        raise ValueError("one region label per query is required")
    centroids = []
    for region in hot_regions:
        rows = np.flatnonzero(regions == region)
        if not len(rows):
            raise ValueError(f"hot region {region} has no query")
        centroids.append(np.asarray(query[rows], dtype=np.float64).mean(axis=0))
    return np.stack(centroids)


def hot_distance(candidates: np.ndarray, centroids: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Squared L2 distance of every candidate to its nearest hot centroid (float64)."""
    out = np.empty(len(candidates), dtype=np.float64)
    c_norm = np.sum(centroids ** 2, axis=1)
    for start in range(0, len(candidates), chunk):
        block = np.asarray(candidates[start:start + chunk], dtype=np.float64)
        scores = np.sum(block ** 2, axis=1)[:, None] + c_norm[None, :] - 2.0 * block @ centroids.T
        out[start:start + len(block)] = scores.min(axis=1)
    return out


def nearest_base(points: np.ndarray, base: np.ndarray, k: int, chunk: int = 128) -> np.ndarray:
    """The ids of each point's k nearest base vectors (squared L2 in float32 norm-dot, ties by id)."""
    base32 = np.asarray(base, dtype=np.float32)
    base_norm = np.sum(base32.astype(np.float64) ** 2, axis=1).astype(np.float32)
    ids = np.empty((len(points), k), dtype=np.int64)
    for start in range(0, len(points), chunk):
        block = np.asarray(points[start:start + chunk], dtype=np.float32)
        scores = base_norm[None, :] - 2.0 * (block @ base32.T)
        # A margin past k keeps every tie at the k-th distance before the exact (distance, id) sort.
        part = np.argpartition(scores, min(k + 8, scores.shape[1] - 1), axis=1)[:, :k + 8]
        for row in range(len(block)):
            cand = part[row]
            order = np.lexsort((cand, scores[row, cand]))[:k]
            ids[start + row] = cand[order]
    return ids


def select(hot_dist: np.ndarray, share: np.ndarray, *, count: int, low: float, high: float,
           max_boundary: int | None = None) -> tuple[np.ndarray, dict]:
    """Indices into the pre-filtered candidates: the boundary ones first (a hot-owner share in [low, high],
    at most max_boundary of them), then the fill, each in hot-distance order (ties by index)."""
    order = np.lexsort((np.arange(len(hot_dist)), hot_dist))
    boundary = [int(i) for i in order if low <= share[i] <= high]
    chosen = boundary[:min(count, count if max_boundary is None else max_boundary)]
    taken = set(chosen)
    fill = [int(i) for i in order if i not in taken][:count - len(chosen)]
    if len(chosen) + len(fill) < count:
        raise ValueError(f"only {len(chosen) + len(fill)} candidates for {count} inserts")
    return np.asarray(chosen + fill, dtype=np.int64), {"boundary": len(chosen), "fill": len(fill),
                                                       "boundary_available": len(boundary)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="float32 source vectors (N x D)")
    parser.add_argument("--candidate-start", type=int, required=True)
    parser.add_argument("--candidate-end", type=int, required=True)
    parser.add_argument("--exclude-below", type=int, required=True,
                        help="source rows below this are the runtime's base or another insert set")
    parser.add_argument("--base", type=Path, required=True, help="the runtime's base.npy")
    parser.add_argument("--membership", type=Path, required=True, help="the runtime's membership.npy")
    parser.add_argument("--query", type=Path, required=True, help="the uniform queries")
    parser.add_argument("--query-regions", type=Path, required=True, help="k-means region of each query")
    parser.add_argument("--stream-manifest", type=Path, required=True, help="the skewed stream's MANIFEST.json")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--prefilter", type=int, default=40000)
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--min-hot-share", type=float, default=0.3)
    parser.add_argument("--max-hot-share", type=float, default=0.7)
    parser.add_argument("--max-boundary", type=int, default=None,
                        help="at most this many boundary inserts (default: --count)")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    if not (args.exclude_below <= args.candidate_start < args.candidate_end):
        raise SystemExit("the candidate rows must lie at or above --exclude-below")
    if not (0 < args.count <= args.prefilter and 0 < args.neighbours and
            0.0 <= args.min_hot_share <= args.max_hot_share <= 1.0 and
            (args.max_boundary is None or 0 <= args.max_boundary)):
        raise SystemExit("invalid count, prefilter, neighbours or share bounds")
    source = np.load(args.source, mmap_mode="r", allow_pickle=False)
    base = np.load(args.base, allow_pickle=False)
    membership = np.load(args.membership, allow_pickle=False)
    query = np.load(args.query, allow_pickle=False)
    regions = np.load(args.query_regions, allow_pickle=False)
    stream = json.loads(args.stream_manifest.read_text(encoding="utf-8"))
    hot_owner, hot_regions = int(stream["hot_owner"]), [int(r) for r in stream["hot_regions"]]
    if source.dtype != np.float32 or base.dtype != np.float32 or query.dtype != np.float32 or \
            source.ndim != 2 or base.ndim != 2 or source.shape[1] != base.shape[1] or query.shape[1] != base.shape[1]:
        raise SystemExit("source, base and query must be float32 matrices of one dimension")
    if len(membership) != len(base) or args.candidate_end > len(source):
        raise SystemExit("membership must label every base row; the candidate range must lie in the source")
    if args.exclude_below > len(source) or not np.array_equal(source[:len(base)], base):
        raise SystemExit("the base must be the source's first rows (the candidates are then disjoint from it)")

    candidates = np.asarray(source[args.candidate_start:args.candidate_end])
    centroids = hot_centroids(query, regions, hot_regions)
    distance = hot_distance(candidates, centroids)
    pre = np.lexsort((np.arange(len(distance)), distance))[:args.prefilter]
    neighbours = nearest_base(candidates[pre], base, args.neighbours)
    owners = membership[neighbours]
    share = np.mean(owners == hot_owner, axis=1)
    picked, counts = select(distance[pre], share, count=args.count, low=args.min_hot_share, high=args.max_hot_share,
                            max_boundary=args.max_boundary)
    rng = np.random.default_rng(args.seed)
    picked = picked[rng.permutation(len(picked))]
    rows = args.candidate_start + pre[picked]
    inserts = np.ascontiguousarray(candidates[pre[picked]])
    home = owners[picked, 0]

    args.output.mkdir(parents=True)
    files = {"insert.npy": inserts, "base_plus_insert.npy": np.ascontiguousarray(np.concatenate([base, inserts])),
             "source_rows.npy": rows.astype("<i8")}
    for name, array in files.items():
        np.save(args.output / name, array)
    halves = {f"half_{h}": {"home_owner_counts": np.bincount(home[h * len(home) // 2:(h + 1) * len(home) // 2],
                                                             minlength=int(membership.max()) + 1).tolist()}
              for h in (0, 1)}
    manifest = {
        "schema": SCHEMA,
        "note": "synthetic test data; supports no performance claim",
        "id_contract": "insert row i becomes id N + i when the file is inserted from row 0; base_plus_insert row r "
                       "is id r",
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in (
            ("source", args.source), ("base", args.base), ("membership", args.membership), ("query", args.query),
            ("query_regions", args.query_regions), ("stream_manifest", args.stream_manifest))},
        "parameters": {"candidate_rows": [args.candidate_start, args.candidate_end], "exclude_below": args.exclude_below,
                       "count": args.count, "prefilter": args.prefilter, "neighbours": args.neighbours,
                       "hot_share_bounds": [args.min_hot_share, args.max_hot_share], "seed": args.seed,
                       "max_boundary": args.max_boundary,
                       "hot_owner": hot_owner, "hot_regions": hot_regions},
        "selection": {**counts, "hot_distance_selected": [float(np.min(distance[pre[picked]])),
                                                          float(np.median(distance[pre[picked]])),
                                                          float(np.max(distance[pre[picked]]))],
                      "hot_distance_all_candidates_median": float(np.median(distance)),
                      "hot_owner_share_selected_mean": float(np.mean(share[picked])),
                      "predicted_home_owner_counts": np.bincount(home, minlength=int(membership.max()) + 1).tolist(),
                      **halves},
        "files": {name: {"sha256": sha256_file(args.output / name), "shape": list(array.shape),
                         "dtype": str(array.dtype)} for name, array in files.items()},
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OK", "output": str(args.output), "selection": manifest["selection"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
