#!/usr/bin/env python3
"""Seeded synthetic dataset: base, query and insert vectors from Gaussian clusters.

Each cluster lies near its own low-dimensional random subspace, with unequal spreads and sizes.
round(--hot-share * Q) queries come from --hot-clusters and round(--insert-share * M) inserts from
--insert-cluster. The rest come from the other clusters by size. Base, queries and inserts use
separate random streams: query options do not change the base, and insert options change neither the base nor
the queries.
Writes the vector files, base_plus_insert.npy, labels and MANIFEST.json into a new directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    n_base: int = 20_000
    dim: int = 64
    clusters: int = 32
    intrinsic_dim: int = 12
    center_scale: float = 10.0
    min_spread: float = 1.0
    max_spread: float = 4.0
    noise: float = 0.1
    size_concentration: float = 8.0
    n_query: int = 4_096
    hot_clusters: tuple[int, ...] = field(default=(0, 1, 2))
    hot_share: float = 0.70
    n_insert: int = 4_000
    insert_cluster: int = 3
    insert_share: float = 0.50
    seed: int = 1

    def validate(self) -> None:
        if min(self.n_base, self.dim, self.clusters, self.intrinsic_dim, self.n_query, self.n_insert) < 1:
            raise ValueError("all counts and dimensions must be positive")
        if self.intrinsic_dim > self.dim:
            raise ValueError("intrinsic_dim cannot exceed dim")
        if self.n_base < self.clusters:
            raise ValueError("n_base must be at least the cluster count")
        if not (0 < self.min_spread <= self.max_spread) or self.center_scale <= 0 or self.noise < 0:
            raise ValueError("invalid spread, center scale or noise")
        if self.size_concentration <= 0:
            raise ValueError("size_concentration must be positive")
        if not self.hot_clusters or len(set(self.hot_clusters)) != len(self.hot_clusters):
            raise ValueError("hot_clusters must be a non-empty list of distinct cluster ids")
        if any(not 0 <= c < self.clusters for c in (*self.hot_clusters, self.insert_cluster)):
            raise ValueError("hot/insert cluster id outside the cluster range")
        if not 0.0 <= self.hot_share <= 1.0 or not 0.0 <= self.insert_share <= 1.0:
            raise ValueError("shares must lie in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def _cluster_sizes(rng: np.random.Generator, config: Config, weights: np.ndarray) -> np.ndarray:
    """Every cluster gets one row first, the rest follow the Dirichlet weights."""
    sizes = np.ones(config.clusters, dtype=np.int64)
    sizes += rng.multinomial(config.n_base - config.clusters, weights)
    return sizes


def _sample(rng: np.random.Generator, labels: np.ndarray, geometry: dict) -> np.ndarray:
    centers, bases, spreads, axis_scale = (geometry["centers"], geometry["bases"],
                                           geometry["spreads"], geometry["axis_scale"])
    scale = spreads[labels][:, None]
    # Draw every random number for all rows first, so the stream does not depend on grouping.
    coeff = rng.standard_normal((len(labels), len(axis_scale))) * axis_scale[None, :] * scale
    noise = rng.standard_normal((len(labels), centers.shape[1])) * (geometry["noise"] * scale)
    vectors = centers[labels] + noise
    for cluster in np.unique(labels):
        rows = labels == cluster
        vectors[rows] += coeff[rows] @ bases[cluster]
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _mixed_labels(rng: np.random.Generator, count: int, share: float, preferred: np.ndarray,
                  clusters: int, weights: np.ndarray) -> np.ndarray:
    """round(share * count) rows from the preferred clusters, the rest from the others by size weight."""
    n_pref = int(round(share * count))
    others = np.setdiff1d(np.arange(clusters), preferred)
    if n_pref < count and not len(others):
        raise ValueError("a share below 1 needs at least one non-preferred cluster")
    pref = rng.choice(preferred, size=n_pref, replace=True) if n_pref else np.zeros(0, dtype=np.int64)
    rest = (rng.choice(others, size=count - n_pref, replace=True, p=weights[others] / weights[others].sum())
            if n_pref < count else np.zeros(0, dtype=np.int64))
    labels = np.concatenate([np.asarray(pref, dtype=np.int64), np.asarray(rest, dtype=np.int64)])
    return labels[rng.permutation(count)]


def generate(config: Config) -> dict[str, np.ndarray]:
    """Arrays of one dataset, deterministic for a config."""
    config.validate()
    geometry_ss, base_ss, query_ss, insert_ss = np.random.SeedSequence(config.seed).spawn(4)
    geo = np.random.Generator(np.random.PCG64(geometry_ss))
    centers = geo.normal(0.0, config.center_scale, size=(config.clusters, config.dim))
    spreads = np.exp(geo.uniform(np.log(config.min_spread), np.log(config.max_spread), size=config.clusters))
    bases = np.empty((config.clusters, config.intrinsic_dim, config.dim))
    for cluster in range(config.clusters):
        q, _ = np.linalg.qr(geo.standard_normal((config.dim, config.intrinsic_dim)))
        bases[cluster] = q.T
    weights = geo.dirichlet(np.full(config.clusters, config.size_concentration))
    geometry = {"centers": centers, "bases": bases, "spreads": spreads, "noise": config.noise,
                "axis_scale": np.linspace(1.0, 0.3, config.intrinsic_dim)}

    rng = np.random.Generator(np.random.PCG64(base_ss))
    sizes = _cluster_sizes(rng, config, weights)
    base_labels = np.repeat(np.arange(config.clusters, dtype=np.int64), sizes)
    base_labels = base_labels[rng.permutation(config.n_base)]
    base = _sample(rng, base_labels, geometry)

    rng = np.random.Generator(np.random.PCG64(query_ss))
    query_labels = _mixed_labels(rng, config.n_query, config.hot_share,
                                 np.asarray(config.hot_clusters, dtype=np.int64), config.clusters, weights)
    query = _sample(rng, query_labels, geometry)

    rng = np.random.Generator(np.random.PCG64(insert_ss))
    insert_labels = _mixed_labels(rng, config.n_insert, config.insert_share,
                                  np.asarray([config.insert_cluster], dtype=np.int64), config.clusters, weights)
    insert = _sample(rng, insert_labels, geometry)

    for name, array in (("base", base), ("query", query), ("insert", insert)):
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite {name} vector")
    return {"base": base, "query": query, "insert": insert,
            "base_labels": base_labels.astype(np.int32), "query_labels": query_labels.astype(np.int32),
            "insert_labels": insert_labels.astype(np.int32),
            "cluster_sizes": sizes, "cluster_spreads": spreads, "cluster_weights": weights}


def summary(config: Config, data: dict[str, np.ndarray]) -> dict:
    hot = np.isin(data["query_labels"], np.asarray(config.hot_clusters))
    return {
        "cluster_sizes": [int(x) for x in data["cluster_sizes"]],
        "cluster_spreads": [round(float(x), 6) for x in data["cluster_spreads"]],
        "query_hot_rows": int(hot.sum()),
        "query_hot_share": float(hot.mean()),
        "query_rows_per_hot_cluster": {str(c): int(np.sum(data["query_labels"] == c))
                                       for c in config.hot_clusters},
        "insert_cluster_rows": int(np.sum(data["insert_labels"] == config.insert_cluster)),
        "insert_cluster_share": float(np.mean(data["insert_labels"] == config.insert_cluster)),
    }


def write_dataset(config: Config, output: Path) -> dict:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists")
    data = generate(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        files = {
            "base.npy": data["base"], "query.npy": data["query"], "insert.npy": data["insert"],
            "base_plus_insert.npy": np.ascontiguousarray(np.concatenate([data["base"], data["insert"]])),
            "base_labels.npy": data["base_labels"], "query_labels.npy": data["query_labels"],
            "insert_labels.npy": data["insert_labels"],
        }
        for name, array in files.items():
            np.save(staging / name, array)
        manifest = {
            "config": {**asdict(config), "hot_clusters": list(config.hot_clusters)},
            "numpy": np.__version__,
            "shapes": {name: list(array.shape) for name, array in files.items()},
            "summary": summary(config, data),
        }
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_cluster_list(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in text.split(",") if item.strip() != "")
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid cluster list {text!r}") from error


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Write a seeded synthetic dataset of Gaussian clusters.")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the dataset")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n-base", type=int, default=defaults.n_base)
    parser.add_argument("--dim", type=int, default=defaults.dim)
    parser.add_argument("--clusters", type=int, default=defaults.clusters)
    parser.add_argument("--intrinsic-dim", type=int, default=defaults.intrinsic_dim)
    parser.add_argument("--center-scale", type=float, default=defaults.center_scale)
    parser.add_argument("--min-spread", type=float, default=defaults.min_spread)
    parser.add_argument("--max-spread", type=float, default=defaults.max_spread)
    parser.add_argument("--noise", type=float, default=defaults.noise)
    parser.add_argument("--size-concentration", type=float, default=defaults.size_concentration)
    parser.add_argument("--n-query", type=int, default=defaults.n_query)
    parser.add_argument("--hot-clusters", type=parse_cluster_list,
                        default=defaults.hot_clusters, help="comma-separated cluster ids (default 0,1,2)")
    parser.add_argument("--hot-share", type=float, default=defaults.hot_share)
    parser.add_argument("--n-insert", type=int, default=defaults.n_insert)
    parser.add_argument("--insert-cluster", type=int, default=defaults.insert_cluster)
    parser.add_argument("--insert-share", type=float, default=defaults.insert_share)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(n_base=args.n_base, dim=args.dim, clusters=args.clusters, intrinsic_dim=args.intrinsic_dim,
                  center_scale=args.center_scale, min_spread=args.min_spread, max_spread=args.max_spread,
                  noise=args.noise, size_concentration=args.size_concentration, n_query=args.n_query,
                  hot_clusters=tuple(args.hot_clusters), hot_share=args.hot_share, n_insert=args.n_insert,
                  insert_cluster=args.insert_cluster, insert_share=args.insert_share, seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        manifest = write_dataset(config, args.output)
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    base = manifest["shapes"]["base.npy"]
    print(f"wrote {args.output} (base {base[0]} x {base[1]}, {manifest['shapes']['query.npy'][0]} queries, "
          f"{manifest['shapes']['insert.npy'][0]} inserts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
