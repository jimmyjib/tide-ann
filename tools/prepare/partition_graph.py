#!/usr/bin/env python3
"""METIS-partition a global directed DiskANN graph without changing its rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.prepare.graph_csr import (CsrScaleError, csr_static_metrics, metis_int32_arrays, read_global_graph_csr,
                                     symmetrize_csr, write_adj_csr)
from tools.prepare.runtime_lib import PrepareError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=4, help="partitions, one per index server (2..64)")
    args = parser.parse_args(argv)
    if not 2 <= args.parts <= 64:
        raise PrepareError("partitioning requires 2..64 parts")
    if args.output.exists():
        raise PrepareError("output already exists")
    base = np.load(args.base, mmap_mode="r")
    if base.dtype != np.float32 or base.ndim != 2:
        raise PrepareError("invalid base")
    try:
        import pymetis
    except ImportError as error:
        raise PrepareError("pymetis is required (pip install pymetis)") from error
    try:
        indptr, indices, start, max_degree = read_global_graph_csr(args.graph, len(base))
        uindptr, uindices = symmetrize_csr(indptr, indices)
        xadj, adjncy = metis_int32_arrays(uindptr, uindices)
    except CsrScaleError as error:
        raise PrepareError(str(error)) from error
    del uindptr, uindices
    cuts, raw_membership = pymetis.part_graph(args.parts, xadj=xadj, adjncy=adjncy)
    del xadj, adjncy
    membership = np.asarray(raw_membership, dtype=np.int32)
    if membership.shape != (len(base),) or set(membership.tolist()) != set(range(args.parts)):
        raise PrepareError("METIS returned an empty or invalid partition")
    args.output.mkdir(parents=True)
    np.save(args.output / "membership.npy", membership)
    for part in range(args.parts):
        directory = args.output / f"partition_{part}"
        directory.mkdir()
        ids = np.flatnonzero(membership == part).astype(np.uint32)
        np.save(directory / "local_to_global.npy", ids)
        local_start = int(np.searchsorted(ids, start)) if membership[start] == part else 0
        write_adj_csr(directory / "adj", indptr, indices, ids, local_start, max_degree)
    manifest = {"N": int(len(base)), "parts": args.parts, "metis_edge_cuts": int(cuts),
                "static_metrics": csr_static_metrics(indptr, indices, membership, args.parts)}
    (args.output / "PARTITION_BUILD.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(f"wrote {args.output} ({args.parts} parts, {int(cuts)} edge cuts)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepareError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
