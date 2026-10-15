#!/usr/bin/env python3
"""Sample head-index node ids, the "ghost" sample (--ratio of the nodes, fixed --seed)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.prepare.runtime_lib import PrepareError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.nodes <= 0 or not 0 < args.ratio <= 1 or args.output.exists():
        raise PrepareError("invalid --nodes/--ratio, or --output exists")
    count = max(1, int(args.nodes * args.ratio))
    ids = np.sort(np.random.RandomState(args.seed).choice(args.nodes, count, replace=False).astype(np.int64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, ids)
    print(f"wrote {args.output} ({count} ids)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepareError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
