#!/usr/bin/env python3
"""Create one deterministic head-ID sample (the head index) shared by every runtime built from the same base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.prepare.runtime_lib import RuntimeErrorChecked, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.nodes <= 0 or not 0 < args.ratio <= 1 or args.output.exists():
        raise RuntimeErrorChecked("invalid ghost-ID request")
    count = max(1, int(args.nodes * args.ratio))
    ids = np.sort(np.random.RandomState(args.seed).choice(args.nodes, count, replace=False).astype(np.int64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, ids)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps({
        "schema": {"name": "tide-repartition-ghost-ids", "version": 1},
        "N": args.nodes, "count": count, "ratio": args.ratio, "seed": args.seed,
        "sha256": sha256(args.output),
    }, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "count": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeErrorChecked as error:
        print(f"GHOST-ID CREATION FAILED: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
