#!/usr/bin/env python3
"""Write a runtime directory for a membership: base, queries, ground truth, head ids and one adj file
per partition."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.prepare.runtime_lib import PrepareError, write_runtime

# same range as partition_graph.py
MIN_PARTS, MAX_PARTS = 2, 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", type=Path, required=True,
                        help="partitioned graph to read: a runtime or partition_graph.py output")
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--ghost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=4, help="number of partitions")
    parser.add_argument("--source-parts", type=int, default=None,
                        help="partitions of --source-runtime (default: --parts)")
    parser.add_argument("--metric", choices=("l2", "cosine"), default="l2")
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    args = parser.parse_args()
    source_parts = args.parts if args.source_parts is None else args.source_parts
    if not MIN_PARTS <= args.parts <= MAX_PARTS or not 1 <= source_parts <= MAX_PARTS:
        raise PrepareError(f"part counts must be in {MIN_PARTS}..{MAX_PARTS} (source runtime: "
                           f"1..{MAX_PARTS}), got {args.parts} and {source_parts}")
    manifest = write_runtime(args.source_runtime, args.membership, args.base, args.query, args.groundtruth,
                             args.ghost, args.output, args.parts, args.metric, args.link_mode,
                             source_parts=source_parts)
    print(f"wrote {args.output} ({manifest['parts']} partitions from {manifest['source_parts']})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepareError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
