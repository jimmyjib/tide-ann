#!/usr/bin/env python3
"""Seal a complete repartitioned runtime while preserving directed adjacency."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.prepare.runtime_lib import RuntimeErrorChecked, seal_runtime

# The server counts the servers accept (one index server per part), as partition_graph.py: 2..64.
MIN_PARTS, MAX_PARTS = 2, 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--ghost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=4, help="part count of the sealed runtime")
    parser.add_argument("--source-parts", type=int, default=None,
                        help="part count of --source-runtime, which is read for its directed graph only "
                             "(default: --parts)")
    parser.add_argument("--metric", choices=("l2", "cosine"), default="l2")
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--label", default="", help="free-form label recorded in the runtime manifest")
    parser.add_argument("--checkpoint", default="", help="free-form checkpoint name recorded in the manifest")
    parser.add_argument("--producer-commit", default="unknown",
                        help="source revision recorded in the manifest (default: unknown)")
    args = parser.parse_args()
    source_parts = args.parts if args.source_parts is None else args.source_parts
    if not MIN_PARTS <= args.parts <= MAX_PARTS or not 1 <= source_parts <= MAX_PARTS:
        raise RuntimeErrorChecked(f"part counts must be in {MIN_PARTS}..{MAX_PARTS} (the source runtime's in "
                                  f"1..{MAX_PARTS}); got {args.parts} and {source_parts}")
    manifest = seal_runtime(args.source_runtime, args.membership, args.base, args.query, args.groundtruth,
                            args.ghost, args.output, args.parts, args.metric, args.link_mode,
                            {"label": args.label, "checkpoint": args.checkpoint,
                             "producer_commit": args.producer_commit},
                            source_parts=source_parts)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "parts": manifest["parts"],
                      "source_parts": manifest["source_parts"],
                      "directed_graph_sha256": manifest["directed_graph_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeErrorChecked as error:
        print(f"RUNTIME SEAL FAILED: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
