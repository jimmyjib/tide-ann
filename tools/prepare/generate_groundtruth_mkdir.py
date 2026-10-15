"""Create the output directory, then run generate_groundtruth.py with the same argv.

generate_groundtruth.py writes `<output>.tmp-<pid>` beside its target and never creates the
directory. This wrapper makes that directory and delegates the unchanged argument vector to
generate_groundtruth.py, so the file is still written by that producer.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PRODUCER = Path(__file__).resolve().parent / "generate_groundtruth.py"


def main() -> int:
    argv = sys.argv[1:]
    if "--output" not in argv:
        raise SystemExit("--output is required")
    Path(argv[argv.index("--output") + 1]).parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(PRODUCER), *argv]
    runpy.run_path(str(PRODUCER), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
