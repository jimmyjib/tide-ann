"""Tests for write_runtime: 2, 8 and 16 parts from a 4-part source, and part counts outside 2..64."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from tools.prepare import write_runtime as cli
from tools.prepare.runtime_lib import PrepareError, load_global_adjacency, read_adj, write_runtime
from tools.prepare.tests.synthetic_runtime import (
    brute_force_groundtruth, make_runtime, static_metrics, verify_runtime, write_adj)

NODES = 160


class Fixture:
    """A four-part source runtime and the other inputs of write_runtime."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source"
        self.data = make_runtime(self.source, n=NODES, dim=5, seed=3, parts=4, ghost_count=6)
        rng = self.data["rng"]
        self.query = root / "query.npy"
        queries = rng.normal(size=(7, 5)).astype(np.float32)
        np.save(self.query, queries)
        self.groundtruth = root / "groundtruth.npy"
        np.save(self.groundtruth, brute_force_groundtruth(self.data["base"], queries))
        self.base = self.source / "base.npy"
        self.ghost = self.source / "ghost" / "ghost_indices.npy"

    def membership(self, parts: int, seed: int = 5) -> Path:
        rng = np.random.default_rng(seed)
        values = np.arange(NODES) % parts
        rng.shuffle(values)
        path = self.root / f"membership_{parts}_{seed}.npy"
        np.save(path, values.astype(np.int32))
        return path

    def argv(self, output: Path, parts: int, *extra: str) -> list[str]:
        return ["write_runtime.py", "--source-runtime", str(self.source), "--membership", str(self.membership(parts)),
                "--base", str(self.base), "--query", str(self.query), "--groundtruth", str(self.groundtruth),
                "--ghost", str(self.ghost), "--output", str(output), "--parts", str(parts), "--metric", "l2",
                "--link-mode", "copy"] + list(extra)


def run_cli(argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", argv), redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main()
        except PrepareError as error:
            return 2, str(error)
    return code, out.getvalue()


class WriteRuntimeTests(unittest.TestCase):
    def test_eight_and_sixteen_parts_from_a_four_part_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            for parts in (8, 16):
                output = Path(tmp) / f"runtime-{parts}"
                membership = fx.membership(parts)
                manifest = write_runtime(fx.source, membership, fx.base, fx.query, fx.groundtruth, fx.ghost, output,
                                         parts, "l2", "copy", source_parts=4)
                self.assertEqual((manifest["parts"], manifest["source_parts"]), (parts, 4))
                self.assertEqual(json.loads((output / "RUNTIME_MANIFEST.json").read_text()), manifest)
                self.assertEqual({path.name for path in output.iterdir()},
                                 {"RUNTIME_MANIFEST.json", "base.npy", "query.npy", "groundtruth.npy", "ghost",
                                  "membership.npy", *(f"partition_{part}" for part in range(parts))})
                self.assertEqual(len(manifest["static_metrics"]["nodes"]), parts)
                self.assertEqual(sum(manifest["static_metrics"]["nodes"]), NODES)
                values = np.load(membership)
                np.testing.assert_array_equal(np.load(output / "membership.npy"), values)
                self.assertEqual(manifest["static_metrics"], static_metrics(fx.data["rows"], values, parts))
                for part in range(parts):
                    ids = np.load(output / f"partition_{part}" / "local_to_global.npy")
                    np.testing.assert_array_equal(ids, np.flatnonzero(values == part))
                    adj = output / f"partition_{part}" / "adj"
                    _, start, _ = read_adj(adj)
                    rows = [fx.data["rows"][int(gid)] for gid in ids]
                    write_adj(Path(tmp) / "reference-adj", rows, start, max(len(row) for row in rows))
                    self.assertEqual(adj.read_bytes(), (Path(tmp) / "reference-adj").read_bytes(), part)
                self.assertFalse((output / f"partition_{parts}").exists())
                rows, _ = load_global_adjacency(output, NODES, parts)
                for mine, theirs in zip(rows, fx.data["rows"]):
                    np.testing.assert_array_equal(mine, theirs)
                verified = verify_runtime(output, fx.data["rows"])
                self.assertEqual(verified["node_counts"], np.bincount(values, minlength=parts).tolist())

    def test_the_source_is_read_with_its_own_part_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            output = Path(tmp) / "runtime"
            with self.assertRaisesRegex(PrepareError, "missing partition payload"):
                write_runtime(fx.source, fx.membership(8), fx.base, fx.query, fx.groundtruth, fx.ghost, output, 8,
                              "l2", "copy")
            self.assertFalse(output.exists())
            # by default, source and output have the same part count
            manifest = write_runtime(fx.source, fx.membership(4), fx.base, fx.query, fx.groundtruth, fx.ghost,
                                     output, 4, "l2", "copy")
            self.assertEqual((manifest["parts"], manifest["source_parts"]), (4, 4))
            with self.assertRaisesRegex(PrepareError, "already exists"):
                write_runtime(fx.source, fx.membership(4), fx.base, fx.query, fx.groundtruth, fx.ghost, output, 4,
                              "l2", "copy")
            with self.assertRaisesRegex(PrepareError, "outside partition range"):
                write_runtime(fx.source, fx.membership(8), fx.base, fx.query, fx.groundtruth, fx.ghost,
                              Path(tmp) / "other", 4, "l2", "copy")

    def test_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            for name, parts, extra in (("eight", 8, ["--source-parts", "4"]), ("four", 4, []),
                                       ("two", 2, ["--source-parts", "4"])):
                output = Path(tmp) / name
                code, out = run_cli(fx.argv(output, parts, *extra))
                self.assertEqual(code, 0, out)
                manifest = json.loads((output / "RUNTIME_MANIFEST.json").read_text())
                self.assertEqual((manifest["parts"], manifest["source_parts"]), (parts, 4))
                self.assertEqual(out, f"wrote {output} ({parts} partitions from 4)\n")
            self.assertEqual(verify_runtime(Path(tmp) / "two", fx.data["rows"])["N"], NODES)
            for parts in (1, 65):
                code, out = run_cli(fx.argv(Path(tmp) / f"bad-{parts}", parts, "--source-parts", "4"))
                self.assertEqual(code, 2)
                self.assertIn("part counts must be in 2..64", out)
                self.assertFalse((Path(tmp) / f"bad-{parts}").exists())
            code, out = run_cli(fx.argv(Path(tmp) / "bad-source", 8, "--source-parts", "5"))
            self.assertEqual(code, 2)
            self.assertIn("missing partition payload", out)
            self.assertFalse((Path(tmp) / "bad-source").exists())


if __name__ == "__main__":
    unittest.main()
