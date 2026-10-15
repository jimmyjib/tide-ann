"""Sealing 2, 8 and 16 parts from a 4-part source, checked by the oracle ``verify_sealed_runtime`` in
``synthetic_runtime``; part counts outside 2..64 are refused.
"""
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

from tools.prepare import seal_runtime as sealer
from tools.prepare.runtime_lib import RuntimeErrorChecked, load_global_adjacency, ordered_graph_hash, seal_runtime
from tools.prepare.tests.synthetic_runtime import brute_force_groundtruth, make_runtime, verify_sealed_runtime

NODES = 160
METADATA = {"label": "fixture", "checkpoint": "fixture", "producer_commit": "0" * 40}


class Fixture:
    """A four-part source runtime and the inputs of a seal."""

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
        return ["seal_runtime.py", "--source-runtime", str(self.source), "--membership", str(self.membership(parts)),
                "--base", str(self.base), "--query", str(self.query), "--groundtruth", str(self.groundtruth),
                "--ghost", str(self.ghost), "--output", str(output), "--parts", str(parts), "--metric", "l2",
                "--link-mode", "copy", "--label", "fixture", "--checkpoint", "fixture",
                "--producer-commit", "0" * 40] + list(extra)


def run_cli(argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", argv), redirect_stdout(out), redirect_stderr(err):
        try:
            code = sealer.main()
        except RuntimeErrorChecked as error:
            return 2, str(error)
    return code, out.getvalue()


class SealPartsTests(unittest.TestCase):
    def test_eight_and_sixteen_parts_from_a_four_part_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            source_hash = ordered_graph_hash(fx.data["rows"])
            for parts in (8, 16):
                output = Path(tmp) / f"sealed-{parts}"
                membership = fx.membership(parts)
                manifest = seal_runtime(fx.source, membership, fx.base, fx.query, fx.groundtruth, fx.ghost, output,
                                        parts, "l2", "copy", METADATA, source_parts=4)
                self.assertEqual((manifest["parts"], manifest["source_parts"]), (parts, 4))
                self.assertEqual(manifest["directed_graph_sha256"], source_hash)
                self.assertEqual(manifest["source_graph_sha256"], source_hash)
                self.assertEqual(len(manifest["static_metrics"]["nodes"]), parts)
                self.assertEqual(sum(manifest["static_metrics"]["nodes"]), NODES)
                values = np.load(membership)
                np.testing.assert_array_equal(np.load(output / "membership.npy"), values)
                for part in range(parts):
                    ids = np.load(output / f"partition_{part}" / "local_to_global.npy")
                    np.testing.assert_array_equal(ids, np.flatnonzero(values == part))
                self.assertFalse((output / f"partition_{parts}").exists())
                rows, _ = load_global_adjacency(output, NODES, parts)
                for mine, theirs in zip(rows, fx.data["rows"]):
                    np.testing.assert_array_equal(mine, theirs)
                verified = verify_sealed_runtime(output)
                self.assertEqual(verified["status"], "VERIFIED")
                self.assertEqual(verified["node_counts"], np.bincount(values, minlength=parts).tolist())

    def test_the_source_is_read_with_its_own_part_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            output = Path(tmp) / "sealed"
            with self.assertRaisesRegex(RuntimeErrorChecked, "missing partition payload"):
                seal_runtime(fx.source, fx.membership(8), fx.base, fx.query, fx.groundtruth, fx.ghost, output, 8,
                             "l2", "copy", METADATA)
            self.assertFalse(output.exists())
            # by default, source and output have the same part count
            manifest = seal_runtime(fx.source, fx.membership(4), fx.base, fx.query, fx.groundtruth, fx.ghost, output,
                                    4, "l2", "copy", METADATA)
            self.assertEqual((manifest["parts"], manifest["source_parts"]), (4, 4))
            with self.assertRaisesRegex(RuntimeErrorChecked, "outside partition range"):
                seal_runtime(fx.source, fx.membership(8), fx.base, fx.query, fx.groundtruth, fx.ghost,
                             Path(tmp) / "other", 4, "l2", "copy", METADATA)

    def test_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            code, out = run_cli(fx.argv(Path(tmp) / "eight", 8, "--source-parts", "4"))
            self.assertEqual(code, 0, out)
            self.assertEqual((json.loads(out)["parts"], json.loads(out)["source_parts"]), (8, 4))
            code, out = run_cli(fx.argv(Path(tmp) / "four", 4))
            self.assertEqual(code, 0, out)
            self.assertEqual((json.loads(out)["parts"], json.loads(out)["source_parts"]), (4, 4))
            # K = 2 (and every count in 2..64) seals; the label, checkpoint and commit are optional.
            argv = [item for item in fx.argv(Path(tmp) / "two", 2, "--source-parts", "4")
                    if item not in ("--label", "--checkpoint", "--producer-commit", "fixture", "0" * 40)]
            code, out = run_cli(argv)
            self.assertEqual(code, 0, out)
            self.assertEqual((json.loads(out)["parts"], json.loads(out)["source_parts"]), (2, 4))
            self.assertEqual(verify_sealed_runtime(Path(tmp) / "two")["status"], "VERIFIED")
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
