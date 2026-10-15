"""Tests for slice_dataset, build_skewed_query_stream and the skewed stream step of prepare_runtime."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import build_skewed_query_stream as skew
from tools.prepare import prepare_runtime as prep
from tools.prepare import slice_dataset as sl


def quiet(main, argv: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(argv)


def blobs(seed: int = 5, per: int = 200, centers: int = 8, dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Eight unit-noise blobs centered 50 apart on the coordinate axes (k-means recovers them)."""
    rng = np.random.default_rng(seed)
    middle = 50.0 * np.eye(centers, dim)
    labels = np.repeat(np.arange(centers), per)
    return (middle[labels] + rng.normal(0.0, 1.0, (len(labels), dim))).astype(np.float32), labels


class SliceTest(unittest.TestCase):
    def test_rows(self) -> None:
        source = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
        query = np.ones((5, 3), dtype=np.float32)
        out = sl.slice_arrays(source, query, base_rows=20, insert_start=20, insert_rows=8)
        self.assertTrue(np.array_equal(out["base.npy"], source[:20]))
        self.assertTrue(np.array_equal(out["insert.npy"], source[20:28]))
        self.assertTrue(np.array_equal(out["base_plus_insert.npy"], source[:28]))
        self.assertTrue(np.array_equal(out["query.npy"], query))

    def test_invalid_arguments(self) -> None:
        source = np.zeros((40, 3), dtype=np.float32)
        query = np.zeros((5, 3), dtype=np.float32)
        for kwargs in ({"base_rows": 20, "insert_start": 10, "insert_rows": 5},
                       {"base_rows": 20, "insert_start": 30, "insert_rows": 20},
                       {"base_rows": 0, "insert_start": 0, "insert_rows": 5}):
            with self.assertRaises(ValueError):
                sl.slice_arrays(source, query, **kwargs)
        with self.assertRaises(ValueError):
            sl.slice_arrays(source, np.zeros((5, 4), dtype=np.float32), base_rows=20, insert_start=20,
                            insert_rows=5)
        with self.assertRaises(ValueError):
            sl.slice_arrays(source.astype(np.float64), query, base_rows=20, insert_start=20, insert_rows=5)

    def test_main_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, qry = Path(tmp) / "base.npy", Path(tmp) / "q.npy"
            np.save(src, np.arange(30 * 4, dtype=np.float32).reshape(30, 4))
            np.save(qry, np.ones((3, 4), dtype=np.float32))
            out = Path(tmp) / "source"
            argv = ["--source-base", str(src), "--query", str(qry), "--base-rows", "10", "--insert-start", "20",
                    "--insert-rows", "5", "--output", str(out)]
            self.assertEqual(quiet(sl.main, argv), 0)
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual((manifest["base_rows"], manifest["insert_start"], manifest["insert_rows"]), (10, 20, 5))
            self.assertEqual(manifest["shapes"]["base_plus_insert.npy"], [15, 4])
            self.assertEqual(np.load(out / "base_plus_insert.npy").shape, (15, 4))
            self.assertEqual(quiet(sl.main, argv), 2)


class SkewedStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query, self.labels = blobs()
        # Owner = blob id modulo 4. The ground-truth top 10 are ten ids of the same blob.
        self.membership = (np.arange(1600) // 200 % 4).astype(np.int32)
        self.truth = np.stack([np.arange(10) + (label * 200) for label in self.labels]).astype(np.uint32)

    def build(self, **overrides) -> dict:
        kwargs = dict(rows=5000, hot_share=0.7, regions=8, hot_regions=2, iterations=25, seed=11)
        kwargs.update(overrides)
        return skew.build(self.query, self.truth, self.membership, **kwargs)

    def test_hot_rows_from_one_owner(self) -> None:
        result = self.build()
        source = result["source_rows"]
        self.assertEqual(len(source), 5000)
        self.assertEqual(len(result["hot_regions"]), 2)
        hot = np.isin(result["labels"][source], result["hot_regions"])
        self.assertEqual(int(hot.sum()), 3500)
        owners = self.membership[self.truth[source, 0]]
        self.assertTrue(np.all(owners[hot] == result["hot_owner"]))
        self.assertEqual(sorted(result["hot_regions"]), sorted(set(result["labels"][source[hot]].tolist())))
        # Both blobs of the hot owner are hot, so the other rows never reach it: its share is 0.7.
        share = np.bincount(owners, minlength=4) / len(owners)
        self.assertAlmostEqual(float(share[result["hot_owner"]]), 0.7, places=6)
        self.assertGreater(float(share.max()) * 4, 1.2)
        everyone = np.bincount(self.membership[self.truth[:, 0]], minlength=4) / len(self.truth)
        self.assertLess(abs(float(everyone.max()) * 4 - 1.0), 0.01)

    def test_deterministic_and_seeded(self) -> None:
        first, second = self.build(), self.build()
        self.assertTrue(np.array_equal(first["source_rows"], second["source_rows"]))
        other = self.build(seed=12)
        self.assertFalse(np.array_equal(first["source_rows"], other["source_rows"]))

    def test_every_pool_query_used_evenly(self) -> None:
        source = self.build()["source_rows"]
        counts = np.bincount(source, minlength=len(self.query))
        hot = counts[counts > counts.mean()]
        self.assertLessEqual(int(hot.max() - hot.min()), 1)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self.build(hot_share=1.0)
        with self.assertRaises(ValueError):
            self.build(hot_regions=3)  # only two of the eight blobs belong to any owner
        with self.assertRaises(ValueError):
            skew.build(self.query, self.truth[:, :5], self.membership, rows=10, hot_share=0.5, regions=8,
                       hot_regions=1, iterations=5, seed=1)

    def test_main_outputs_align(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, array in (("q.npy", self.query), ("gt.npy", self.truth), ("m.npy", self.membership)):
                np.save(root / name, array)
            out = root / "stream"
            argv = ["--query", str(root / "q.npy"), "--groundtruth", str(root / "gt.npy"), "--membership",
                    str(root / "m.npy"), "--rows", "1000", "--regions", "8", "--hot-regions", "2",
                    "--output", str(out)]
            self.assertEqual(quiet(skew.main, argv), 0)
            source = np.load(out / "source_rows.npy")
            self.assertTrue(np.array_equal(np.load(out / "query.npy"), self.query[source]))
            self.assertTrue(np.array_equal(np.load(out / "groundtruth.npy"), self.truth[source]))
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(sorted(manifest), ["hot_owner", "hot_regions", "hot_rows", "parameters", "query_rows",
                                                "regions", "stream_rows"])
            self.assertEqual((manifest["query_rows"], manifest["stream_rows"], manifest["hot_rows"]), (1600, 1000, 700))
            self.assertEqual(quiet(skew.main, argv), 2)


class SkewedPlanTest(unittest.TestCase):
    def test_skewed_stream_step(self) -> None:
        root = Path("/data/sliced")
        steps = dict(prep.build_steps(root, "/py", n_nodes=200_000, stream_mode="skewed", hot_share=0.7))
        stream = steps["stream"]
        self.assertTrue(stream[2].endswith("build_skewed_query_stream.py"))
        self.assertEqual(stream[stream.index("--membership") + 1], "/data/sliced/runtime/membership.npy")
        self.assertEqual(stream[stream.index("--rows") + 1], "262144")
        self.assertEqual(stream[stream.index("--hot-share") + 1], "0.7")
        self.assertTrue((prep.TOOLS / Path(stream[2]).name).is_file())

    def test_default_plan(self) -> None:
        steps = dict(prep.build_steps(Path("/d"), "/py", n_nodes=20_000))
        self.assertEqual(steps["stream"], ["/py", "-B", str(prep.TOOLS / "build_query_load_stream.py"),
                                           "--query", "/d/runtime/query.npy", "--groundtruth",
                                           "/d/runtime/groundtruth.npy", "--repeats", "64", "--seed", "2",
                                           "--output", "/d/stream"])


if __name__ == "__main__":
    unittest.main()
