"""Unit tests for the real-data slice tools: dataset slicing, the skewed stream and the skewed prep plan."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import build_skewed_query_stream as skew
from tools.prepare import prepare_demo_runtime as prep
from tools.prepare import slice_dataset as sl


def blobs(seed: int = 5, per: int = 200, centres: int = 8, dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Eight unit-noise blobs centred 50 apart on the coordinate axes (k-means recovers them)."""
    rng = np.random.default_rng(seed)
    middle = 50.0 * np.eye(centres, dim)
    labels = np.repeat(np.arange(centres), per)
    return (middle[labels] + rng.normal(0.0, 1.0, (len(labels), dim))).astype(np.float32), labels


class SliceTest(unittest.TestCase):
    def test_rows_and_contract(self) -> None:
        source = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
        query = np.ones((5, 3), dtype=np.float32)
        out = sl.slice_arrays(source, query, base_rows=20, insert_start=20, insert_rows=8)
        self.assertTrue(np.array_equal(out["base.npy"], source[:20]))
        self.assertTrue(np.array_equal(out["insert.npy"], source[20:28]))
        self.assertTrue(np.array_equal(out["base_plus_insert.npy"], source[:28]))
        self.assertTrue(np.array_equal(out["query.npy"], query))

    def test_refusals(self) -> None:
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

    def test_main_writes_manifest_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, qry = Path(tmp) / "base.npy", Path(tmp) / "q.npy"
            np.save(src, np.arange(30 * 4, dtype=np.float32).reshape(30, 4))
            np.save(qry, np.ones((3, 4), dtype=np.float32))
            out = Path(tmp) / "source"
            argv = ["--source-base", str(src), "--query", str(qry), "--base-rows", "10", "--insert-start", "20",
                    "--insert-rows", "5", "--label", "t", "--output", str(out)]
            self.assertEqual(sl.main(argv), 0)
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(manifest["rows"]["insert"], [20, 25])
            for name, record in manifest["files"].items():
                self.assertEqual(record["sha256"], sl.sha256_file(out / name), name)
            self.assertEqual(np.load(out / "base_plus_insert.npy").shape, (15, 4))
            self.assertEqual(sl.main(argv), 2)


class SkewedStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query, self.labels = blobs()
        # Owner = blob id modulo 4; ground truth top-10 = ten ids of the same blob.
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
        # Both blobs of the hot owner are hot, so the rest rows never reach it: share exactly 0.7.
        share = result["proxy"]["stream"]["top1_owner_share"][result["hot_owner"]]
        self.assertAlmostEqual(share, 0.7, places=6)
        self.assertGreater(result["proxy"]["stream"]["top1_max_over_mean"], 1.2)
        self.assertLess(abs(result["proxy"]["native"]["top1_max_over_mean"] - 1.0), 0.01)

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

    def test_refusals(self) -> None:
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
                    "--output", str(out), "--label", "t"]
            self.assertEqual(skew.main(argv), 0)
            source = np.load(out / "source_rows.npy")
            self.assertTrue(np.array_equal(np.load(out / "query.npy"), self.query[source]))
            self.assertTrue(np.array_equal(np.load(out / "groundtruth.npy"), self.truth[source]))
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(manifest["hot_rows"], 700)
            self.assertEqual(skew.main(argv), 2)


class SkewedPlanTest(unittest.TestCase):
    def test_skewed_stream_step(self) -> None:
        root = Path("/data/demo-m")
        steps = dict(prep.build_steps(root, "/py", n_nodes=200_000, producer_commit="abc", name="demo-m",
                                      checkpoint="demo-m", stream_mode="skewed", hot_share=0.7))
        stream = steps["stream"]
        self.assertTrue(stream[2].endswith("build_skewed_query_stream.py"))
        self.assertEqual(stream[stream.index("--membership") + 1], "/data/demo-m/runtime/membership.npy")
        self.assertEqual(stream[stream.index("--rows") + 1], "262144")
        self.assertEqual(stream[stream.index("--hot-share") + 1], "0.7")
        self.assertEqual(steps["seal"][steps["seal"].index("--checkpoint") + 1], "demo-m")
        self.assertTrue((prep.TOOLS / Path(stream[2]).name).is_file())

    def test_default_plan_is_demo_s(self) -> None:
        steps = dict(prep.build_steps(Path("/d"), "/py", n_nodes=20_000, producer_commit="abc"))
        self.assertEqual(steps["stream"][steps["stream"].index("--label") + 1], "demo-s-native-x64")
        self.assertEqual(steps["seal"][steps["seal"].index("--checkpoint") + 1], "demo-s")
        self.assertTrue(steps["stream"][2].endswith("build_query_load_stream.py"))


if __name__ == "__main__":
    unittest.main()
