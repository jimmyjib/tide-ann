"""Unit tests of the biased insert set (make_biased_inserts.py)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import make_biased_inserts as mb


class SelectionTest(unittest.TestCase):
    def test_nearest_base_ties_by_id(self) -> None:
        base = np.array([[0, 0], [1, 0], [1, 0], [3, 0], [0, 2]], dtype=np.float32)
        ids = mb.nearest_base(np.array([[1, 0], [0, 0]], dtype=np.float32), base, 3)
        self.assertEqual(ids.tolist(), [[1, 2, 0], [0, 1, 2]])

    def test_hot_distance_and_centroids(self) -> None:
        query = np.array([[0, 0], [2, 0], [10, 10], [12, 10]], dtype=np.float32)
        centroids = mb.hot_centroids(query, np.array([5, 5, 7, 7]), [7, 5])
        self.assertTrue(np.allclose(centroids, [[11, 10], [1, 0]]))
        distance = mb.hot_distance(np.array([[1, 0], [11, 11]], dtype=np.float32), centroids)
        self.assertTrue(np.allclose(distance, [0.0, 1.0]))
        with self.assertRaises(ValueError):
            mb.hot_centroids(query, np.array([5, 5, 7, 7]), [9])

    def test_select_boundary_then_fill(self) -> None:
        distance = np.array([0.5, 0.1, 0.3, 0.2, 0.9])
        share = np.array([0.5, 1.0, 0.4, 0.0, 0.6])
        picked, counts = mb.select(distance, share, count=4, low=0.3, high=0.7)
        # Boundary in hot-distance order: 2 (0.3), 0 (0.5), 4 (0.9); then the nearest other: 1 (0.1).
        self.assertEqual(picked.tolist(), [2, 0, 4, 1])
        self.assertEqual(counts, {"boundary": 3, "fill": 1, "boundary_available": 3})
        with self.assertRaises(ValueError):
            mb.select(distance, share, count=6, low=0.3, high=0.7)
        # At most one boundary insert: 2 (the nearest boundary one), then the nearest others 1, 3, 0.
        picked, counts = mb.select(distance, share, count=4, low=0.3, high=0.7, max_boundary=1)
        self.assertEqual(picked.tolist(), [2, 1, 3, 0])
        self.assertEqual(counts, {"boundary": 1, "fill": 3, "boundary_available": 3})


class MainTest(unittest.TestCase):
    def test_outputs_and_contract(self) -> None:
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = rng.normal(0.0, 1.0, (400, 4)).astype(np.float32)
            base = source[:100]
            membership = (np.arange(100) % 4).astype("<u4")
            query = rng.normal(0.0, 1.0, (40, 4)).astype(np.float32)
            regions = (np.arange(40) % 5).astype("<i4")
            for name, array in (("source", source), ("base", base), ("membership", membership), ("query", query),
                                ("regions", regions)):
                np.save(root / f"{name}.npy", array)
            (root / "stream.json").write_text(json.dumps({"hot_owner": 3, "hot_regions": [1, 2]}))
            argv = ["--source", str(root / "source.npy"), "--candidate-start", "150", "--candidate-end", "400",
                    "--exclude-below", "150", "--base", str(root / "base.npy"), "--membership",
                    str(root / "membership.npy"), "--query", str(root / "query.npy"), "--query-regions",
                    str(root / "regions.npy"), "--stream-manifest", str(root / "stream.json"), "--count", "20",
                    "--prefilter", "60", "--neighbours", "5", "--output", str(root / "out")]
            self.assertEqual(mb.main(argv), 0)
            out = root / "out"
            inserts = np.load(out / "insert.npy")
            rows = np.load(out / "source_rows.npy")
            self.assertEqual(inserts.shape, (20, 4))
            self.assertTrue(np.array_equal(inserts, source[rows]))
            self.assertTrue(np.all(rows >= 150) and len(set(rows.tolist())) == 20)
            self.assertTrue(np.array_equal(np.load(out / "base_plus_insert.npy"), np.concatenate([base, inserts])))
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(manifest["schema"], mb.SCHEMA)
            selection = manifest["selection"]
            self.assertEqual(selection["boundary"] + selection["fill"], 20)
            self.assertEqual(sum(selection["predicted_home_owner_counts"]), 20)
            # Deterministic: the same inputs select the same rows in the same order.
            self.assertEqual(mb.main(argv[:-1] + [str(root / "again")]), 0)
            self.assertTrue(np.array_equal(np.load(root / "again" / "source_rows.npy"), rows))
            # Refusals: an existing output, candidates below the excluded rows, a base that is not the source's.
            with self.assertRaises(SystemExit):
                mb.main(argv)
            bad = list(argv)
            bad[bad.index("--candidate-start") + 1] = "120"
            bad[-1] = str(root / "bad")
            with self.assertRaises(SystemExit):
                mb.main(bad)
            np.save(root / "base.npy", base + 1)
            with self.assertRaises(SystemExit):
                mb.main(argv[:-1] + [str(root / "other")])


if __name__ == "__main__":
    unittest.main()
