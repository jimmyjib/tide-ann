"""Tests for split_queries_by_cluster and select_insert_batch."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import select_insert_batch, split_queries_by_cluster


def run(module, argv: list[str]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = module.main(argv)
    if code != 0:
        raise AssertionError(f"exit {code}")
    return json.loads(out.getvalue().strip().splitlines()[-1])


class SplitQueriesTest(unittest.TestCase):
    def test_one_file_per_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source").mkdir()
            query = np.arange(20, dtype=np.float32).reshape(10, 2)
            labels = np.array([0, 2, 2, 1, 0, 2, 5, 0, 1, 2], dtype=np.int32)
            np.save(root / "source" / "query.npy", query)
            np.save(root / "source" / "query_labels.npy", labels)
            counts = run(split_queries_by_cluster, ["--source", str(root / "source"), "--hot-clusters", "2,0",
                                                    "--output", str(root / "control")])
            self.assertEqual(counts, {"0": 3, "2": 4})
            for cluster in (0, 2):
                rows = np.load(root / "control" / f"query-hot-{cluster}.npy")
                self.assertEqual(rows.dtype, np.float32)
                np.testing.assert_array_equal(rows, query[labels == cluster])
            self.assertFalse((root / "control" / "query-hot-1.npy").exists())
            with self.assertRaisesRegex(SystemExit, "query-hot-0.npy"):  # an existing file is kept
                run(split_queries_by_cluster, ["--source", str(root / "source"), "--hot-clusters", "1,0",
                                               "--output", str(root / "control")])
            self.assertFalse((root / "control" / "query-hot-1.npy").exists())
            with self.assertRaises(SystemExit):  # no query comes from cluster 3
                run(split_queries_by_cluster, ["--source", str(root / "source"), "--hot-clusters", "3",
                                               "--output", str(root / "other")])


class SelectInsertBatchTest(unittest.TestCase):
    def test_rows_nearest_to_the_server_come_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # server 0 owns the vectors near (0, 0), server 1 the vectors near (10, 10)
            points = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]], dtype=np.float32)
            owners = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
            # pool rows 1, 3 and 4 are nearest to server 1. Of rows 0, 2 and 5 (nearest to server 0), row 2 is
            # the closest to server 1.
            pool = np.array([[0.5, 0.5], [10.5, 10.5], [4, 4], [9.5, 10], [10, 9.5], [0, 0.5]], dtype=np.float32)
            np.save(root / "points.npy", points)
            np.save(root / "owners.npy", owners)
            np.save(root / "pool.npy", pool)
            argv = ["--points", str(root / "points.npy"), "--owners", str(root / "owners.npy"), "--server", "1",
                    "--servers", "2", "--pool", str(root / "pool.npy"), "--first-row", "0", "--pool-rows", "6",
                    "--output", str(root / "control" / "insert.npy"),
                    "--base-plus-inserts", str(root / "gt-input" / "base-plus-inserts.npy")]
            summary = run(select_insert_batch, argv + ["--rows", "3"])
            batch = np.load(root / "control" / "insert.npy")
            np.testing.assert_array_equal(batch, pool[[1, 3, 4]])  # pool order
            self.assertEqual(summary, {"batch_rows": 3, "base_plus_inserts_rows": 9, "pool_rows_on_server": 3,
                                       "nearest_on_server": 3, "nearest_owner_counts": [0, 3]})
            combined = np.load(root / "gt-input" / "base-plus-inserts.npy")
            np.testing.assert_array_equal(combined[:len(points)], points)
            for i in range(len(batch)):  # insert row i is id N + i
                np.testing.assert_array_equal(combined[len(points) + i], batch[i])
            # one more row: row 2 comes next
            for name in ("control/insert.npy", "gt-input/base-plus-inserts.npy"):
                (root / name).unlink()
            run(select_insert_batch, argv + ["--rows", "4"])
            np.testing.assert_array_equal(np.load(root / "control" / "insert.npy"), pool[[1, 2, 3, 4]])
            with self.assertRaisesRegex(SystemExit, "already exists"):  # a second run keeps the first batch
                select_insert_batch.main(argv + ["--rows", "3"])
            np.testing.assert_array_equal(np.load(root / "control" / "insert.npy"), pool[[1, 2, 3, 4]])
            (root / "control" / "insert.npy").unlink()
            with self.assertRaisesRegex(SystemExit, "base-plus-inserts.npy"):  # either output is enough
                select_insert_batch.main(argv + ["--rows", "3"])
            (root / "gt-input" / "base-plus-inserts.npy").unlink()
            with self.assertRaisesRegex(SystemExit, "bad --first-row"):  # more rows than the pool
                select_insert_batch.main(argv + ["--rows", "7"])


if __name__ == "__main__":
    unittest.main()
