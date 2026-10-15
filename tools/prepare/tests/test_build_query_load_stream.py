import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import build_query_load_stream as builder


def run_cli(argv):
    old = sys.argv
    sys.argv = ["prog"] + argv
    try:
        return builder.main()
    finally:
        sys.argv = old


class LoadStreamTest(unittest.TestCase):
    def build(self, root: Path, repeats=3, seed=7):
        q = np.arange(20, dtype=np.float32).reshape(5, 4)
        gt = np.arange(15, dtype=np.int32).reshape(5, 3)
        np.save(root / "q.npy", q)
        np.save(root / "gt.npy", gt)
        out = root / "stream"
        rc = run_cli(["--query", str(root / "q.npy"),
                      "--groundtruth", str(root / "gt.npy"),
                      "--repeats", str(repeats), "--seed", str(seed),
                      "--output", str(out), "--label", "fixture"])
        self.assertEqual(rc, 0)
        return q, gt, out

    def test_alignment_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q, gt, out = self.build(root)
            sq = np.load(out / "query.npy")
            sg = np.load(out / "groundtruth.npy")
            self.assertEqual(sq.shape, (15, 4))
            self.assertEqual(sg.shape, (15, 3))
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(manifest["schema"], "tide-query-load-stream-v1")
            self.assertEqual(set(manifest), {
                "schema", "note", "label", "seed", "repeats", "native_rows",
                "stream_rows", "source", "per_pass_permutation_sha256", "output"})
            # Every stream row's GT row must be the GT of that query row.
            lookup = {tuple(row): gt[i] for i, row in enumerate(q)}
            for row, gt_row in zip(sq, sg):
                np.testing.assert_array_equal(lookup[tuple(row)], gt_row)
            # Each pass is a permutation of the full native set.
            for ordinal in range(3):
                chunk = sq[ordinal * 5:(ordinal + 1) * 5]
                self.assertEqual(
                    sorted(map(tuple, chunk)), sorted(map(tuple, q)))
                expected = np.random.Generator(np.random.PCG64(7 + ordinal)).permutation(5)
                np.testing.assert_array_equal(chunk, q[expected])
                np.testing.assert_array_equal(sg[ordinal * 5:(ordinal + 1) * 5], gt[expected])

    def test_deterministic_by_seed(self):
        with tempfile.TemporaryDirectory() as a, \
                tempfile.TemporaryDirectory() as b:
            _, _, out_a = self.build(Path(a))
            _, _, out_b = self.build(Path(b))
            qa = (out_a / "query.npy").read_bytes()
            qb = (out_b / "query.npy").read_bytes()
            self.assertEqual(qa, qb)
            ma = json.loads((out_a / "MANIFEST.json").read_text())
            mb = json.loads((out_b / "MANIFEST.json").read_text())
            self.assertEqual(ma["per_pass_permutation_sha256"],
                             mb["per_pass_permutation_sha256"])

    def test_row_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "q.npy", np.zeros((4, 2), dtype=np.float32))
            np.save(root / "gt.npy", np.zeros((5, 3), dtype=np.int32))
            with self.assertRaises(SystemExit):
                run_cli(["--query", str(root / "q.npy"),
                         "--groundtruth", str(root / "gt.npy"),
                         "--repeats", "2", "--seed", "1",
                         "--output", str(root / "s"), "--label", "x"])


if __name__ == "__main__":
    unittest.main()
