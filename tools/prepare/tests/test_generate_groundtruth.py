"""Tests for generate_groundtruth."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import generate_groundtruth
from tools.prepare.generate_groundtruth import select_topk


class GenerateGroundTruthTest(unittest.TestCase):
    def test_command_line_matches_brute_force(self):
        rng = np.random.default_rng(4)
        base = rng.normal(size=(generate_groundtruth.BASE_CHUNK + 100, 6)).astype(np.float32)
        queries = rng.normal(size=(generate_groundtruth.QUERY_CHUNK + 6, 6)).astype(np.float32)
        l2 = ((queries[:, None, :].astype(np.float64) - base[None, :, :].astype(np.float64)) ** 2).sum(axis=2)
        unit_base = base / np.linalg.norm(base, axis=1, keepdims=True)
        unit_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        cosine = unit_queries.astype(np.float64) @ unit_base.astype(np.float64).T
        # (score matrix, ids of the exact top 10, smaller is better)
        expected = {"l2": (l2, np.argsort(l2, axis=1, kind="stable")[:, :10]),
                    "cosine": (-cosine, np.argsort(-cosine, axis=1, kind="stable")[:, :10])}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "base.npy", base)
            np.save(root / "query.npy", queries)
            for metric, (scores, ids) in expected.items():
                output = root / "gt" / metric / "groundtruth.npy"  # the parent does not exist yet
                argv = ["--base", str(root / "base.npy"), "--query", str(root / "query.npy"),
                        "--output", str(output), "--metric", metric, "--k", "10"]
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(generate_groundtruth.main(argv), 0)
                self.assertEqual(out.getvalue(), f"wrote {output} ({len(queries)} x 10)\n")
                result = np.load(output)
                self.assertEqual(result.dtype, np.int32)
                if metric == "l2":  # float64 on both sides
                    np.testing.assert_array_equal(result, ids)
                else:  # float32 in the tool: compare the scores, near-ties may swap
                    np.testing.assert_allclose(np.take_along_axis(scores, result.astype(np.int64), axis=1),
                                               np.take_along_axis(scores, ids, axis=1), rtol=0, atol=1e-6)
                with self.assertRaises(SystemExit):  # an existing output is never overwritten
                    generate_groundtruth.main(argv)

    def test_cutoff_tie_uses_smallest_vector_ids(self):
        scores = np.array([[0., 1., 1., 1., 2.]], dtype=np.float32)
        ids = np.array([[9, 8, 3, 5, 1]], dtype=np.int32)
        _, chosen = select_topk(scores, ids, 3, largest=False)
        np.testing.assert_array_equal(chosen, [[9, 3, 5]])


if __name__ == "__main__":
    unittest.main()
