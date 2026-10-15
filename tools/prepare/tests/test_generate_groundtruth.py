import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare.generate_groundtruth import (
    select_topk,
    squared_l2_scores,
    stable_inner_products,
)


class GenerateGroundTruthTest(unittest.TestCase):
    def test_range_manifest_binds_identity_and_tie_break(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.npy"
            query = root / "query.npy"
            output = root / "range.npy"
            np.save(base, np.array([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32))
            np.save(query, np.array([[0., 0.], [2., 2.]], dtype=np.float32))
            completed = subprocess.run([
                sys.executable,
                str(Path(__file__).parents[1] / "generate_groundtruth.py"),
                "--base", str(base), "--query", str(query), "--output", str(output),
                "--metric", "l2", "--k", "1", "--l2-float64",
                "--l2-kernel", "scipy-cdist", "--query-start", "1", "--query-end", "2",
                "--verified-base-sha256", "b" * 64,
                "--verified-query-sha256", "q" * 64,
            ], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((root / "range.npy.json").read_text())
            self.assertEqual(manifest["query_range"], [1, 2])
            self.assertEqual(manifest["full_query_count"], 2)
            self.assertEqual(manifest["base_sha256"], "b" * 64)
            self.assertEqual(manifest["query_sha256"], "q" * 64)
            self.assertEqual(
                manifest["tie_break"],
                "distance_or_negative_similarity_then_vector_id_ascending",
            )
            np.testing.assert_array_equal(np.load(output), [[2]])

    def test_scipy_cdist_matches_direct_squared_difference(self):
        queries = np.array([[1., 2.], [-3., 4.]], dtype=np.float64)
        vectors = np.array([[5., 6.], [1., -2.], [0., 0.]], dtype=np.float64)
        scores, kernel = squared_l2_scores(queries, vectors, kernel="scipy-cdist")
        np.testing.assert_array_equal(
            scores,
            np.sum((queries[:, None, :] - vectors[None, :, :]) ** 2, axis=2),
        )
        self.assertEqual(kernel, "scipy-cdist-sqeuclidean-float64")

    def test_nonfinite_matmul_uses_exact_einsum_fallback(self):
        queries = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
        vectors = np.array([[5., 6.], [7., 8.]], dtype=np.float64)

        def broken(left, right):
            return np.full((left.shape[0], right.shape[1]), np.inf)

        products, kernel = stable_inner_products(queries, vectors, matmul=broken)
        np.testing.assert_array_equal(products, queries @ vectors.T)
        self.assertEqual(kernel, "einsum-fallback")

    def test_cutoff_tie_uses_smallest_vector_ids(self):
        scores = np.array([[0., 1., 1., 1., 2.]], dtype=np.float32)
        ids = np.array([[9, 8, 3, 5, 1]], dtype=np.int32)
        _, chosen = select_topk(scores, ids, 3, largest=False)
        np.testing.assert_array_equal(chosen, [[9, 3, 5]])


if __name__ == "__main__":
    unittest.main()
