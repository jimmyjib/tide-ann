"""Unit tests for the synthetic demo dataset and the demo runtime step plan."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from tools.prepare import make_synthetic as ms
from tools.prepare import prepare_demo_runtime as prep

SMALL = ms.Config(n_base=2_000, dim=16, clusters=8, intrinsic_dim=4, n_query=500, hot_clusters=(0, 1, 2),
                  hot_share=0.7, n_insert=300, insert_cluster=3, insert_share=0.5, seed=7)


class GenerateTest(unittest.TestCase):
    def test_shapes_and_dtypes(self) -> None:
        data = ms.generate(SMALL)
        self.assertEqual(data["base"].shape, (2_000, 16))
        self.assertEqual(data["query"].shape, (500, 16))
        self.assertEqual(data["insert"].shape, (300, 16))
        for name in ("base", "query", "insert"):
            self.assertEqual(data[name].dtype, np.float32)
            self.assertTrue(data[name].flags["C_CONTIGUOUS"])
            self.assertTrue(np.isfinite(data[name]).all())
        self.assertEqual(int(data["cluster_sizes"].sum()), 2_000)
        self.assertTrue((data["cluster_sizes"] >= 1).all())
        self.assertEqual(sorted(set(data["base_labels"].tolist())), list(range(8)))

    def test_deterministic_and_seed_sensitive(self) -> None:
        first, second = ms.generate(SMALL), ms.generate(SMALL)
        for name in ("base", "query", "insert", "query_labels", "insert_labels"):
            self.assertTrue(np.array_equal(first[name], second[name]), name)
        other = ms.generate(replace(SMALL, seed=8))
        self.assertFalse(np.array_equal(first["base"], other["base"]))

    def test_query_options_do_not_change_base_or_inserts(self) -> None:
        first = ms.generate(SMALL)
        hotter = ms.generate(replace(SMALL, hot_share=0.9))
        self.assertTrue(np.array_equal(first["base"], hotter["base"]))
        self.assertTrue(np.array_equal(first["insert"], hotter["insert"]))
        self.assertFalse(np.array_equal(first["query_labels"], hotter["query_labels"]))

    def test_hot_share_and_insert_bias(self) -> None:
        data = ms.generate(SMALL)
        hot = np.isin(data["query_labels"], [0, 1, 2])
        self.assertEqual(int(hot.sum()), 350)  # exactly round(0.7 * 500)
        for cluster in (0, 1, 2):
            self.assertGreater(int(np.sum(data["query_labels"] == cluster)), 80)
        self.assertEqual(int(np.sum(data["insert_labels"] == 3)), 150)  # exactly round(0.5 * 300)
        self.assertGreater(len(set(data["insert_labels"].tolist())), 2)

    def test_clusters_are_separated(self) -> None:
        data = ms.generate(SMALL)
        query, base = data["query"].astype(np.float64), data["base"].astype(np.float64)
        dist = (query ** 2).sum(1)[:, None] + (base ** 2).sum(1)[None, :] - 2.0 * query @ base.T
        nearest = data["base_labels"][np.argmin(dist, axis=1)]
        self.assertGreater(float(np.mean(nearest == data["query_labels"])), 0.99)

    def test_unequal_spreads(self) -> None:
        spreads = ms.generate(ms.Config())["cluster_spreads"]
        self.assertGreater(float(spreads.max() / spreads.min()), 2.0)

    def test_default_sizes(self) -> None:
        data = ms.generate(ms.Config())
        self.assertEqual(data["base"].shape, (20_000, 64))
        self.assertEqual(data["query"].shape, (4_096, 64))
        self.assertEqual(data["insert"].shape, (4_000, 64))
        self.assertEqual(len(data["cluster_sizes"]), 32)
        self.assertEqual(int(np.isin(data["query_labels"], [0, 1, 2]).sum()), round(0.7 * 4_096))
        self.assertEqual(int(np.sum(data["insert_labels"] == 3)), 2_000)

    def test_invalid_configs(self) -> None:
        for bad in (replace(SMALL, hot_clusters=(0, 0)), replace(SMALL, insert_cluster=8),
                    replace(SMALL, intrinsic_dim=17), replace(SMALL, hot_share=1.5)):
            with self.assertRaises(ValueError):
                ms.generate(bad)


class WriteTest(unittest.TestCase):
    def test_write_and_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            argv = ["--output", str(out), "--n-base", "600", "--dim", "8", "--clusters", "4",
                    "--intrinsic-dim", "3", "--n-query", "64", "--hot-clusters", "0,1", "--n-insert", "40",
                    "--insert-cluster", "2", "--seed", "3"]
            self.assertEqual(ms.main(argv), 0)
            base, insert = np.load(out / "base.npy"), np.load(out / "insert.npy")
            combined = np.load(out / "base_plus_insert.npy")
            self.assertTrue(np.array_equal(combined, np.concatenate([base, insert])))
            manifest = json.loads((out / "MANIFEST.json").read_text())
            for name, record in manifest["files"].items():
                self.assertEqual(record["sha256"], ms.sha256_file(out / name), name)
            self.assertEqual(manifest["config"]["hot_clusters"], [0, 1])
            self.assertEqual(manifest["shapes"]["query.npy"], [64, 8])
            self.assertEqual(ms.main(argv), 2)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["demo"])


class PrepPlanTest(unittest.TestCase):
    def test_step_plan_uses_existing_tools_in_order(self) -> None:
        root = Path("/data/demo")
        steps = prep.build_steps(root, "/py", n_nodes=20_000, producer_commit="abc")
        self.assertEqual([name for name, _ in steps],
                         ["partition", "ghost", "gt-base", "gt-base-plus-insert", "seal", "row-bytes", "stream"])
        for name, argv in steps:
            tool = Path(next(item for item in argv if item.endswith(".py")))
            self.assertTrue((prep.TOOLS / tool.name).is_file(), name)
        seal = dict(steps)["seal"]
        self.assertEqual(seal[seal.index("--groundtruth") + 1], "/data/demo/gt/base/groundtruth.npy")
        self.assertEqual(seal[seal.index("--parts") + 1], "4")
        after = dict(steps)["gt-base-plus-insert"]
        self.assertEqual(after[after.index("--base") + 1], "/data/demo/source/base_plus_insert.npy")
        self.assertEqual(dict(steps)["ghost"][dict(steps)["ghost"].index("--nodes") + 1], "20000")


if __name__ == "__main__":
    unittest.main()
