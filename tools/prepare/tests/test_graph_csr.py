"""Tests for graph_csr against scipy and row-list references, and for partition_graph."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import partition_graph
from tools.prepare.graph_csr import (
    CsrFormatError,
    CsrScaleError,
    csr_static_metrics,
    metis_int32_arrays,
    read_global_graph_csr,
    row_blocks_by_edges,
    symmetrize_csr,
    transpose_csr,
    write_adj_csr,
)
from tools.prepare.tests.synthetic_runtime import (
    balanced_membership, build_undirected_csr, random_rows, static_metrics, write_adj, write_global_graph)


def rows_to_csr(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(rows) + 1, dtype=np.uint64)
    indptr[1:] = np.cumsum([len(r) for r in rows]).astype(np.uint64)
    indices = np.concatenate(rows).astype(np.uint32) if rows else np.zeros(0, dtype=np.uint32)
    return indptr, indices


class SymmetrizeTests(unittest.TestCase):
    def test_symmetrize_matches_scipy_reference_across_block_sizes(self) -> None:
        rng = np.random.default_rng(3)
        n = 120
        rows = random_rows(n, rng, 2, 9)
        indptr, indices = rows_to_csr(rows)
        U = build_undirected_csr(rows, n)
        for block_edges in (5, 64, 1 << 24):
            uindptr, uindices = symmetrize_csr(indptr, indices, block_edges=block_edges)
            self.assertEqual(uindptr.dtype, np.uint64)
            self.assertEqual(uindices.dtype, np.uint32)
            np.testing.assert_array_equal(uindptr.astype(np.int64), U.indptr.astype(np.int64))
            np.testing.assert_array_equal(uindices.astype(np.int64), U.indices.astype(np.int64))

    def test_symmetrize_drops_self_loops_and_duplicates(self) -> None:
        rows = [np.array([1, 1, 0, 2], dtype=np.uint32), np.array([2], dtype=np.uint32),
                np.array([2, 0], dtype=np.uint32), np.array([], dtype=np.uint32)]
        indptr, indices = rows_to_csr(rows)
        uindptr, uindices = symmetrize_csr(indptr, indices, block_edges=3)
        expected = [[1, 2], [0, 2], [0, 1], []]
        for node, row in enumerate(expected):
            self.assertEqual(uindices[uindptr[node]:uindptr[node + 1]].tolist(), row)

    def test_transpose_matches_scipy(self) -> None:
        import scipy.sparse as sp
        rng = np.random.default_rng(5)
        n = 80
        rows = random_rows(n, rng, 1, 6)
        indptr, indices = rows_to_csr(rows)
        A = sp.csr_matrix((np.ones(len(indices)), indices.astype(np.int64), indptr.astype(np.int64)), shape=(n, n))
        T = A.T.tocsr()
        T.sort_indices()
        for block_edges in (4, 1 << 24):
            rindptr, rindices = transpose_csr(indptr, indices, block_edges=block_edges)
            np.testing.assert_array_equal(rindptr.astype(np.int64), T.indptr.astype(np.int64))
            np.testing.assert_array_equal(rindices.astype(np.int64), T.indices.astype(np.int64))

    def test_row_blocks_cover_rows_in_order(self) -> None:
        indptr = np.array([0, 3, 3, 10, 11, 20], dtype=np.uint64)
        blocks = list(row_blocks_by_edges(indptr, 7))
        self.assertEqual(blocks[0][0], 0)
        self.assertEqual(blocks[-1][1], 5)
        for (a, b), (c, _) in zip(blocks, blocks[1:]):
            self.assertEqual(b, c)
            self.assertLess(a, b)
        # a single row larger than the block still forms its own block
        self.assertIn((4, 5), blocks)


class ReaderTests(unittest.TestCase):
    def test_global_graph_matches_rows_and_validates(self) -> None:
        rng = np.random.default_rng(2)
        n = 70
        rows = random_rows(n, rng, 0, 5)
        max_degree = max(len(r) for r in rows)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "graph"
            write_global_graph(path, rows, 3, max_degree)
            indptr, indices, start, degree = read_global_graph_csr(path, n, block_rows=16)
            self.assertEqual((start, degree), (3, max_degree))
            self.assertEqual(indptr.dtype, np.uint64)
            self.assertEqual(indices.dtype, np.uint32)
            ref_indptr, ref_indices = rows_to_csr(rows)
            np.testing.assert_array_equal(indptr, ref_indptr)
            np.testing.assert_array_equal(indices, ref_indices)
            # trailing data
            path.write_bytes(path.read_bytes() + b"\x00\x00\x00\x00")
            write_global_graph(Path(raw) / "g2", rows, 3, max_degree)
            broken = Path(raw) / "g2"
            payload = bytearray(broken.read_bytes())
            payload += b"\x00\x00\x00\x00"
            payload[0:8] = (len(payload)).to_bytes(8, "little")
            broken.write_bytes(bytes(payload))
            with self.assertRaisesRegex(CsrFormatError, "trailing"):
                read_global_graph_csr(broken, n)
            # neighbor outside the graph
            bad_rows = [r.copy() for r in rows]
            bad_rows[5] = np.array([n + 3], dtype=np.uint32)
            write_global_graph(Path(raw) / "g3", bad_rows, 3, max_degree)
            with self.assertRaisesRegex(CsrFormatError, "neighbor"):
                read_global_graph_csr(Path(raw) / "g3", n)
            # degree above the header maximum
            write_global_graph(Path(raw) / "g4", rows, 3, max_degree - 1)
            with self.assertRaisesRegex(CsrFormatError, "degree exceeds"):
                read_global_graph_csr(Path(raw) / "g4", n)


class ReferenceTests(unittest.TestCase):
    def test_metrics_and_writer_match_row_helpers(self) -> None:
        rng = np.random.default_rng(9)
        n = 64
        rows = random_rows(n, rng, 0, 6)
        membership = balanced_membership(n, rng)
        indptr, indices = rows_to_csr(rows)
        self.assertEqual(csr_static_metrics(indptr, indices, membership, 4, block_edges=5),
                         static_metrics(rows, membership, 4))
        ids = np.flatnonzero(membership == 2)
        with tempfile.TemporaryDirectory() as raw:
            ref = Path(raw) / "ref"
            out = Path(raw) / "csr"
            size_ref = write_adj(ref, [rows[int(g)] for g in ids], 2, 6)
            size_csr = write_adj_csr(out, indptr, indices, ids, 2, 6, block_rows=3)
            self.assertEqual(size_ref, size_csr)
            self.assertEqual(ref.read_bytes(), out.read_bytes())


class MetisGuardTests(unittest.TestCase):
    def test_metis_int32_limits(self) -> None:
        with self.assertRaisesRegex(CsrScaleError, "2\\^31.*64-bit|64-bit"):
            metis_int32_arrays(np.array([0, 2 ** 31], dtype=np.uint64), np.zeros(0, dtype=np.uint32))
        uindptr = np.array([0, 1, 2], dtype=np.uint64)
        uindices = np.array([1, 0], dtype=np.uint32)
        xadj, adjncy = metis_int32_arrays(uindptr, uindices)
        self.assertEqual((xadj.dtype, adjncy.dtype), (np.int32, np.int32))
        self.assertEqual((xadj.tolist(), adjncy.tolist()), ([0, 1, 2], [1, 0]))
        with self.assertRaises(CsrFormatError):
            metis_int32_arrays(uindptr, np.array([1, 5], dtype=np.uint32))


def run_partition_graph(argv: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return partition_graph.main(argv)


class PartitionToolTests(unittest.TestCase):
    def test_partition_files_match_the_row_helpers(self) -> None:
        """METIS gets the symmetrized graph, and the adj files equal the row-list write_adj of the same rows."""
        import pymetis
        rng = np.random.default_rng(21)
        n = 200
        rows = random_rows(n, rng, 2, 8)
        base = rng.normal(size=(n, 4)).astype(np.float32)
        max_degree = max(len(r) for r in rows)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_global_graph(root / "graph", rows, 17, max_degree)
            np.save(root / "base.npy", base)
            argv = ["--graph", str(root / "graph"), "--base", str(root / "base.npy"), "--output", str(root / "out")]
            self.assertEqual(run_partition_graph(argv), 0)
            undirected = [set(map(int, row)) for row in rows]
            for source, row in enumerate(rows):
                for target in row:
                    undirected[int(target)].add(source)
            cuts, expected = pymetis.part_graph(4, adjacency=[sorted(row) for row in undirected])
            membership = np.load(root / "out" / "membership.npy")
            self.assertEqual(membership.dtype, np.int32)
            np.testing.assert_array_equal(membership, np.asarray(expected, dtype=np.int32))
            for part in range(4):
                directory = root / "out" / f"partition_{part}"
                ids = np.load(directory / "local_to_global.npy")
                np.testing.assert_array_equal(ids, np.flatnonzero(membership == part).astype(np.uint32))
                start = int(np.searchsorted(ids, 17)) if membership[17] == part else 0
                write_adj(root / f"adj-{part}", [rows[int(g)] for g in ids], start, max_degree)
                self.assertEqual((directory / "adj").read_bytes(), (root / f"adj-{part}").read_bytes(), part)
            manifest = json.loads((root / "out" / "PARTITION_BUILD.json").read_text())
            self.assertEqual(sorted(manifest), ["N", "metis_edge_cuts", "parts", "static_metrics"])
            self.assertEqual((manifest["N"], manifest["parts"], manifest["metis_edge_cuts"]), (n, 4, cuts))
            self.assertEqual(manifest["static_metrics"], static_metrics(rows, membership, 4))
            self.assertEqual(sorted(path.name for path in (root / "out").iterdir()),
                             ["PARTITION_BUILD.json", "membership.npy", "partition_0", "partition_1", "partition_2",
                              "partition_3"])

    def test_partition_graph_eight_parts(self) -> None:
        """K = 8 works, and K outside 2..64 is rejected."""
        rng = np.random.default_rng(22)
        n = 400
        rows = random_rows(n, rng, 2, 8)
        base = rng.normal(size=(n, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_global_graph(root / "graph", rows, 17, max(len(r) for r in rows))
            np.save(root / "base.npy", base)
            common = ["--graph", str(root / "graph"), "--base", str(root / "base.npy")]
            self.assertEqual(run_partition_graph(common + ["--output", str(root / "k8"), "--parts", "8"]), 0)
            membership = np.load(root / "k8" / "membership.npy")
            self.assertEqual(sorted(set(membership.tolist())), list(range(8)))
            manifest = json.loads((root / "k8" / "PARTITION_BUILD.json").read_text())
            self.assertEqual(manifest["parts"], 8)
            owned = [np.load(root / "k8" / f"partition_{p}" / "local_to_global.npy") for p in range(8)]
            self.assertEqual(sorted(np.concatenate(owned).tolist()), list(range(n)))
            for parts in ("1", "65"):
                with self.assertRaises(partition_graph.PrepareError):
                    run_partition_graph(common + ["--output", str(root / f"bad{parts}"), "--parts", parts])


if __name__ == "__main__":
    unittest.main()
