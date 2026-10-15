"""Hybrid PQ own-only storage data tools: the PQ file formats and hash (pq_files), the per-partition
vector writer and its manifest (write_partition_vectors) and unit-length normalisation
(normalize_vectors)."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare import normalize_vectors, pq_files
from tools.prepare import write_partition_vectors as wpv

DIM, M, N, PARTS = 8, 4, 64, 4


def centroids() -> np.ndarray:
    """Centroid k of every 2-dim chunk is (k % 16, k // 16)."""
    grid = np.stack([np.arange(256) % 16, np.arange(256) // 16], axis=1).astype(np.float32)
    return np.repeat(grid[None], M, axis=0)


def make_runtime(root: Path) -> tuple[Path, Path]:
    """A sealed-looking 4-part runtime (manifest, base, membership, partitions) and a PQ directory
    holding build_pq's two files for it."""
    rng = np.random.default_rng(5)
    runtime, pq_dir = root / "runtime", root / "pq"
    runtime.mkdir()
    pq_dir.mkdir()
    base = rng.integers(0, 16, size=(N, DIM)).astype(np.float32)
    np.save(runtime / "base.npy", base)
    membership = (np.arange(N) * 7) % PARTS
    np.save(runtime / "membership.npy", membership.astype(np.uint32))
    for rank in range(PARTS):
        (runtime / f"partition_{rank}").mkdir()
        ids = np.flatnonzero(membership == rank)[::-1].astype(np.uint32)  # not ascending
        np.save(runtime / f"partition_{rank}" / "local_to_global.npy", ids)
        (runtime / f"partition_{rank}" / "adj").write_bytes(b"adj-%d" % rank)
    (runtime / "RUNTIME_MANIFEST.json").write_text(json.dumps({"N": N, "dimension": DIM, "parts": PARTS,
                                                               "metric": "l2"}))
    table = centroids()
    pq_files.write_pivots(pq_dir / "pq_pivots.bin", table)
    pq_files.write_codes(pq_dir / "pq_codes.bin", pq_files.encode(base, table), table)
    return runtime, pq_dir


class PQFilesTest(unittest.TestCase):
    def test_formats_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = centroids()
            pivots, codes = Path(tmp) / "p.bin", Path(tmp) / "c.bin"
            pq_files.write_pivots(pivots, table)
            raw = pivots.read_bytes()
            self.assertEqual(struct.unpack_from("<IIII", raw), (pq_files.PIVOTS_MAGIC, DIM, M, 256))
            dim, m, read = pq_files.read_pivots(pivots)
            self.assertEqual((dim, m), (DIM, M))
            self.assertTrue(np.array_equal(read, table))
            # The hash is FNV-1a 64 over dim, M, K (u32 LE) then the centroid bytes (pq.h pivots_hash).
            value = pq_files.FNV_OFFSET
            for byte in struct.pack("<III", DIM, M, 256) + table.astype("<f4").tobytes():
                value = ((value ^ byte) * pq_files.FNV_PRIME) % (1 << 64)
            self.assertEqual(pq_files.pivots_hash(DIM, M, table), value)
            vectors = np.array([[0, 0, 3, 1, 15, 15, 2, 0], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
            encoded = pq_files.encode(vectors, table)
            self.assertEqual(encoded.tolist(), [[0, 19, 255, 2], [33, 67, 101, 135]])
            pq_files.write_codes(codes, encoded, table)
            self.assertEqual(pq_files.read_codes_header(codes), (2, M, value))
            self.assertTrue(np.array_equal(pq_files.read_codes(codes), encoded))
            self.assertEqual(pq_files.check_pair(pivots, codes, 2, DIM)["pivots_hash"], f"{value:016x}")
            with self.assertRaises(pq_files.PQFileError):
                pq_files.check_pair(pivots, codes, 3, DIM)
            other = table.copy()
            other[0, 0, 0] = 99
            codes.unlink()
            pq_files.write_codes(codes, encoded, other)
            with self.assertRaisesRegex(pq_files.PQFileError, "other pivots"):
                pq_files.check_pair(pivots, codes, 2, DIM)
            pivots.write_bytes(raw[:-4])
            with self.assertRaises(pq_files.PQFileError):
                pq_files.read_pivots(pivots)


class WritePartitionVectorsTest(unittest.TestCase):
    def test_vectors_in_local_to_global_order_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime, pq_dir = make_runtime(Path(tmp))
            record = wpv.write_partition_vectors(runtime, pq_dir)
            base = np.load(runtime / "base.npy")
            for rank in range(PARTS):
                ids = np.load(runtime / f"partition_{rank}" / "local_to_global.npy")
                vectors = np.load(pq_dir / f"partition_{rank}" / "vectors.npy")
                self.assertEqual(vectors.dtype, np.dtype("<f4"))
                self.assertTrue(np.array_equal(vectors, base[ids]))
            manifest = json.loads((pq_dir / "PQ_MANIFEST.json").read_text())
            self.assertEqual(manifest, record)
            self.assertEqual(manifest["pq"]["M"], M)
            self.assertEqual(sorted(manifest["inventory"]),
                             sorted(["pq_pivots.bin", "pq_codes.bin"] +
                                    [f"partition_{r}/vectors.npy" for r in range(PARTS)]))
            path = pq_dir / "partition_2" / "vectors.npy"
            self.assertEqual(manifest["inventory"]["partition_2/vectors.npy"]["sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(pq_files.PQFileError, "already sealed"):
                wpv.write_partition_vectors(runtime, pq_dir)

    def test_codes_of_other_vectors_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime, pq_dir = make_runtime(Path(tmp))
            table = centroids()
            codes = pq_files.read_codes(pq_dir / "pq_codes.bin")
            (pq_dir / "pq_codes.bin").unlink()
            pq_files.write_codes(pq_dir / "pq_codes.bin", codes[::-1], table)  # the rows of other nodes
            with self.assertRaisesRegex(pq_files.PQFileError, "encode to their stored code"):
                wpv.write_partition_vectors(runtime, pq_dir)

    def test_partition_must_hold_its_own_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime, pq_dir = make_runtime(Path(tmp))
            ids = np.load(runtime / "partition_1" / "local_to_global.npy")
            np.save(runtime / "partition_1" / "local_to_global.npy", np.append(ids, np.uint32(0)))
            with self.assertRaises(pq_files.PQFileError):
                wpv.write_partition_vectors(runtime, pq_dir)


class NormalizeTest(unittest.TestCase):
    def test_unit_rows_and_zero_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.array([[3, 4], [1, 0], [-2, 0]], dtype=np.float32)
            np.save(root / "a.npy", data)
            np.save(root / "b.npy", np.array([[0, 2]], dtype=np.float64))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                normalize_vectors.main(["--input", str(root / "a.npy"), "--output", str(root / "a1.npy"),
                                        "--input", str(root / "b.npy"), "--output", str(root / "b1.npy")])
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["status"], "COMPLETE")
            unit = np.load(root / "a1.npy")
            self.assertEqual(unit.dtype, np.dtype("<f4"))
            self.assertTrue(np.allclose(unit, [[0.6, 0.8], [1, 0], [-1, 0]]))
            self.assertTrue(np.allclose(np.load(root / "b1.npy"), [[0, 1]]))
            # On unit vectors squared L2 = 2 - 2 cos: the L2 order is the cosine order.
            cos = unit @ unit[0]
            l2 = ((unit - unit[0]) ** 2).sum(axis=1)
            self.assertTrue(np.allclose(l2, 2 - 2 * cos, atol=1e-6))
            np.save(root / "z.npy", np.array([[1, 1], [0, 0]], dtype=np.float32))
            with self.assertRaises(SystemExit):
                normalize_vectors.main(["--input", str(root / "z.npy"), "--output", str(root / "z1.npy")])
            self.assertFalse((root / "z1.npy").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                normalize_vectors.main(["--input", str(root / "z.npy"), "--output", str(root / "z2.npy"),
                                        "--zero-rows", "keep"])
            self.assertTrue(np.array_equal(np.load(root / "z2.npy")[1], [0, 0]))
            with self.assertRaises(SystemExit):  # an existing output is never overwritten
                normalize_vectors.main(["--input", str(root / "a.npy"), "--output", str(root / "a1.npy")])


if __name__ == "__main__":
    unittest.main()
