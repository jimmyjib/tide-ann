#!/usr/bin/env python3
"""Read, write and check the PQ files of tide-build-pq (format defined in pq.h).

pq_pivots.bin: u32 magic 0x50515442, dim, M, K = 256, then float32 centroids [M][K][dim / M].
pq_codes.bin: u32 magic 0x44435150, n, M, 0, u64 pivots_hash, then n x M code bytes (row i = node i).
pivots_hash is FNV-1a 64 over dim, M, K (u32 little-endian) and the centroid bytes.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

PIVOTS_MAGIC = 0x50515442  # little-endian u32, bytes on disk 42 54 51 50
CODES_MAGIC = 0x44435150   # little-endian u32, bytes on disk 50 51 43 44
K = 256
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


class PQFileError(ValueError):
    pass


def fnv1a64(data: bytes, value: int = FNV_OFFSET) -> int:
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & MASK64
    return value


def pivots_hash(dim: int, m: int, centroids: np.ndarray) -> int:
    header = struct.pack("<III", dim, m, K)
    return fnv1a64(np.ascontiguousarray(centroids, dtype="<f4").tobytes(), fnv1a64(header))


def read_pivots(path: Path) -> tuple[int, int, np.ndarray]:
    """Returns (dim, M, centroids [M, K, dim / M]). Raises PQFileError on a malformed file."""
    raw = Path(path).read_bytes()
    if len(raw) < 16:
        raise PQFileError(f"{path}: truncated pivots header")
    magic, dim, m, k = struct.unpack_from("<IIII", raw, 0)
    if magic != PIVOTS_MAGIC or k != K or m == 0 or dim == 0 or dim % m:
        raise PQFileError(f"{path}: not a PQ pivots file (magic/K/M/dim)")
    sub = dim // m
    expected = 16 + m * K * sub * 4
    if len(raw) != expected:
        raise PQFileError(f"{path}: {len(raw)} bytes, expected {expected}")
    centroids = np.frombuffer(raw, dtype="<f4", offset=16).reshape(m, K, sub).copy()
    return dim, m, centroids


def read_codes_header(path: Path) -> tuple[int, int, int]:
    """(n, M, pivots hash) of a codes file whose size matches its header."""
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24:
        raise PQFileError(f"{path}: truncated codes header")
    magic, n, m, zero, recorded = struct.unpack("<IIIIQ", header)
    if magic != CODES_MAGIC or zero != 0 or m == 0:
        raise PQFileError(f"{path}: not a PQ codes file")
    if path.stat().st_size != 24 + n * m:
        raise PQFileError(f"{path}: size is not the header plus n * M bytes")
    return n, m, recorded


def read_codes(path: Path) -> np.ndarray:
    n, m, _ = read_codes_header(path)
    return np.fromfile(path, dtype=np.uint8, offset=24).reshape(n, m)


def encode(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Codes [rows, M] uint8 of float32 vectors [rows, dim] (nearest centroid per chunk)."""
    m, _, sub = centroids.shape
    rows = np.asarray(vectors, dtype=np.float32).reshape(len(vectors), m, sub)
    codes = np.empty((len(vectors), m), dtype=np.uint8)
    for chunk in range(m):
        diff = rows[:, chunk, None, :] - centroids[chunk][None, :, :]
        codes[:, chunk] = np.argmin(np.einsum("rkd,rkd->rk", diff, diff), axis=1)
    return codes


def write_pivots(path: Path, centroids: np.ndarray) -> None:
    """Write centroids [M, K, sub_dim] as a pivots file."""
    m, k, sub = centroids.shape
    if k != K:
        raise PQFileError("pivots need K = 256 centroids per chunk")
    with Path(path).open("xb") as handle:
        handle.write(struct.pack("<IIII", PIVOTS_MAGIC, m * sub, m, K))
        handle.write(np.ascontiguousarray(centroids, dtype="<f4").tobytes())


def write_codes(path: Path, codes: np.ndarray, centroids: np.ndarray) -> None:
    """Write codes [n, M] as a codes file bound to the hash of centroids."""
    m, _, sub = centroids.shape
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != m:
        raise PQFileError("codes must be n x M bytes")
    with Path(path).open("xb") as handle:
        handle.write(struct.pack("<IIIIQ", CODES_MAGIC, codes.shape[0], m, 0, pivots_hash(m * sub, m, centroids)))
        handle.write(codes.tobytes())


def check_pair(pivots: Path, codes: Path, n: int, dim: int) -> dict:
    """Check that the pivots and codes match each other, n and dim. Returns a summary dict."""
    p_dim, m, centroids = read_pivots(pivots)
    c_n, c_m, recorded = read_codes_header(codes)
    if p_dim != dim:
        raise PQFileError(f"pivots dimension {p_dim} != dataset dimension {dim}")
    if c_m != m:
        raise PQFileError(f"codes M {c_m} != pivots M {m}")
    if c_n != n:
        raise PQFileError(f"codes rows {c_n} != node count {n}")
    actual = pivots_hash(p_dim, m, centroids)
    if recorded != actual:
        raise PQFileError(f"codes were encoded with other pivots (hash {recorded:016x} != {actual:016x})")
    return {"dim": p_dim, "M": m, "sub_dim": p_dim // m, "K": K, "rows": c_n, "pivots_hash": f"{actual:016x}"}
