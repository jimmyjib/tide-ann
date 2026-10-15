"""Chunked CSR construction for large directed Vamana graphs.

Every builder here works on flat numpy arrays (``indptr`` uint64, ``indices``
uint32) and walks the graph in row or edge blocks; no per-node Python lists
are created, so the memory is a small constant times the edge count.

* ``read_global_graph_csr`` streams one global DiskANN graph file
  (``<QIIQ`` header, then per row ``<I degree`` + ``degree`` uint32 ids).
* ``transpose_csr`` builds the in-edge CSR with a blocked counting sort.
* ``iter_undirected_blocks`` / ``symmetrize_csr`` build the undirected
  structure of A + A^T: per row block the out- and in-edges are packed into
  uint64 (row, column) keys, sorted and de-duplicated (self loops dropped),
  so every row comes out sorted exactly like the set-based row-list path
  of ``partition_graph`` (without ``--csr``).
* ``undirected_edge_cuts`` scores memberships from that block stream
  without materialising the undirected CSR.
* ``metis_int32_arrays`` prepares pymetis inputs and refuses graphs that do
  not fit METIS' 32-bit ``idx_t`` (nnz or weight sums >= 2**31), naming the
  64-bit METIS fallback.
* ``csr_ordered_graph_hash``, ``csr_static_metrics`` and ``write_adj_csr``
  reproduce ``runtime_lib``'s row-list helpers byte for byte from the CSR.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from tools.prepare.runtime_lib import RuntimeErrorChecked

INT32_LIMIT = 2 ** 31
UINT32_LIMIT = 2 ** 32
DEFAULT_BLOCK_EDGES = 1 << 24      # 16M edge slots per block (~0.5 GB of transient int64 scratch)
DEFAULT_BLOCK_ROWS = 1 << 20
METIS_64BIT_ADVICE = (
    "pymetis links a 32-bit METIS (idx_t = int32), so this graph must go through the 64-bit "
    "METIS fallback: a METIS build with IDXTYPEWIDTH=64 (gpmetis / a metis Python binding "
    "compiled with IDXTYPEWIDTH=64), or coarsen the graph below 2^31 undirected entries first"
)


class CsrScaleError(RuntimeError):
    """The graph does not fit the 32-bit contract of the requested consumer."""


class CsrFormatError(RuntimeErrorChecked):
    """A graph file violates the DiskANN adjacency layout."""


# --------------------------------------------------------------------------
# Block iteration helpers
# --------------------------------------------------------------------------

def _check_node_count(n_nodes: int) -> None:
    if n_nodes <= 0:
        raise CsrScaleError("graph has no nodes")
    if n_nodes >= UINT32_LIMIT:
        raise CsrScaleError(f"{n_nodes} nodes do not fit uint32 node ids")


def row_blocks_by_edges(indptr: np.ndarray, block_edges: int = DEFAULT_BLOCK_EDGES) -> Iterator[tuple[int, int]]:
    """Yield (a, b) row ranges holding at most ``block_edges`` entries each
    (always at least one row, so a single huge row still forms a block)."""
    n = len(indptr) - 1
    if block_edges <= 0:
        raise ValueError("block_edges must be positive")
    a = 0
    while a < n:
        target = int(indptr[a]) + block_edges
        b = int(np.searchsorted(indptr, target, side="right")) - 1
        b = max(a + 1, min(b, n))
        yield a, b
        a = b


def _edge_blocks(nnz: int, block_edges: int) -> Iterator[tuple[int, int]]:
    for a in range(0, nnz, block_edges):
        yield a, min(nnz, a + block_edges)


def _degrees(indptr: np.ndarray) -> np.ndarray:
    return np.diff(indptr).astype(np.int64)


def row_bytes_from_csr(indptr: np.ndarray) -> np.ndarray:
    """Directed row bytes (4-byte degree word + 4 bytes per neighbour), int64."""
    return 4 + 4 * _degrees(indptr)


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def read_global_graph_csr(path: Path, n_nodes: int, block_rows: int = DEFAULT_BLOCK_ROWS
                          ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Stream a global DiskANN graph file into (indptr uint64, indices uint32, start_node, max_degree).

    Validation matches ``partition_graph.read_global_graph``: header size,
    start node, per-row degree bound, neighbour range and trailing data."""
    _check_node_count(n_nodes)
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24:
        raise CsrFormatError("short graph header")
    file_size, max_degree, start_node, _ = struct.unpack("<QIIQ", header)
    if file_size != path.stat().st_size or start_node >= n_nodes:
        raise CsrFormatError("invalid graph header")
    if (file_size - 24) % 4:
        raise CsrFormatError("graph payload is not a whole number of uint32 words")
    words = np.memmap(path, dtype="<u4", mode="r", offset=24) if file_size > 24 else np.zeros(0, dtype="<u4")
    total_words = int(len(words))
    degrees = np.zeros(n_nodes, dtype=np.int64)
    cursor = 0
    for node in range(n_nodes):
        if cursor >= total_words:
            raise CsrFormatError("truncated graph degree")
        degree = int(words[cursor])
        if degree > max_degree:
            raise CsrFormatError("degree exceeds graph max")
        cursor += 1 + degree
        if cursor > total_words:
            raise CsrFormatError("truncated graph neighbours")
        degrees[node] = degree
    if cursor != total_words:
        raise CsrFormatError("unexpected graph trailing data")
    indptr = np.zeros(n_nodes + 1, dtype=np.uint64)
    np.cumsum(degrees, dtype=np.uint64, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.uint32)
    for a in range(0, n_nodes, block_rows):
        b = min(n_nodes, a + block_rows)
        w0 = int(indptr[a]) + a
        w1 = int(indptr[b]) + b
        chunk = np.asarray(words[w0:w1], dtype=np.uint32)
        degree_positions = (indptr[a:b].astype(np.int64) + np.arange(a, b, dtype=np.int64)) - w0
        keep = np.ones(w1 - w0, dtype=bool)
        keep[degree_positions] = False
        rows = chunk[keep]
        if rows.size and int(rows.max()) >= n_nodes:
            raise CsrFormatError("invalid graph neighbour")
        indices[int(indptr[a]):int(indptr[b])] = rows
    return indptr, indices, int(start_node), int(max_degree)


# --------------------------------------------------------------------------
# Transpose and symmetrisation
# --------------------------------------------------------------------------

def transpose_csr(indptr: np.ndarray, indices: np.ndarray,
                  block_edges: int = DEFAULT_BLOCK_EDGES) -> tuple[np.ndarray, np.ndarray]:
    """In-edge CSR (uint64 indptr, uint32 indices) by a blocked counting sort.

    Each in-neighbour list is sorted ascending by source id (blocks are
    consumed in increasing source order and the block sort is stable)."""
    n = len(indptr) - 1
    _check_node_count(n)
    nnz = int(indptr[-1])
    if len(indices) != nnz:
        raise CsrFormatError("indptr / indices length mismatch")
    in_degree = np.zeros(n, dtype=np.int64)
    for a, b in _edge_blocks(nnz, block_edges):
        in_degree += np.bincount(indices[a:b], minlength=n)
    rindptr = np.zeros(n + 1, dtype=np.uint64)
    np.cumsum(in_degree, dtype=np.uint64, out=rindptr[1:])
    rindices = np.empty(nnz, dtype=np.uint32)
    cursor = rindptr[:-1].astype(np.int64)
    degrees = _degrees(indptr)
    for a, b in row_blocks_by_edges(indptr, block_edges):
        lo, hi = int(indptr[a]), int(indptr[b])
        if hi == lo:
            continue
        dst = indices[lo:hi].astype(np.int64)
        src = np.repeat(np.arange(a, b, dtype=np.uint32), degrees[a:b])
        order = np.argsort(dst, kind="stable")
        dst = dst[order]
        src = src[order]
        unique_dst, first, counts = np.unique(dst, return_index=True, return_counts=True)
        rank = np.arange(len(dst), dtype=np.int64) - np.repeat(first, counts)
        rindices[cursor[dst] + rank] = src
        cursor[unique_dst] += counts
    return rindptr, rindices


def iter_undirected_blocks(indptr: np.ndarray, indices: np.ndarray, rindptr: np.ndarray, rindices: np.ndarray,
                           block_edges: int = DEFAULT_BLOCK_EDGES
                           ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (a, b, local_indptr int64[b-a+1], columns uint32) for the undirected
    rows of nodes a..b-1: out-edges and in-edges packed into uint64
    ``(row - a) * n + column`` keys, sorted and de-duplicated; self loops are
    dropped.  Rows are therefore sorted ascending like ``sorted(set(...))``."""
    n = len(indptr) - 1
    _check_node_count(n)
    out_degree = _degrees(indptr)
    in_degree = _degrees(rindptr)
    combined = np.zeros(n + 1, dtype=np.uint64)
    np.cumsum(out_degree + in_degree, dtype=np.uint64, out=combined[1:])
    big_n = np.uint64(n)
    for a, b in row_blocks_by_edges(combined, block_edges):
        out_dst = indices[int(indptr[a]):int(indptr[b])]
        in_dst = rindices[int(rindptr[a]):int(rindptr[b])]
        local_ids = np.arange(b - a, dtype=np.uint64)
        src = np.concatenate((np.repeat(local_ids, out_degree[a:b]), np.repeat(local_ids, in_degree[a:b])))
        dst = np.concatenate((out_dst, in_dst)).astype(np.uint64)
        keep = (src + np.uint64(a)) != dst
        keys = np.unique(src[keep] * big_n + dst[keep])
        rows = keys // big_n
        columns = (keys - rows * big_n).astype(np.uint32)
        local_indptr = np.zeros(b - a + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows.astype(np.int64), minlength=b - a), out=local_indptr[1:])
        yield a, b, local_indptr, columns


def symmetrize_csr(indptr: np.ndarray, indices: np.ndarray,
                   block_edges: int = DEFAULT_BLOCK_EDGES) -> tuple[np.ndarray, np.ndarray]:
    """Undirected CSR (uint64 indptr, uint32 sorted indices) of A + A^T without self loops."""
    n = len(indptr) - 1
    rindptr, rindices = transpose_csr(indptr, indices, block_edges)
    uindptr = np.zeros(n + 1, dtype=np.uint64)
    blocks: list[np.ndarray] = []
    for a, b, local_indptr, columns in iter_undirected_blocks(indptr, indices, rindptr, rindices, block_edges):
        uindptr[a + 1:b + 1] = uindptr[a] + local_indptr[1:].astype(np.uint64)
        blocks.append(columns)
    uindices = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.uint32)
    return uindptr, uindices


def undirected_edge_cuts(indptr: np.ndarray, indices: np.ndarray, memberships: Sequence[np.ndarray],
                         block_edges: int = DEFAULT_BLOCK_EDGES) -> list[int]:
    """Undirected edge cut of every membership (unordered node pairs joined in
    either direction whose owners differ), from one block stream."""
    n = len(indptr) - 1
    memberships = [np.asarray(m) for m in memberships]
    for m in memberships:
        if m.shape != (n,):
            raise ValueError("membership length mismatch")
    cuts = [0] * len(memberships)
    if not memberships:
        return cuts
    rindptr, rindices = transpose_csr(indptr, indices, block_edges)
    for a, b, local_indptr, columns in iter_undirected_blocks(indptr, indices, rindptr, rindices, block_edges):
        rows = a + np.repeat(np.arange(b - a, dtype=np.int64), np.diff(local_indptr))
        upper = columns.astype(np.int64) > rows
        left = rows[upper]
        right = columns[upper].astype(np.int64)
        for k, m in enumerate(memberships):
            cuts[k] += int(np.count_nonzero(m[left] != m[right]))
    return cuts


# --------------------------------------------------------------------------
# METIS inputs
# --------------------------------------------------------------------------

def metis_int32_arrays(uindptr: np.ndarray, uindices: np.ndarray, vweights: np.ndarray | None = None,
                       eweights: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Cast a CSR to pymetis' int32 xadj/adjncy (+ optional weights), refusing
    anything METIS' 32-bit idx_t cannot address."""
    n = len(uindptr) - 1
    nnz = int(uindptr[-1])
    if n >= INT32_LIMIT:
        raise CsrScaleError(f"{n} nodes >= 2^31 do not fit pymetis' 32-bit idx_t; {METIS_64BIT_ADVICE}")
    if nnz >= INT32_LIMIT:
        raise CsrScaleError(f"undirected nnz {nnz} >= 2^31 does not fit pymetis' 32-bit idx_t "
                            f"(xadj overflows); {METIS_64BIT_ADVICE}")
    if len(uindices) != nnz:
        raise CsrFormatError("indptr / indices length mismatch")
    if nnz and int(uindices.max()) >= n:
        raise CsrFormatError("adjacency column outside the graph")
    xadj = uindptr.astype(np.int32)
    adjncy = uindices.astype(np.int32)
    vw = None
    if vweights is not None:
        vw64 = np.asarray(vweights, dtype=np.int64)
        if vw64.shape != (n,) or vw64.min(initial=0) < 0:
            raise CsrFormatError("vertex weights must be one non-negative integer per node")
        if int(vw64.sum()) >= INT32_LIMIT:
            raise CsrScaleError(f"vertex weight sum {int(vw64.sum())} >= 2^31 overflows METIS' idx_t; "
                                f"reduce the weight scale or use {METIS_64BIT_ADVICE}")
        vw = vw64.astype(np.int32)
    ew = None
    if eweights is not None:
        ew64 = np.asarray(eweights, dtype=np.int64)
        if ew64.shape != (nnz,) or ew64.min(initial=0) < 0 or (nnz and int(ew64.max()) >= INT32_LIMIT):
            raise CsrFormatError("edge weights must be one non-negative int32 per adjacency entry")
        ew = ew64.astype(np.int32)
    return xadj, adjncy, vw, ew


# --------------------------------------------------------------------------
# runtime_lib equivalents on CSR arrays
# --------------------------------------------------------------------------

def _interleave(headers: np.ndarray, payload: np.ndarray, header_words: int) -> np.ndarray:
    """Word stream of [header_i (header_words words) | row_i] records; ``headers``
    is (rows, header_words) and ``payload`` the concatenated rows in order."""
    degrees = headers[:, -1].astype(np.int64)
    record = header_words + degrees
    total = int(record.sum())
    out = np.empty(total, dtype="<u4")
    starts = np.zeros(len(degrees), dtype=np.int64)
    if len(degrees) > 1:
        starts[1:] = np.cumsum(record)[:-1]
    mask = np.ones(total, dtype=bool)
    for word in range(header_words):
        out[starts + word] = headers[:, word]
        mask[starts + word] = False
    out[mask] = payload
    return out


def csr_ordered_graph_hash(indptr: np.ndarray, indices: np.ndarray, block_rows: int = DEFAULT_BLOCK_ROWS) -> str:
    """Identical to ``runtime_lib.ordered_graph_hash(rows)``: SHA-256 over
    ``<II gid degree`` + row words for every node in id order."""
    digest = hashlib.sha256()
    n = len(indptr) - 1
    degrees = _degrees(indptr)
    for a in range(0, n, block_rows):
        b = min(n, a + block_rows)
        headers = np.stack((np.arange(a, b, dtype=np.int64), degrees[a:b]), axis=1)
        digest.update(_interleave(headers, indices[int(indptr[a]):int(indptr[b])], 2).tobytes())
    return digest.hexdigest()


def write_adj_csr(path: Path, indptr: np.ndarray, indices: np.ndarray, ids: np.ndarray, start_node: int,
                  max_degree: int, block_rows: int = DEFAULT_BLOCK_ROWS) -> int:
    """Byte-identical to ``runtime_lib.write_adj(path, [rows[g] for g in ids], start_node, max_degree)``."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0:
        raise RuntimeErrorChecked("cannot write empty partition")
    if not 0 <= start_node < len(ids):
        raise RuntimeErrorChecked("invalid partition start node")
    degrees = _degrees(indptr)
    size = 24 + int((4 + 4 * degrees[ids]).sum())
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<QIIQ", size, int(max_degree), int(start_node), 0))
        for a in range(0, len(ids), block_rows):
            chunk = ids[a:a + block_rows]
            deg = degrees[chunk]
            starts = indptr[chunk].astype(np.int64)
            exclusive = np.zeros(len(chunk), dtype=np.int64)
            if len(chunk) > 1:
                exclusive[1:] = np.cumsum(deg)[:-1]
            gather = np.repeat(starts - exclusive, deg) + np.arange(int(deg.sum()), dtype=np.int64)
            handle.write(_interleave(deg.reshape(-1, 1), indices[gather], 1).tobytes())
    return size


def csr_static_metrics(indptr: np.ndarray, indices: np.ndarray, membership: np.ndarray, parts: int,
                       block_edges: int = DEFAULT_BLOCK_EDGES) -> dict:
    """Identical to ``runtime_lib.static_metrics(rows, membership, parts)`` from CSR arrays."""
    n = len(indptr) - 1
    membership = np.asarray(membership).astype(np.int64, copy=False)
    if membership.shape != (n,):
        raise RuntimeErrorChecked("membership length mismatch")
    degree = _degrees(indptr)
    edge_counts = np.rint(np.bincount(membership, weights=degree, minlength=parts)).astype(np.int64)
    adjacency_bytes = 24 + np.rint(np.bincount(membership, weights=4 + 4 * degree, minlength=parts)).astype(np.int64)
    cross_counts = np.zeros(parts, dtype=np.int64)
    boundary_counts = np.zeros(parts, dtype=np.int64)
    for a, b in row_blocks_by_edges(indptr, block_edges):
        lo, hi = int(indptr[a]), int(indptr[b])
        if hi == lo:
            continue
        local_src = np.repeat(np.arange(b - a, dtype=np.int64), degree[a:b])
        src_owner = membership[a:b][local_src]
        cross = src_owner != membership[indices[lo:hi]]
        cross_counts += np.bincount(src_owner[cross], minlength=parts)
        per_node_cross = np.bincount(local_src[cross], minlength=b - a)
        boundary_counts += np.bincount(membership[a:b][per_node_cross > 0], minlength=parts)
    total_edges = int(degree.sum())
    node_counts = [int(np.sum(membership == part)) for part in range(parts)]
    return {
        "nodes": node_counts,
        "node_balance_max_over_mean": float(max(node_counts) / np.mean(node_counts)),
        "directed_edges": [int(x) for x in edge_counts],
        "adjacency_bytes": [int(x) for x in adjacency_bytes],
        "cross_directed_edges": [int(x) for x in cross_counts],
        "cross_ratio": float(cross_counts.sum() / total_edges) if total_edges else 0.0,
        "boundary_nodes": [int(x) for x in boundary_counts],
        "degree": {"p50": float(np.percentile(degree, 50)), "p95": float(np.percentile(degree, 95)),
                   "p99": float(np.percentile(degree, 99)), "max": int(degree.max(initial=0))},
    }
