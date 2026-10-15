# Storage modes: hybrid PQ own-only storage and full replication

Every server of a cluster runs one of two storage modes, chosen at start-up. The coordinator refuses a
cluster whose servers disagree on the mode or on the vector dimension.

| | `--pq` (default): hybrid PQ own-only | `--full-vectors`: full replication |
|---|---|---|
| original vectors on server r | only the nodes r owns | every node |
| PQ codes on server r | every node | none |
| neighbour scoring | exact for owned neighbours, PQ (ADC) for the others | exact |
| expanded nodes | re-scored exactly on their owner | exact already |
| returned results | exact distances | exact distances |
| a MOVE ships | the move list with every adjacency row and its digests to every server; each original vector to its target only | the move list with every adjacency row and its digest to every server |
| files the server loads | `membership.npy`, `partition_<r>/{adj,local_to_global.npy,vectors.npy}`, `pq_pivots.bin`, `pq_codes.bin` | `membership.npy`, `partition_<r>/{adj,local_to_global.npy}`, `base.npy` |

Product quantization (PQ) here is standard PQ (no rotation), ported from the DiskANN reference
implementation ([NOTICE](../NOTICE)): the vector is cut into M equal chunks of dim / M floats (M must
divide the dimension), each chunk is quantized to one of 256 centroids, so a code is M bytes. Distances
are squared L2 everywhere; PQ distances approximate squared L2.

## Scoring under `--pq`

When a server expands a node it owns, it scores the node's out-neighbours against the query
(`hybrid_score` in `src/tide/include/server_types.h`):

- **exact** when the server holds the neighbour's original vector (it owns the neighbour);
- **ADC** (asymmetric distance: the query against the neighbour's PQ code, through a per-query table of
  M x 256 entries built on first use) when it holds only the code;
- **missing** when the code of a just-inserted id has not arrived yet (its broadcast is still queued):
  the neighbour gets an infinite distance and costs no work.

A beam entry carries a flag saying whether its distance is exact. When the owner expands an entry whose
distance is an estimate, it re-scores the entry with the original vector first (a **re-rank**). When the
search ends, entries still without an exact distance are dropped, so the results returned to the
coordinator carry exact distances.

Graph maintenance under `--pq` (insert pruning, backlink pruning) uses the same rule for a pair of nodes:
exact when both vectors are present, ADC when one is, code-to-code distance (a precomputed M x 256 x 256
table) when neither is.

The controller's weight W(v) counts every distance evaluation of an expansion: exact, ADC and the
re-rank ([CONTROLLER.md](CONTROLLER.md)). The `stats` reply reports them per server
(`search.exact_count`, `adc_count`, `rerank_count`, `missing_code_count`, `adc_table_count`;
[METRICS.md](METRICS.md)).

## Inserts and moves under `--pq`

- **Insert.** The home server stores the new vector in its own store and the code; every other server
  stores only the code, from the broadcast or backlink that carries the vector. A code is written once
  (the first delivery encodes it, later ones skip).
- **MOVE.** PREPARE sends every server the whole move list (every moved node's row with its row and
  vector digests) and each moved node's original vector to its target only; the source checks the vector
  digest against its own copy; ACCEPT frees the source's row and vector slot into a pool that later
  inserts and moves reuse. `membership-check` verifies afterwards that every server holds exactly the
  original vectors of the nodes it owns (byte for byte, against the coordinator's copy).
- The budgets of the controller count dim x 4 vector bytes per node under `--pq`, since a MOVE ships them.

## Memory

Per server, under `--pq`: dim x 4 bytes per owned node (original vectors) + M bytes per node of the whole
index (codes) + the PQ tables (centroids, dim x 256 x 4 bytes, and the code-to-code table, M x 256 x 256
x 4 bytes) + the adjacency rows of the owned nodes. Under `--full-vectors`: dim x 4 bytes per node of the
whole index + the adjacency rows.

The stores grow in chunks of 100,000 rows (own vectors) or ids (codes, mappings), so a small partition
pays for a whole chunk: on a 20,000-node dataset own-only storage uses about as much or more resident
memory than full replication; the saving appears once partitions fill their chunks. The `stats` reply
reports each server's allocated `storage.vector_bytes`, `code_bytes` and `pq_table_bytes`.

The coordinator keeps every original vector in both modes (`base.npy` and the inserted vectors), for the
head index, `recreate-head`, `reset` and `membership-check`.

## Files

| File | Written by | Read by |
|---|---|---|
| `pq_pivots.bin` | `tide-build-pq` | every server under `--pq` |
| `pq_codes.bin` | `tide-build-pq --codes` | every server under `--pq` |
| `partition_<r>/vectors.npy` | `tools/prepare/write_partition_vectors.py` | server r under `--pq` |
| `PQ_MANIFEST.json` | `write_partition_vectors.py` | the tools (identity of the PQ directory) |

Formats are in [DATA_FORMAT.md](DATA_FORMAT.md). At start-up a server checks that `pq_codes.bin` has
one code per id of `membership.npy`, the pivots' M and the pivots' hash (codes encoded with other pivots
are refused), that `vectors.npy` has `local_to_global.npy`'s row count and the pivots' dimension, and
that its rows are in `local_to_global` order (a sample of up to 1,024 rows must encode to their nodes'
stored codes, at least 99 % of them). A missing file stops the server with a hint to `--full-vectors`;
there is no fallback to `base.npy`.

Training (`tide-build-pq <base.npy> <pivots> [M] [iters] [--codes <codes>]`): k-means per chunk on up
to 256,000 rows of the base (stride-sampled), 15 iterations by default, seeded and deterministic; M
defaults to dim / 8 when 8 divides the dimension. It also prints a self-test: the mean relative error
of ADC against exact distances on sampled pairs.

## Evidence

Hybrid PQ own-only storage as implemented here has been run locally only (C++ unit tests, local runs on a
200,000 x 96 slice of real vectors, and the demo); it has not been measured on a serving cluster.

Two earlier implementations of it were measured on 4 servers with P2P search, on one
128-dimensional dataset (SIFT, 500,000 vectors). Both stored a server's own original vectors and a code
of every node, and both used about 75 % less memory per server than full replication. Both were slower
than full replication:

- the first used the same scoring rule as this code (exact for owned neighbours, ADC for the others, an
  exact re-rank of every expanded node); it needed about 14 % more CPU per query and more hops per
  query (about 2.2 against 1.9, hops counting every server visit);
- the second used ADC for all navigation and exact distances only to re-rank expanded nodes; it lost
  about 2.4 recall points (about 96.5 % against 99 % recall@10) and searched and inserted more slowly.

The likely reasons: at this low dimension an exact distance is already cheap, so building the ADC tables
and the exact re-ranks cost more than the exact distances they replace; and the PQ estimates made routing
worse, so a query was handed between servers more often (more hops) even though it computed slightly
fewer distances. Whether PQ storage is faster at a high dimension has not been measured. Every
measurement of the runtime controller
and of the static placements used full replication. See [LIMITATIONS.md](LIMITATIONS.md).
