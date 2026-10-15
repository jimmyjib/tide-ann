# Storage modes

Every server of a cluster runs one of two storage modes, chosen at start-up. The coordinator rejects a cluster
whose servers differ in mode or in vector dimension. With `--pq` (default), each server keeps the original
vectors of its own nodes and PQ codes of all nodes. This is PQ mode below. With `--full-vectors`, each server
keeps every vector (`base.npy`).

| | PQ mode (`--pq`) | `--full-vectors` |
|---|---|---|
| original vectors on server r | the nodes r owns | every node |
| PQ codes on server r | every node | none |
| neighbor scoring | exact if owned, ADC otherwise | exact |
| expanded nodes | re-scored exactly on their owner | already exact |
| returned results | exact distances | exact distances |
| a MOVE sends | rows to all servers, vectors to the target | rows to all servers |
| files the server loads | `membership.npy`, `partition_<r>/`, `pq_pivots.bin`, `pq_codes.bin` | `membership.npy`, `partition_<r>/`, `base.npy` |

PQ is standard product quantization without rotation, ported from DiskANN ([NOTICE](../NOTICE)). A vector is
cut into M equal chunks of dim / M floats, so M must divide the dimension. Each chunk is quantized to one of
256 centroids, so a code is M bytes. All distances are squared L2, and PQ distances approximate them.

## Scoring in PQ mode

When a server expands a node it owns, it scores the node's out-neighbors against the query:

- exact, when the server owns the neighbor and holds its original vector
- ADC (asymmetric distance), using a per-query table of M x 256 entries, when it holds only the code
- missing, when this server has not finished storing an inserted id's code (a backlink can link the id
  while another message for the id is still writing the code): infinite distance, no work counted

A beam entry records whether its distance is exact. When the owner expands an entry with an estimated
distance, it first re-scores the entry with the original vector, which is called a re-rank. When the search
ends, entries without an exact distance are dropped, so the results carry exact distances. Under `--pq`, a
search with a node's own vector as the query does not always find that node at L = 50, because the other
candidates are ranked by approximate PQ distances. A larger search list (`recall --search-l`) finds more of
them.

Insert and backlink pruning use the same rule for a pair of nodes: exact when both vectors are present, ADC
when one is, and a precomputed M x 256 x 256 code-to-code table when neither is. W(v) counts exact, ADC and
re-rank evaluations ([controller.md](controller.md)). `stats` reports them per server (`search.exact_count`,
`adc_count`, `rerank_count`, `missing_code_count`, see [metrics.md](metrics.md)).

## Inserts and moves in PQ mode

- An insert's home server stores the new vector and its code. Every other server stores only the code,
  encoded from the broadcast or backlink that carries the vector. A code is written once.
- A MOVE sends a moved node's original vector to its new owner only. The source checks the vector digest
  against its own copy, and ACCEPT frees the source's row and vector slot for later inserts and moves.
  `membership-check` verifies that every server holds the original vectors of the nodes it owns.
- The controller's byte budgets count dim x 4 vector bytes per node, because a MOVE sends the vector.

## Memory

Per server in PQ mode: dim x 4 bytes per owned node, M bytes per node of the whole index, the centroids
(dim x 256 x 4 bytes), the code-to-code table (M x 256 x 256 x 4 bytes) and the adjacency rows of the owned
nodes. With `--full-vectors`: dim x 4 bytes per node of the whole index plus the adjacency rows. Stores grow
in chunks of 100,000 rows, so small partitions can use as much memory as full replication. `stats` reports
each server's allocated `storage.vector_bytes`, `code_bytes` and `pq_table_bytes`. The coordinator keeps
every vector of the dataset in both modes, for the head index, `recreate-head`, `reset` and
`membership-check`.

## Files

| File | Written by | Read by |
|---|---|---|
| `pq_pivots.bin` | `tide-build-pq` | every server in PQ mode |
| `pq_codes.bin` | `tide-build-pq --codes` | every server in PQ mode |
| `partition_<r>/vectors.npy` | `tools/prepare/write_partition_vectors.py` | server r in PQ mode |

At start-up a server checks that the codes, pivots and `vectors.npy` match, and stops with a hint to
`--full-vectors` if a PQ file is missing. The formats are in [data_format.md](data_format.md).
`tide-build-pq <base.npy> <pivots> [M] [iters] [--codes <codes>]` runs seeded k-means per chunk on up to
256,000 stride-sampled base rows, 15 iterations by default. M defaults to dim / 8 when 8 divides the
dimension. It also prints the mean relative error of ADC against exact distances on sampled pairs.
