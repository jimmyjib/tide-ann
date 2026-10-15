# Data formats

These are the files the binaries and tools read or write. Ids are global node ids: load-time ids are
`0..N0-1` (row i of `base.npy` is id i), and inserted ids continue from `N0` in insertion order. Arrays are
NumPy `.npy` files (format version 1, little-endian, C order). Binary files are little-endian.

## Directory layouts

The data tools build this layout under one dataset root ([prepare_dataset.md](prepare_dataset.md)):

```
source/                      make_synthetic.py or slice_dataset.py
  base.npy  query.npy  insert.npy  base_plus_insert.npy  MANIFEST.json
index/                       build_vamana.py
  vamana  (+ diskannpy's other index files)  GRAPH_BUILD.json
partition/                   partition_graph.py (METIS)
  membership.npy  partition_<r>/{adj,local_to_global.npy}  PARTITION_BUILD.json
ghost/ghost_indices.npy      make_ghost_ids.py
gt/base/groundtruth.npy                  generate_groundtruth.py on base.npy
gt/base-plus-insert/groundtruth.npy      generate_groundtruth.py on base_plus_insert.npy
runtime/                     write_runtime.py: the coordinator's data directory
  base.npy  query.npy  groundtruth.npy  membership.npy  ghost/ghost_indices.npy
  partition_<r>/{adj,local_to_global.npy}  RUNTIME_MANIFEST.json
stream/query.npy  stream/groundtruth.npy build_query_load_stream.py or build_skewed_query_stream.py
pq/                          tide-build-pq, write_partition_vectors.py
  pq_pivots.bin  pq_codes.bin  partition_<r>/vectors.npy
```

The coordinator reads `runtime/` except the partition files. Server r reads `membership.npy`,
`partition_<r>/adj` and `partition_<r>/local_to_global.npy` from `runtime/`, plus `pq_pivots.bin`,
`pq_codes.bin` and `partition_<r>/vectors.npy` from `pq/` under `--pq`, or `base.npy` under
`--full-vectors`. It reads only its own `partition_<r>/`, so one flat directory can serve every rank.

## Files read by the binaries

| File | Type and shape | Contents | Read by |
|---|---|---|---|
| `base.npy` | float32 [N0, dim] | load-time vectors, row i = id i | coordinator, servers under `--full-vectors` |
| `query.npy` | float32 [Q, dim] | default queries of `recall` | coordinator |
| `groundtruth.npy` | int32, uint32 or int64 [Q, G], G >= `--K` | ids of each query's nearest neighbors, nearest first | coordinator |
| `membership.npy` | int32 or uint32 [N0] | owner rank of every load-time id, below K | coordinator, every server |
| `ghost/ghost_indices.npy` | int64 or int32 [H] | head-index sample of load-time ids ("ghost" in file and tool names) | coordinator |
| `partition_<r>/local_to_global.npy` | uint32 or int32 [n_r] | ids server r owns, ascending | server r |
| `partition_<r>/vectors.npy` | float32 [n_r, dim] | original vectors of server r's ids, same order | server r under `--pq` |

The adjacency file and the PQ files (servers under `--pq`) are binary:

```
partition_<r>/adj   u64 file size, u32 maximum degree, u32 start node, u64 reserved, then per row in
                    local_to_global order: u32 degree, degree x u32 neighbor ids (global, any owner)
pq_pivots.bin       u32 magic 0x50515442, u32 dim, u32 M, u32 256, float32 centroids [M][256][dim / M]
pq_codes.bin        u32 magic 0x44435150, u32 n (= N0), u32 M, u32 0, u64 pivots hash, uint8 [n][M]
```

Recall@`--K` uses the first `--K` columns of the ground truth. Row i of `adj` and of `vectors.npy` is id
`local_to_global[i]`, and row i of the codes is id i. The coordinator also accepts an int64 `membership.npy`.
Vector files (`base.npy`, `query.npy`, insert files) must be float32 in C order. Their dtype is not checked.
The pivots hash is FNV-1a 64 over dim, M and 256 (each a u32), then the centroid bytes. At start-up every
server checks that `membership.npy` and `local_to_global.npy` are 1-D int32 or uint32 arrays, that owner
labels are below `--num-servers`, and that the PQ files and `vectors.npy` match
([pq_storage.md](pq_storage.md)). Under `--full-vectors` it checks only that `base.npy` has one row per
`membership.npy` entry.

Admin commands read insert files (`insert --file`, `mixed --insert-file`) and `recall --query-file` as float32
[rows, dim]. Ground truth files (`--groundtruth`, `mixed --insert-gt`) are int32, uint32 or int64 [rows, G]
with G >= `--K`. `recall --self-ids` takes a uint32 or int32 [n] id list, and `migration-apply` a plan file
(below). After inserts, compute ground truth on the base plus the inserted vectors (`base_plus_insert.npy`).

## Files written by the binaries

| File | Type | Contents | Writer |
|---|---|---|---|
| `membership-check --write F` | uint32 [N] `.npy` | one owner per id, load-time and inserted | coordinator |
| `<journal_dir>/controller-journal.jsonl` | JSON lines | one event per controller transition ([controller.md](controller.md#journal)) | coordinator |
| `<journal_dir>/window-<epoch>/plan.bin`, `revert-plan.bin` | plan files, version 2 | the controller's batches and their inverses | coordinator |
| `debug_<data-dir name>_<mode><n>_<time>.csv` | CSV | per-query debug output in the working directory ([debug.md](debug.md)) | coordinator |
| `debug_results.npy` | uint32 [searches, `--K`] `.npy` | top ids per search, overwritten per command | coordinator |

## Migration plan files

A plan file holds one MOVE. It is version 2, or version 3 when every move also carries the node's original
vector. The controller writes version 2 plans whose moves carry no rows, so applying one of them by hand
needs `--refresh`. A plan applies only while its `epoch_from` equals the coordinator's ownership epoch and
its source servers match the current owners. Layout, little-endian:

```
8 bytes  "TIDEMIG\0"
u32      version: 2, or 3 with vectors
u32      mode = 0 (MOVE)
u32      vector dimension (> 0)                  version 3
u32      server count (= K)
u64      epoch_from (the coordinator's current ownership epoch)
u64      epoch_to = epoch_from + 1
u64      move count (>= 1)
per move:
  u32 id, u32 from server, u32 to server, u32 degree, u64 row digest
  degree x u32 neighbor ids
  u64 vector digest, dim x float32 vector        version 3
```

Ids and neighbors must be below the coordinator's node count, an id may appear once, and from and to must
differ. Under `--pq` a plan must be version 3 or be applied with `--refresh`, which takes the current rows
and vectors from the servers. Under `--full-vectors` a version 2 plan applies as written. The digests are
defined in `src/tide/include/membership_migration_plan.h`.

## Tool outputs

The tools also write `source/base_plus_insert.npy` (float32 [N0 + M, dim], row N0 + i is insert row i),
`source/{base,query,insert}_labels.npy` (int32, the synthetic cluster of each row) and the query streams in
`stream/`, a long search load for a controller phase. Some tools also write a JSON file with their
parameters and statistics (`MANIFEST.json`, `GRAPH_BUILD.json`, `PARTITION_BUILD.json`,
`RUNTIME_MANIFEST.json`). The Python tools refuse to overwrite an existing output.
