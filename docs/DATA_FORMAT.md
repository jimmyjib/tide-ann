# Data formats

Every file the binaries read or write, with its type, shape, order and the tool that produces it. Ids are
global node ids: load-time ids are `0..N0-1` (row i of `base.npy` is id i), inserted ids continue from
`N0` in insertion order. Arrays are NumPy `.npy` files (format version 1, little-endian, C order).
Binary files are little-endian.

Notation: K is the number of servers. The result size of a search is the coordinator's `--K` flag
(top-k, default 10); G below is the number of ground-truth ids per row.

## Directory layouts

The data tools build this layout (as in `examples/demo.py`, dataset root = the working directory):

```
source/                      make_synthetic.py or slice_dataset.py
  base.npy  query.npy  insert.npy  base_plus_insert.npy  MANIFEST.json
index/                       build_vamana.py
  vamana  (+ diskannpy's other index files)  GRAPH_BUILD.json
partition/                   partition_graph.py (METIS)
  membership.npy  partition_<r>/{adj,local_to_global.npy}
ghost/ghost_indices.npy      make_ghost_ids.py
gt/base/groundtruth.npy                  generate_groundtruth.py on base.npy
gt/base-plus-insert/groundtruth.npy      generate_groundtruth.py on base_plus_insert.npy
runtime/                     seal_runtime.py: the coordinator's data directory
  base.npy  query.npy  groundtruth.npy  membership.npy  ghost/ghost_indices.npy
  partition_<r>/{adj,local_to_global.npy}  RUNTIME_MANIFEST.json  COMPLETE
row-bytes/directed_row_bytes.npy         extract_directed_row_bytes.py
stream/query.npy  stream/groundtruth.npy build_query_load_stream.py or build_skewed_query_stream.py
pq/                          tide-build-pq, write_partition_vectors.py
  pq_pivots.bin  pq_codes.bin  partition_<r>/vectors.npy  PQ_MANIFEST.json
PREP_COMPLETE.json           prepare_demo_runtime.py (steps partition .. stream)
```

`prepare_demo_runtime.py` runs `partition_graph`, `make_ghost_ids`, both ground truths,
`seal_runtime`, `extract_directed_row_bytes` and one stream builder in that order.

`examples/demo.py` also writes `pool-<c>/insert.npy` (one `make_synthetic` insert pool per hot cluster
c), `control/` (the query file and its ground truth, one query file per hot cluster, and the insert
batch `insert.npy` it chooses while the cluster runs) and `gt/after-insert/groundtruth.npy` (the ground
truth on the base plus the batch); its `source/insert.npy` holds one unused vector.

**Coordinator data directory** (`--data-dir`): `base.npy`, `query.npy`, `groundtruth.npy`,
`membership.npy`, `ghost/ghost_indices.npy`. (`runtime/` holds them; the partition files there are not
read by the coordinator.)

**Server r data directory** (`--data-dir`): `membership.npy`, `partition_<r>/adj`,
`partition_<r>/local_to_global.npy`, and under `--pq` `pq_pivots.bin`, `pq_codes.bin`,
`partition_<r>/vectors.npy`, under `--full-vectors` `base.npy`. A server reads only its own
`partition_<r>/`; one directory may serve every rank. The layout is flat; its files come from:

| Server file | Source |
|---|---|
| `membership.npy`, `partition_<r>/adj`, `partition_<r>/local_to_global.npy` | `runtime/` |
| `pq_pivots.bin`, `pq_codes.bin`, `partition_<r>/vectors.npy` | `pq/` (`--pq`) |
| `base.npy` | `runtime/` (`--full-vectors`) |

The README's "Prepare a dataset" section links them into one directory; `examples/demo.py` links one
directory per server (`views/server-<r>/`).

## Files read by the binaries

| File | Type and shape | Contents | Read by | Produced by |
|---|---|---|---|---|
| `base.npy` | float32 [N0, dim] | the load-time vectors; row i = id i | coordinator; servers under `--full-vectors` | `make_synthetic.py`, `slice_dataset.py` (`source/`), copied by `seal_runtime.py` |
| `query.npy` | float32 [Q, dim] | the default query set of `recall` (the first `--queries` rows are loaded) | coordinator | `make_synthetic.py`, `slice_dataset.py`; a load stream from `build_*_stream.py` |
| `groundtruth.npy` | int32, uint32 or int64 [Q, G], G >= the result size `--K` | row i: the ids of query i's exact nearest neighbours, nearest first; recall@`--K` uses the first `--K` columns. Checked: dtype, 2D shape, a row for every loaded query the command scores, at least `--K` columns, and the file length | coordinator (`recall` default) | `generate_groundtruth.py` (int32) |
| `membership.npy` | int32 or uint32 [N0] (the coordinator also accepts int64; the servers refuse any other dtype, an int64 file included) | the owner rank of every load-time id, each below the server count K | coordinator, every server | `partition_graph.py`, `seal_runtime.py` |
| `ghost/ghost_indices.npy` | int64 or int32 [H] | the head-index sample (load-time ids) | coordinator | `make_ghost_ids.py` (1 %, seed 42 in the demo) |
| `partition_<r>/local_to_global.npy` | uint32 or int32 [n_r] (any other dtype is refused) | the ids server r owns, ascending; row i of `adj` and of `vectors.npy` is id `local_to_global[i]` | server r | `partition_graph.py`, `seal_runtime.py` |
| `partition_<r>/adj` | binary | header: u64 file size, u32 maximum degree, u32 start node, u64 reserved; then for each of the n_r rows in `local_to_global` order: u32 degree, degree x u32 neighbour ids (global ids, any owner) | server r | `partition_graph.py`, `seal_runtime.py` |
| `partition_<r>/vectors.npy` | float32 [n_r, dim] | the original vectors of the ids server r owns, in `local_to_global` order | server r under `--pq` | `write_partition_vectors.py` |
| `pq_pivots.bin` | binary | u32 magic 0x50515442 (bytes on disk `42 54 51 50`), u32 dim, u32 M, u32 centroid count (256), then float32 centroids [M][256][dim / M] | servers under `--pq` | `tide-build-pq` |
| `pq_codes.bin` | binary | u32 magic 0x44435150 (bytes on disk `50 51 43 44`), u32 n, u32 M, u32 0, u64 pivots hash (FNV-1a 64 over dim, M, the centroid count as u32 and the centroid bytes), then n x M uint8 codes, row i = id i | servers under `--pq` (n must equal N0) | `tide-build-pq --codes` |

Both PQ magics are read as little-endian u32 values and compared as numbers. Every server checks that
`membership.npy` and `local_to_global.npy` are 1D int32 or uint32 arrays as long as their shape, that
the owner labels are below `--num-servers`, that `pq_codes.bin` matches
`membership.npy` and the pivots, and that `vectors.npy` matches `local_to_global.npy` (row count,
dimension and order, [STORAGE_PQ.md](STORAGE_PQ.md)). `base.npy`, `query.npy`, the insert files and
`recall --query-file` are read without a dtype, memory-order or length check: they must be float32 C-order
matrices of the size their header states.

## Inputs of admin commands

| Argument | Type and shape | Contents |
|---|---|---|
| `insert --file F`, `mixed --insert-file F` | float32 [rows, dim] | vectors to insert; `--start-id S` / `--insert-start-id S` is the first row used; the ids assigned continue after the ids the coordinator knows |
| `recall --query-file F` | float32 [rows, dim], not checked | queries instead of `query.npy`; `--query-offset` skips rows |
| `recall --groundtruth F`, `mixed --groundtruth F` | int32, uint32 or int64 [rows, G], G >= `--K`, a row for every loaded query the command scores | ground truth of the queries used; after inserts compute it on base + inserted vectors with the inserted ids (`base_plus_insert.npy`). A file that is malformed or does not cover the queries is refused before any query is sent ([ADMIN_COMMANDS.md](ADMIN_COMMANDS.md#recall-alias-baseline)) |
| `recall --self-ids F` | uint32 or int32 [n] (any other dtype is refused) | row i of `--query-file` is the vector of id `F[i]`; reports where each id is found |
| `mixed --insert-gt F` | int32, uint32 or int64 [rows, G], G >= `--K` | ground truth of the inserted vectors used as queries |
| `migration-apply <plan>` | binary plan file (below) | a MOVE plan |
| `controller-start <journal_dir>` | directory | must exist and be writable by the coordinator |

## Files written by the binaries

| File | Type | Contents | Writer |
|---|---|---|---|
| `membership-check --write F` | uint32 [N] `.npy` | the coordinator's membership, one owner per id (load-time and inserted) | coordinator |
| `<journal_dir>/controller-journal.jsonl` | JSON lines | one event per controller transition ([CONTROLLER.md](CONTROLLER.md)) | coordinator |
| `<journal_dir>/window-<epoch>/plan.bin`, `revert-plan.bin` | migration plan files (version-2 skeletons) | records of the controller's bursts and their inverses | coordinator |
| `debug_<data-dir name>_<mode><n>_<time>.csv`, `..._hops.csv` | CSV | per-query debug output of `--debug` (the `_hops.csv` file holds its header only), in the coordinator's working directory ([DEBUG.md](DEBUG.md)) | coordinator |
| `debug_results.npy` | uint32 [completed searches, `--K`] `.npy` | the top-`--K` ids of every completed search of a `--debug` `recall` or `mixed` command (UINT32_MAX pads a short result), in the coordinator's working directory; each such command overwrites it | coordinator |
| `pq_pivots.bin`, `pq_codes.bin` | binary | above | `tide-build-pq` |

## Migration plan files

`migration-apply` reads a plan file. The only plan files this repository writes are the controller's
records `window-<epoch>/plan.bin` and `revert-plan.bin`: version-2 skeletons whose moves carry no rows or
vectors, so applying one needs `--refresh`. Like any plan, it applies only while its `epoch_from` equals
the coordinator's current ownership epoch and its from servers match the current membership; once the
controller has committed or reverted that burst, the file is a record only. No tool writes a plan for
manual use. Layout (little-endian):

```
8 bytes  "TIDEMIG\0"
u32      version: 1, 2, 3 or 4
u32      mode = 0 (MOVE)                         versions >= 2
u32      vector dimension (> 0)                  versions >= 3
u32      vector target server                    version 4
u32      server count (must equal K, the server count)
u64      epoch_from (the coordinator's current ownership epoch)
u64      epoch_to = epoch_from + 1
u64      move count (>= 1)
per move:
  u32 id, u32 from server, u32 to server, u32 degree, u64 row digest
  degree x u32 neighbour ids
  u64 vector digest                              versions >= 3
  dim x float32 vector                           version 3; version 4 only for moves to the vector target
```

Ids and neighbours must be below the coordinator's node count (load-time plus inserted ids); an id may appear once; from and to must
differ. Under `--pq` a plan must carry each moved node's vector (version 3, and it must equal the
original byte for byte) or be applied with `--refresh`, which takes the current rows and vectors from
the servers; versions 1 and 2 apply as written only under `--full-vectors`. The digests are defined in
`src/tide/include/membership_migration_plan.h`.

## Tool outputs used for checking

| File | Type | Producer | Use |
|---|---|---|---|
| `source/base_plus_insert.npy` | float32 [N0 + M, dim] | `make_synthetic.py`, `slice_dataset.py` | ground truth after the inserts (row N0 + i = insert row i) |
| `source/{base,query,insert}_labels.npy` | int32 | `make_synthetic.py` | the synthetic cluster of each row |
| `row-bytes/directed_row_bytes.npy` | int64 [N0] | `extract_directed_row_bytes.py` | 4 + 4 x out-degree per node, the adjacency part of the controller's row bytes |
| `stream/query.npy`, `stream/groundtruth.npy` | float32, int32 | `build_query_load_stream.py` (seeded shuffles of the queries) or `build_skewed_query_stream.py` (most rows from a few hot query regions) | a long search load for a controller phase |
| `*MANIFEST.json`, `COMPLETE`, `GRAPH_BUILD.json`, `PREP_COMPLETE.json` | JSON | the tools | sha256 and parameters of every output; a tool refuses to overwrite an existing output |

`normalize_vectors.py` writes unit-length float32 copies of float32 or float64 inputs (for cosine data);
`make_biased_inserts.py` selects an insert set near hot query regions from a larger vector set.
