# Architecture

One graph ANN index, partitioned across K index servers, that keeps accepting inserts and repartitions
itself at runtime. This document describes the processes, what each one holds, and how searches,
inserts, drains and node moves flow between them. The repartitioning controller has its own document,
[CONTROLLER.md](CONTROLLER.md); storage modes are in [STORAGE_PQ.md](STORAGE_PQ.md) and the input files
in [DATA_FORMAT.md](DATA_FORMAT.md).

Notation: K is the number of servers. The coordinator's flag `--K` is a different quantity, the result
size (top-k, default 10). W(v) is the search work of node v and W[p] the load of server p; the flag `--W`
is the beam width.

## Processes

| Process | Count | Role |
|---|---|---|
| `tide-server` | K (2 to 64; runs used 4 and 8, [LIMITATIONS.md](LIMITATIONS.md)) | holds one partition; expands search states, links inserted nodes, applies backlinks and broadcasts, counts per-node work, executes MOVEs |
| `tide-coordinator` | 1 | head index, query dispatch and result collection, the insert pipeline, `drain`, the membership and its ownership epoch, the repartitioning controller, the admin socket |
| admin client | any | sends one text command per request (`tools/coord_client.py`, `tools/local_cluster.py send`, or any ZeroMQ REQ socket) |

Sockets (ZeroMQ over TCP):

- every server binds a ROUTER socket on `--port` (default 29600; the examples use 28800);
- the coordinator connects one DEALER socket to every server (`--servers`, rank order);
- with P2P search every server also connects a DEALER socket to every peer (`--peers`, rank order, its
  own address included), and forwards search states directly;
- the coordinator binds a REP socket on `tcp://*:29699` for admin commands. A command is one request,
  its reply one message.

The coordinator is single-command: it handles one admin command at a time, and every workload
(`recall`, `insert`, `mixed`) runs to completion inside its command.

## What each process holds

**Server r** (its data directory, see [DATA_FORMAT.md](DATA_FORMAT.md)):

- `membership.npy`: the owner of every load-time id. Every server keeps the owner of every id it knows
  (load-time ids, and inserted ids through the broadcasts), so it can tell where a neighbour lives.
- its partition: `partition_<r>/adj` (the out-neighbour rows of the nodes it owns, as global ids) and
  `partition_<r>/local_to_global.npy` (row i of `adj` is node `local_to_global[i]`);
- vectors, by storage mode: under `--pq` (the default) the original vectors of its own nodes
  (`partition_<r>/vectors.npy`) and a PQ code of every node (`pq_codes.bin`, `pq_pivots.bin`); under
  `--full-vectors` every vector (`base.npy`);
- per-node work counters for the controller's windows, and a pool of free row slots (rows that an
  accepted MOVE gave away; later inserts and moves reuse them).

**Coordinator** (its data directory):

- `base.npy` (every load-time vector) and every inserted vector, `query.npy`, `groundtruth.npy` (the
  default ground truth of `recall`), `membership.npy`, and the head-index sample
  `ghost/ghost_indices.npy`;
- the membership (owner of every id, load-time and inserted) and its ownership epoch (raised by every
  committed MOVE);
- the controller's runtime, history and journal while a controller phase runs.

The coordinator therefore holds every vector of the dataset; the servers do not (under `--pq`).

## Search

1. **Entry.** The head index is a fixed sample of the load-time ids (`ghost/ghost_indices.npy`; 1 % in
   the prepared data), following the BatANN design. The coordinator finds the sample node nearest to the query by brute
   force, makes it the entry point and sends the search state to the server that owns it.
2. **Expansion.** A server expands the unexpanded beam entries it owns (up to the beam width, which the
   coordinator's `--W` sets and every search state carries): it
   reads their adjacency rows, scores the neighbours (exactly, or with PQ codes under `--pq`, see
   [STORAGE_PQ.md](STORAGE_PQ.md)) and merges them into the beam (search list size `L`).
3. **Forwarding.** When the best unexpanded entries belong to another server, the state (beam, visited
   set, route) moves there: directly from server to server with P2P (`--p2p` on the coordinator and
   `--peers` on every server), or back through the coordinator without it. Each server visit, the entry
   server included, counts as one hop: a query that never leaves its entry server has 1 hop, and the
   number of server-to-server moves is hops - 1.
4. **Done.** When no unexpanded entry is left in the beam, the server returns the top results (`--K` of
   them) to the coordinator. The coordinator may merge vectors it has inserted but not yet seen applied
   into the result (`merge_search`; `recall --no-merge` turns it off) and scores recall@`--K` against the
   ground truth.

The work of step 2 is what the controller measures: W(v) is the number of distance evaluations spent
when node v is expanded on its owner.

## Insert

`insert N --file F` inserts rows of F; the new ids continue after the ids the coordinator knows
(`n_base + inserted so far`). Each insert goes through four phases (details and timings in
[METRICS.md](METRICS.md)):

1. **Search.** A search for the new vector (insert parameters `--insert-L`, `--insert-W`).
2. **Link.** The coordinator sends the vector and the candidates to the owner of the nearest result,
   which allocates a row (a free slot first, else a new one), selects the out-neighbours with
   alpha-RNG pruning (`--insert-R`, `--insert-alpha`) and updates its own neighbours' rows. A candidate
   this server has not received yet (an id another server inserted whose broadcast or backlink is still
   queued here) is dropped and counted (`insert_neighbors_undelivered` in `stats`): it has no owner label,
   vector or code here, so it is never read and never linked.
3. **Backlinks.** Every other server that owns a new neighbour adds the back edge (pruning the row when
   it grows beyond R plus `--edge-overload`).
4. **Broadcast.** Every server learns the new id's owner and stores its vector (`--full-vectors`) or
   its PQ code (`--pq`).

Servers apply inserts, backlinks and broadcasts in batches (`--insert-batch-size`,
`--backlink-batch-size`, `--broadcast-batch-size`). A partial batch waits for a full one or for a
drain, so the graph seen by searches can lag the inserts. Several worker threads apply batches while
searches run: a node's vector or code that arrives twice (its backlink and its broadcast) is written
once, and the later delivery waits until that write is complete. Searches and prunes read an inserted
node's replicated vector only once that write is complete (under `--full-vectors`; under `--pq` its code
is read only once it is stored).

## Drain

`drain` (and every command that needs a quiescent cluster: `controller-start`, `membership-check`,
`migration-*`, `shutdown`) waits until every pending insert reply, backlink and broadcast is applied and
acknowledged on every server, then reports what it settled.

## Membership, ownership epoch and MOVE

A node's owner is the server holding its adjacency row (and, under `--pq`, its original vector). The
membership changes only through a MOVE, a stop-the-world transaction over all servers:

1. **REFRESH** (controller and `migration-apply --refresh`): the coordinator pulls each moved node's
   current row (and, under `--pq`, its vector) from its current owner.
2. **PREPARE**: every server receives the whole move list, with every moved node's row and row digest
   (under `--pq` also its vector digest), and checks it against its own membership; under `--pq` each
   moved node's original vector goes to its target server only. A target installs the rows (and
   vectors) of the nodes moving to it in free or new row slots. A source checks each row digest, and
   under `--pq` each vector digest, against its own copy before anything changes.
3. **COMMIT**: every server publishes the new owners.
4. **ACCEPT**: the move becomes final; sources free the moved rows and vectors into their slot pool.
   **ROLLBACK** (before ACCEPT) restores the previous state.

The ownership epoch increases with every committed MOVE; servers refuse a plan for a different epoch.
The controller issues MOVEs itself (its fence, [CONTROLLER.md](CONTROLLER.md)); `migration-apply`,
`migration-accept` and `migration-rollback` issue them by hand ([ADMIN_COMMANDS.md](ADMIN_COMMANDS.md)).
`membership-check` verifies after the fact that every server agrees with the coordinator and, under
`--pq`, holds exactly the original vectors of the nodes it owns.

## Phases

Inserts and the controller alternate:

```
insert phase:      insert ..., insert ..., drain           (searches allowed)
controller phase:  controller-start  ->  search load  ->  controller-stop
insert phase:      insert ...                              (reuses slots freed by accepted MOVEs)
...
```

`controller-start` drains, then freezes the node count N of its windows (every id the coordinator knows);
inserts, `mixed`, `recreate-head`, `reset` and `migration-apply` are refused until `controller-stop`.
The controller acts only while a search command runs, because each window measures the work of the
current search load.

`reset` returns every server and the coordinator to the load-time state (load-time membership, no
inserted ids, epoch 0, the load-time head index) and drops the controller history.

## Process options

### `tide-server`

| Option | Default | Meaning |
|---|---|---|
| `--data-dir PATH` | required | the server's data directory |
| `--rank N` | required | this server's rank, 0..K-1 |
| `--num-servers K` | required | the server count; every owner label must be below K |
| `--port PORT` | 29600 | the ROUTER port |
| `--peers LIST` | none | P2P: all K addresses in rank order, this server's own included |
| `--threads N` | 8 | search worker threads |
| `--pq` / `--full-vectors` | `--pq` | storage mode (exclusive) |
| `--insert-batch-size N`, `--backlink-batch-size N`, `--broadcast-batch-size N` | 16 each | maintenance batch sizes |
| `--edge-overload N` | 16 | a row is pruned when it grows beyond R + N |
| `--io-recv-budget N`, `--io-send-budget N` | 1, 0 (all) | messages received / sent per I/O loop iteration |
| `--print-wire-contract` | | print the metric and wire contract as JSON and exit |
| `--help` | | print the options and exit |

### `tide-coordinator`

| Option | Default | Meaning |
|---|---|---|
| `--data-dir PATH` | required | the coordinator's data directory |
| `--servers LIST` | required | all K server addresses in rank order |
| `--p2p` | off | P2P search (the servers need `--peers`) |
| `--queries N` | 200 | rows of `query.npy` loaded (fewer if the file is shorter) |
| `--L N`, `--K N`, `--W N` | 50, 10, 8 | search list size, result size, and the beam width: the entries a server expands per visit (sent in every search state, so every server uses it) |
| `--batch-size N` | 8 | queries in flight per server thread (an admin command can change it) |
| `--insert-L N`, `--insert-W N`, `--insert-R N`, `--insert-alpha F` | 64, 8, 64, 1.2 | insert search (list size, beam width) and pruning (maximum out-degree, alpha) |
| `--debug` | off | per-query debug CSVs and the recall-miss analysis for every command, in the working directory ([DEBUG.md](DEBUG.md)) |
| `--cluster-check-sec S` | 300 | wait this long at start for every server's stats reply; the start is refused unless every server runs the same storage mode with `base.npy`'s dimension |
| `--print-wire-contract` | | print the metric and wire contract as JSON and exit |
| `--help` | | print the options and exit |

Both binaries refuse an unknown option (they print the options and exit 1). Workloads, inserts, ground
truth files and the controller are given per admin command ([ADMIN_COMMANDS.md](ADMIN_COMMANDS.md)),
not as process options. At start-up the coordinator prints every admin command it accepts.

## Source layout

| Path | Contents |
|---|---|
| `src/tide/tide_server.cpp`, `src/tide/server/` | the server: I/O loop and workers, search expansion (`handlers.cpp`), loading and inserts (`graph_ops.cpp`), stats windows (`stats_window.cpp`), MOVE (`membership_migration*.cpp`) |
| `src/tide/tide_coordinator.cpp`, `src/tide/coordinator/` | the coordinator: admin commands (`admin_commands.cpp`), workloads (`workload_runners.cpp`, `pipeline/`), the controller (`window_*.cpp`), result JSON (`metrics.cpp`) |
| `src/tide/include/` | wire protocol (`protocol/`, `state_protocol.h`, `window_snapshot_wire.h`), server state (`server_types.h`), PQ (`pq.h`), controller and migration headers |
| `src/tide/build_pq.cpp` | `tide-build-pq` |
| `src/common/include/` | distance kernels and `.npy` readers and writers |
| `tests/` | C++ unit tests (`ctest`) and the planner's golden output |
| `tools/prepare/` | data preparation (Python) and its tests |
| `tools/coord_client.py`, `tools/local_cluster.py` | admin client; local Docker cluster |
| `examples/demo.py` | the end-to-end demo |
| `docker/` | images and a Compose file |

The C++ code uses two namespaces: `ann` for the shared headers (the distance kernels and `.npy` readers and
writers in `src/common/include/`, and in `src/tide/include/` the wire messages in `protocol/`, `metrics/`,
`pq.h`, `head_index.h`, `thread_pool.h` and `coordinator_utils.h`) and `tide` for the rest (servers,
coordinator, controller, migrations). The two main source files are `tide_server.cpp` and
`tide_coordinator.cpp`.
