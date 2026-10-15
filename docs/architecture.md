# Architecture

tide-ann runs one coordinator and K index servers. Each server owns one partition of the graph.

```
        admin client (tools/coord_client.py: one ZeroMQ REQ per command, port 29699)
                                      |
                               tide-coordinator
           head index, query dispatch, inserts, drain, runtime controller
          |                     |                     |                       |
    tide-server 0         tide-server 1         tide-server 2     ...    tide-server K-1
    partition 0           partition 1           partition 2              partition K-1
          \___________ P2P search-state forwarding, backlinks, broadcasts ___________/
```

K is the number of servers. Do not confuse it with the coordinator flags `--K` (result size, default 10)
and `--W` (beam width).

## Processes

- `tide-server` (K of them, 2 to 64): holds one partition, runs search steps, inserts and node moves, and
  counts per-node work.
- `tide-coordinator`: head index, query dispatch, inserts, the controller and the admin socket on port 29699.
- an admin client: `tools/coord_client.py` or any ZeroMQ REQ socket.

Every server binds a ZeroMQ ROUTER socket on `--port` (default 29600), and the coordinator connects a
DEALER socket to each server (`--servers`, in rank order). With P2P search every server also connects to
every peer (`--peers`, in rank order, its own address included) and forwards search states directly. The
coordinator answers admin commands on a REP socket at `tcp://*:29699`, one reply per request. Each
workload (`recall`, `insert`, `mixed`) runs to completion inside its admin command. The process options
are listed in [run_cluster.md](run_cluster.md).

## What each process holds

Server r loads its data directory ([data_format.md](data_format.md)). It holds the owner of every id it
knows (load-time ids from `membership.npy`, inserted ids from the broadcasts) and its partition:
`partition_<r>/adj` with the out-neighbor rows of its nodes as global ids, and
`partition_<r>/local_to_global.npy`, where row i of `adj` is node `local_to_global[i]`. It also holds the
vectors of its storage mode ([pq_storage.md](pq_storage.md)), per-node work counters for the controller,
and a pool of free row slots that later inserts and moves reuse.

The coordinator holds `base.npy` and every inserted vector, `query.npy`, `groundtruth.npy` (the default
ground truth of `recall`), `membership.npy`, the head-index sample `ghost/ghost_indices.npy`, the owner of
every id and the ownership epoch. During a controller phase it also holds the controller's state, history
and journal. A cluster holds at most 100,000,000 ids and 10,000,000 rows per server.

## Search

1. The coordinator picks the head-index node nearest to the query by brute force and sends the search
   state to the server that owns it. The head index is a fixed sample of the load-time ids (1% in the
   prepared data), following the BatANN design. File, tool and flag names call it the ghost sample
   (`ghost/ghost_indices.npy`, `make_ghost_ids.py`, `--ghost`).
2. A server expands the unexpanded beam entries it owns, up to the beam width `--W`. It reads their
   adjacency rows, scores the neighbors ([pq_storage.md](pq_storage.md)) and merges them into the beam,
   whose size is the search list size `L`.
3. When the best unexpanded entries belong to another server, the state (beam, visited set and route)
   moves there. With P2P (`--p2p` on the coordinator, `--peers` on every server) it goes directly from
   server to server. Without P2P it goes back through the coordinator.
4. When no unexpanded entry is left, the server returns the top `--K` results to the coordinator. The
   coordinator may merge inserted vectors that are still pending into the results (`recall --no-merge`
   turns this off) and scores recall@`--K`.

Each server visit counts as one hop, including the entry server. A query that never leaves its entry
server has 1 hop. The controller measures the work of step 2 per node ([controller.md](controller.md)).

## Insert

`insert N --file F` inserts rows of F. The new ids continue after the ids the coordinator knows. Each
insert has four steps, timed in [metrics.md](metrics.md):

1. The coordinator searches for the new vector (`--insert-L`, `--insert-W`).
2. The coordinator sends the vector and candidates to the owner of the nearest result. That server puts the
   new node in a free row slot if it has one and links it with alpha-RNG pruning (`--insert-R`,
   `--insert-alpha`). Candidates it has not received yet are skipped (`insert_neighbors_undelivered`).
3. Every other server that owns a new neighbor adds the back edge. It prunes a row that grows beyond R
   plus `--edge-overload`.
4. Every server learns the new id's owner and stores its vector (`--full-vectors`) or its PQ code (`--pq`).

Servers apply inserts, backlinks and broadcasts in batches (`--*-batch-size`), using several worker
threads while searches run. A partial batch waits for a full one or for a drain, so the graph seen by
searches can lag the inserts.

`drain` waits until every pending insert reply, backlink and broadcast is applied and acknowledged on every
server. `controller-start`, `membership-check`, `migration-*` and `shutdown` drain first.

## Membership, ownership epoch and MOVE

A node's owner is the server that holds its adjacency row and, under `--pq`, its original vector. The
owners change only through a MOVE, a stop-the-world transaction over all servers:

1. Refresh (controller and `migration-apply --refresh`): the coordinator pulls each moved node's current
   row, and under `--pq` its vector, from its current owner.
2. PREPARE: every server receives the whole move list with each moved node's row and row digest, and under
   `--pq` its vector digest. Under `--pq` each moved node's original vector goes only to its target. A
   target installs the incoming rows and vectors in free or new row slots. A source checks every digest
   against its own copy before anything changes.
3. COMMIT: every server publishes the new owners.
4. ACCEPT: the move becomes final, and the sources free the moved rows and vectors into their slot pool.
   ROLLBACK, before ACCEPT, restores the previous state.

The ownership epoch increases with every committed MOVE, and servers reject a plan for another epoch. The
controller issues MOVEs itself, and `migration-apply` issues them by hand
([admin_commands.md](admin_commands.md)). `membership-check` verifies that every server agrees with the
coordinator and, under `--pq`, holds the original vectors of the nodes it owns.

## Phases

```
insert phase:      insert ..., insert ..., drain           (searches allowed)
controller phase:  controller-start  ->  search load  ->  controller-stop
insert phase:      insert ...                              (reuses slots freed by accepted MOVEs)
...
```

`controller-start` drains first. While the phase is active, inserts and the other commands listed in
[admin_commands.md](admin_commands.md#replies) are rejected. See
[controller.md](controller.md#phases-with-inserts).

## Source files

- `src/tide/tide_server.cpp`: server I/O loop and workers
- `src/tide/server/handlers.cpp`, `graph_ops.cpp`: search expansion, loading and inserts
- `src/tide/server/stats_window.cpp`, `membership_migration*.cpp`: work counters and node moves
- `src/tide/tide_coordinator.cpp`: coordinator start-up and daemon loop
- `src/tide/coordinator/admin_commands.cpp`: admin commands
- `src/tide/coordinator/workload_runners.cpp`, `pipeline/`: query and insert pipeline
- `src/tide/coordinator/window_*.cpp`: the controller
- `src/tide/coordinator/metrics.cpp`: result JSON
- `src/tide/include/`: wire protocol, server state, PQ, controller headers
- `src/common/include/`: distance kernels, `.npy` reader and writer
- `src/tide/build_pq.cpp`: `tide-build-pq`

Shared headers use namespace `ann`. Everything else is in `tide`.
