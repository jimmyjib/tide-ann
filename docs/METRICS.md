# Metrics of the coordinator's JSON results

Every workload command (`recall`, `insert`, `mixed`, `drain`, `stats`, ...) answers with a JSON result. This
document explains the timing model behind the search and insert metrics and lists where each field is
measured. The emitting code, `src/tide/include/metrics/json_output.h`, is authoritative for field names;
the example values below are illustrative.

## 1. Output

A result is printed between two markers (the admin client prints the whole reply):

```
===JSON_RESULT===
{json content}
===JSON_END===
```

Notes on the current output:

- `observability.diagnostic_mode` marks diagnostic workload variants such as `insert --search-only`, `--no-backlink`
  or `--no-broadcast`; such a run does not measure the full insert path.
- Insert search metrics are emitted under `insert.search.*`.
- `merge_search.*` audits the merge of pending inserts into search results. The merge changes results before
  recall is computed, so it is part of the search, not a diagnostic. `search.qps` and
  `wall_clock.search_only_qps` measure the foreground search path; `wall_clock.total_with_metrics_ms`,
  `wall_clock.search_with_metrics_qps` and `merge_search.wall_ms` include the metric thread's merge work.
  `calls`, `pending_total`, `pending_eligible`, `pending_topk` and `result_changed` show whether pending
  inserts were considered and whether they changed a top-K. `recall --no-merge` turns the merge off.
- Insert maintenance can still be pending when an insert command returns. This is legal: the deferred
  backlinks, broadcasts and insert responses settle later or at an explicit `drain`. The `status` command's
  reply shows the current queue in `maintenance.pending` (`insert_responses`, `backlink_inserts`,
  `broadcast_inserts`); the `drain` reply shows what was pending when the drain began in `pending_at_start`
  (`insert_responses`, `backlink_batches`, `broadcasts`).
- `coord_timing_totals.poll_wait_us`, `poll_count` and `process_us` are zero: the multithreaded coordinator
  has no single poll loop. `coord_timing_totals.send_us` / `recv_us` are summed from the per-hop
  coordinator timing.
- `coord_timing_totals.response_queue_us` is the coordinator's local hand-off: a server thread has received
  the response and the main thread has not processed it yet. It is subtracted when the network time is
  computed (section 2).
- `query_latency.*` is the end-to-end workload latency from issue to the final response. With the
  all-at-once scheduler it includes queue backlog on the coordinator and the servers; use `hop_rtt.*` and
  `per_hop_breakdown.*` for per-hop service and network analysis.
- `latency.*` is the namespace for new analysis: `latency.end_to_end` keeps the backlog-inclusive workload
  latency, `latency.service` gives the per-hop service terms. Its scope is `search_workloads` in search and
  mixed modes and `insert_workloads` in insert-only mode, so that it never reports a zero search latency.
  Insert-only percentiles are the insert-search percentiles plus the average INSERT RPC time
  (`latency.end_to_end.percentiles_source` says so). `query_latency.*`, `per_query_breakdown.*` and
  `per_hop_breakdown.*` are kept as aliases.
- `route_traces.path_transport_residual` is the coordinator's single-clock path RTT minus the measured
  route-local server processing, route queueing and coordinator overhead; its `count` is the number of
  samples. Negative residuals are not clamped: a nonzero `negative_count` is an accounting failure to
  investigate. (Arrival and departure times recorded by different server processes are not comparable, so
  no per-hop network time is derived from them.)
- Insert maintenance is split into the part done during the insert workload and the part done by the drain.
  `insert.total_backlinks` and `insert.avg_backlinks` count the first part only; compare backlink volumes
  with `maintenance.backlink.combined_edges` and `maintenance.backlink.combined_avg_per_insert` after an
  explicit drain.

## 2. Time measurement

### Core relation

```
query_latency.avg_ms ~= hop_rtt.avg_ms x avg_hops
```

| Variable | Measured as | Code | JSON field |
|---|---|---|---|
| `hop.total_us` | `recv_end - item.send_time` | `pipeline_core.cpp` | `hop_rtt.avg_ms` |
| `query_latency_us` | `recv_end - item.start_time` | `pipeline_core.cpp` | `query_latency.avg_ms` |

- `query_latency ~= sum(hop.total_us) + gaps between hops`
- `per_hop_breakdown.avg_latency_ms` = `hop_rtt.avg_ms`

### Network time of a hop

```cpp
// src/tide/include/types/timing_types.h
pure_network_us() = total_us
                  - queue_wait_us
                  - measured_server_total_us()
                  - measured_coord_total_us()

measured_server_total_us() =
    server_processing_us + zmq_recv_us + serialize_us
    + response_queue_wait_us + io_processing_us
    + worker_overhead_us + send_prep_us

measured_coord_total_us() =
    coord_serialize_us + coord_response_queue_us + coord_recv_us + coord_deserialize_us
```

Rule: a time interval measured on one side that physically overlaps work already counted on the other side
is not subtracted. Three timings are therefore reported for diagnosis but left out of the subtraction:

| Field | Why it is not subtracted | Effect |
|---|---|---|
| `zmq_send_us` (server) | it can be measured only after the send, so the header carries the previous reply's value | a small positive bias (a few microseconds) |
| `coord_send_us` (coordinator) | the send window includes the ZeroMQ I/O thread hand-off, the kernel send and the server's start of work, which the server already counts | a small positive bias |
| poll-to-receive wait (coordinator) | the coordinator's idle poll overlaps the server's processing of the hop; the field is not recorded | none |

Subtracting them would count the same physical time twice and make `pure_network_us` negative.

### Timeline of one query

```
                          query_latency_us
    |<-------------------------------------------------------->|
    start_time         hop1 RTT        gap    hop2 RTT       recv_end
    |                 |<----->|              |<----->|          |
    |----first hop----|       |---gap---|    |--hop--|----------|
                      ^                      ^
                hop.total_us            hop.total_us
```

Example: `hop_rtt.avg_ms` 7.12, `avg_hops` 1.94: 7.12 x 1.94 = 13.81 ms against a measured
`query_latency.avg_ms` of 13.86 ms.

## 3. Data flow

### One search hop

```
COORDINATOR
  [1] send_workload()
      t0 = now(); serialize_state(); t1 = now()      coord_serialize_us = t1 - t0
      t2 = now(); zmq::send(); t3 = now()            coord_send_us = t3 - t2   (diagnostic only)
      item.send_time = t3                            <- RTT starts
                         | network
                         v
SERVER I/O thread
  [2] recv_start = now(); zmq::recv(); recv_end = now()      zmq_recv_us = recv_end - recv_start
  [3] parse and route; work.enqueue_time = now()             <- queue wait starts
      work_queue.push(work)                                   io_processing_us = enqueue - recv_end
                         | work queue (queue_wait_us)
                         v
SERVER worker thread
  [4] pickup_time = now()                                     queue_wait_us = pickup_time - work.enqueue_time
  [5] deserialize_state()                                     deser_us
  [6] lock: shared_lock(g_resize_mutex) + per-node lock (adjacency snapshot)
                                                              lock_wait_us = t_lock_acquired - t_lock_start
  [7] process_state() (greedy search)                         server_processing_us = t_search_end - t_search_start
  [8] serialize the response; resp.enqueue_time = now(); response_queue.push(resp)
      total_processing_us = deser_us + server_processing_us   (the source of server_ms)
                         | response queue
                         v
SERVER I/O thread
  [9] pop_time = now()                                        response_queue_wait_us = pop_time - resp.enqueue_time
  [10] send_start = now(); zmq::send(); send_end = now()      send_prep_us = send_start - pop_time
                                                              zmq_send_us = send_end - send_start (diagnostic only)
       The server writes its timing fields into the reply header right before the send; zmq_send_us of
       the current reply is known only afterwards, so the header carries the previous reply's value.
                         | network
                         v
COORDINATOR
  [11] process_response(): recv_start = now(); zmq::recv(); recv_end = now()   <- RTT ends
       coord_recv_us = recv_end - recv_start
       total_rtt = recv_end - item.send_time                  (hop_rtt)
       deserialize_response()                                 coord_deserialize_us
```

### Per query and per hop

```
per_query_breakdown (divided by the number of searches):
  server_ms.total        = sum(measured_server_total - response_queue) / queries
  server_ms.lock_wait_ms = sum(lock_wait_us) / queries          (included in server_ms.total)
  worker_queue_ms        = sum(queue_wait_us) / queries
  response_queue_ms      = sum(response_queue_wait_us) / queries
  network_ms             = sum(pure_network_us) / queries
  coord_total_ms         = sum(measured_coord_total_us) / queries

per_hop_breakdown (divided by the total search hops):
  avg_latency_ms = sum(hop_rtt) / total_hops, and the same terms per hop

  avg_hops = total_search_hops / searches  (hops = servers that processed a query's state, the entry
                                           server included: 1 = finished on the entry server)
  per_query ~= per_hop x avg_hops
  server + worker_queue + response_queue + network + coord_total ~= avg_latency
```

The terms leave out only the diagnostic-only fields (`coord_send_us`, `zmq_send_us`), so their sum falls short
of `avg_latency_ms` by a few hundredths of a millisecond.

**P2P.** With `--p2p` the servers forward states to each other and the coordinator sees only the final reply
of a query. The coordinator-side per-hop terms cannot be computed then: `coord_total_ms_available` and
`coord_timing_per_hop_us.available` are `false`; use `coord_timing_totals` for coordinator load.

### One insert (four phases)

```
PHASE 1  SEARCH_PENDING
  send_workload() for the search; item.phase_start_time = now()
  search hops as for a query; every reply is a hop sample
  the final (DONE) reply: item.search_us = now() - item.phase_start_time     -> insert search latency

PHASE 2  INSERT_PENDING (the INSERT RPC to the owner)
  item.phase_start_time = now()
  server: work-queue wait; allocation (shared_lock, exclusive only to resize); prune (shared_lock,
          read-only neighbour selection); edges (shared_lock + per-node locks in sorted order); memory
          allocation; prune of same-partition backlinks
  INSERT_RESPONSE: item.insert_us = now() - item.phase_start_time            -> insert_rpc
  its cross-partition edges become pending backlinks

PHASE 3  BACKLINK_PENDING (in parallel)
  BACKLINK to every server that owns a new neighbour (a few per insert)
  each server: per-node lock per backlink
  all BACKLINK_ACKs: item.backlink_us = now() - item.phase_start_time

PHASE 4  BROADCAST_PENDING (in parallel)
  BROADCAST to every server (the new id's owner and vector)
  each server: shared_lock, exclusive to resize
  all BROADCAST_ACKs: item.broadcast_us = now() - item.phase_start_time; item.done = true
```

Phases 3 and 4 run in parallel across servers: their wall time is the maximum RTT, while their summed lock
wait is a sum over the parallel requests, so the lock wait can exceed the wall time. Servers apply inserts,
backlinks and broadcasts in batches (`--insert-batch-size`, `--backlink-batch-size`,
`--broadcast-batch-size`); a partial batch waits for a full one or for `drain`.

`unified_timing` reports the four phases' wall time (`search_phase_ms`, `insert_rpc_ms`, `backlink_ms`,
`broadcast_ms`); `insert.query_latency.avg_ms` is the search plus the INSERT RPC per insert.

The INSERT RPC breakdown (`insert.per_query_breakdown`: queue wait, deserialize, lock wait, prune, memory,
local backlink prune) does not cover the coordinator's serialize / send / receive / deserialize time or the
server's ZeroMQ and response-queue time, so its sum is below `insert_rpc_ms`.

## 4. Search metrics

| Section | Fields | Measured in |
|---|---|---|
| top level | `timestamp`, `dataset`, `mode`, `parameters.*` | `json_output.h`, command arguments (`workload_runners.cpp`) |
| `search` | `count`, `recall`, `qps`, `avg_hops` (server visits per search, the entry server included), `avg_dist_comp`, `total_time_ms`, `result_digest` | `compute_recall()` (`workload_types.h`), `metrics.cpp`, `pipeline_core.cpp` |
| `wall_clock` | `search_ms`, `insert_ms`, `search_only_qps`, `insert_only_tps`, `*_with_metrics_*` | `pipeline_core.cpp`, `workload_runners.cpp` |
| `hop_distribution` | `{"hops": queries, ...}` | `metrics.cpp` |
| `query_latency` | `avg_ms`, `min/p50/p99/max_ms` | `pipeline_core.cpp` -> `metrics.cpp` (sorted samples) |
| `latency` | `end_to_end.*`; `service.hop_rtt_avg_ms`, `server_per_hop_ms`, `worker_queue_per_hop_ms`, `response_queue_per_hop_ms`, `transport_residual_per_hop_ms` | `metrics.cpp` |
| `per_query_breakdown` | `server_ms.total`, `server_ms.lock_wait_ms`, `worker_queue_ms`, `response_queue_ms`, `network_ms`, `coord_total_ms`, `coord_total_ms_available` | server: `tide_server.cpp`, `timing_types.h`; coordinator: `pipeline_core.cpp`; totals: `metrics.cpp` |
| `per_hop_breakdown` | the same terms per hop, `avg_latency_ms` (= `hop_rtt.avg_ms`) | `metrics.cpp` |
| `hop_rtt` | `avg_ms`, `min/p50/p99/max_ms` | `pipeline_core.cpp` -> `metrics.cpp` |
| `server_timing_us` | `zmq_recv`, `zmq_send`, `deserialize`, `serialize`, `queue_depth`, `resp_queue_wait`, `io_processing`, `worker_overhead`, `send_prep` | server I/O and worker threads (`tide_server.cpp`), collected in `pipeline_core.cpp` |
| `coord_timing_per_hop_us` | `available`, `serialize`, `send` (diagnostic), `response_queue`, `recv`, `deserialize` | `workload_generator.cpp`, `pipeline_core.cpp` |
| `coord_timing_totals` | `serialize_us`, `send_us`, `response_queue_us`, `recv_us`, `deserialize_us`, counts, `total_hops` | coordinator totals |
| `bottleneck` | `overhead_per_hop_us`, `overhead_ratio` | `metrics.cpp` |
| `expansion` | `local_ratio`, `local_expanded`, `top_W`, `avg_neighbors_evaluated` | server process metrics, `metrics.cpp` |
| `distance` | `avg_pq_dist`, `avg_full_dist` | `metrics.cpp` |
| `per_hop_distance` | `per_hop_avg`, `hop0_query_count`, `avg_hop0_dist`, `multihop_query_count`, `multihop_total_hops`, `avg_multihop_dist_per_hop` (under P2P the final visit only, [Distance counts](#distance-counts)) | `metrics.cpp` |
| `server_load` | requests per server | coordinator totals |
| `lock_metrics` | lock counts and waits, `phase_specific.*`, `per_server`, `per_server_phase` (with `graph`: `bl_requested`, `bl_applied`, `bl_skip_not_local`, `bl_skip_dup`), `resize_mutex_profile.per_server_search_sub_stage` | server lock metrics, `metrics.cpp` |
| `backlink`, `broadcast` | `count`, `avg_per_insert`, `wall_clock_ms`, `total_lock_wait_ms` | insert pipeline |
| `network_baseline` | `ping_count`, `servers[i].*`, `overall.*`, `direct.*` | `recall --ping-baseline` |
| `self_query` | per query: its own node found, its rank and distance (`recall --self-ids`) | `workload_runners.cpp` |

`bottleneck.overhead_per_hop_us` = `coord_serialize_us + coord_send_us + coord_recv_us + coord_deserialize_us`:
it shows the coordinator's whole per-hop overhead, so it includes `coord_send_us`, which the network-time
subtraction leaves out.

### Distance counts

Under P2P the coordinator receives only the reply of each query's final server visit (the servers forward
intermediate search states to each other directly). `distance.avg_full_dist` (exact and re-rank
evaluations), `distance.avg_pq_dist` (ADC evaluations) and their sum `search.avg_dist_comp` therefore count
that final visit only, divided by every completed query. A query that finished on its entry server is counted
in full; a query that visited several servers loses the work of its earlier visits. `per_hop_distance` has the
same limitation: every query lands in `hop0_query_count` / `avg_hop0_dist` with its final visit's count, and
the `multihop_*` fields stay 0 even when queries crossed partitions (`hop_distribution` and `search.avg_hops`
give the visit counts). For two queries, A with 100 distances on one server, and B with 120 on its entry
server and 80 on a second server:

```text
servers' stats sum / query = (100 + 120 + 80) / 2 = 150
avg_dist_comp              = (100 + 80) / 2       = 90
```

For the total distributed search work, take the delta of every server's `search.distance_count`
(= `exact_count` + `adc_count` + `rerank_count`) between two `stats` replies and divide by the queries in
between.

### Distance kinds, W(v) and the controller's prices under hybrid PQ storage

Server counters (`stats` reply, `servers[].search.*`, cumulative since start or `reset`; a window's
statistics acknowledgement carries the same values):

| Counter | Meaning |
|---|---|
| `distance_count` | distance evaluations = `exact_count` + `adc_count` + `rerank_count` |
| `exact_count` | neighbours scored against a vector this server holds |
| `adc_count` | neighbours scored against their PQ code with the query's ADC table (own-only storage) |
| `rerank_count` | expanded entries re-scored exactly on their owner (own-only storage); one evaluation each |
| `missing_code_count` | neighbours with neither vector nor code, scored FLT_MAX: **not** distance work |
| `adc_table_count`, `adc_table_us` | ADC query tables built (at most one per request) and their build time |
| `distance_us` | time of the distance work above (re-ranks included); the ADC table build is not part of it |

The window counter W(v) counts the same evaluations per expanded node v: its scored neighbours plus 1 when the
expansion re-ranked v. The controller's per-server prices are c_d = T_d / D (distance time per distance
evaluation) and c_r = (T_tot - T_d) / R (the rest of the search time per request), so the ADC table build, a
fixed cost per request, lands in c_r; the acknowledgement's table count allows pricing it as a separate term.
Under full replication there is no re-rank, no ADC and no table. In both modes a neighbour id beyond the
vectors a server has stored (an inserted id whose broadcast has not reached it yet) is scored FLT_MAX, counted
as missing and left out of `distance_count` and W(v); searches that run while inserts are being applied can
therefore count a few fewer distances. After a `drain` nothing is missing.

## 5. Insert metrics

| Section | Fields | Measured in |
|---|---|---|
| `insert` | `count`, `throughput`, `avg_backlinks` / `total_backlinks` / `total_broadcasts` (during the insert workload only) | `metrics.cpp` |
| `insert.query_latency` | `avg_ms` = `search_ms` + `insert_rpc_ms` (backlinks and broadcasts excluded), `semantics` | `metrics.cpp` |
| `insert.per_query_breakdown` | the INSERT RPC on its server (no hops): `queue_wait_ms`, `server_ms.total` and `server_ms.lock_wait_ms` (allocation and edge locks), `deserialize_ms`, `prune_ms`, `memory_ms`, `local_backlink_prune_ms` | `workload_runners.cpp` from the INSERT_RESPONSE header |
| `insert.search` | `count`, `total_hops`, `avg_hops`, `hop_distribution`, `query_latency.*`, `per_query_breakdown.*`, `per_hop_breakdown.*`, `hop_rtt.*` (the search section's format) | `metrics.cpp` |
| `maintenance.backlink` | `pre_drain_edges`, `drain_exact_edges`, `combined_edges`, `combined_avg_per_insert`, `metric_complete` | insert workload and `drain` |
| `maintenance.broadcast` | `pre_drain_acks`, `drain_acks`, `combined_acks` | insert workload and `drain` |
| `unified_timing` | `search_phase_ms`, `insert_rpc_ms`, `backlink_ms`, `broadcast_ms` | insert pipeline |
| `resize` | `total_count`, `total_ms`, `by_vector.*`, `by_server.*` | server handlers |
| `beam_tracking` | `beam_total_entries`, `beam_inserted_entries`, `beam_inserted_ratio`, `result_total_entries`, `result_inserted_entries`, `result_inserted_ratio` | `metric_thread.cpp` |
| `server_stats` | `per_server_nodes`, `total_nodes` | GET_STATS round |
| `partition_analysis` | `entry_partition`, `final_partition`, `transitions` | `debug_utils.cpp` |

Insert `server_ms` excludes the resize time (`processing_us - resize_total_us`); the `resize` section reports it.
After a `drain`, compare backlink and broadcast volumes with `maintenance.*.combined_*`, not with the insert
result alone: without a drain the tail of the maintenance is still queued.

## Appendix A: where the measurements are taken

| File | Measures |
|---|---|
| `coordinator/pipeline/pipeline_core.cpp` | end-to-end latency, hop RTT, coordinator I/O timing |
| `coordinator/pipeline/metric_thread.cpp` | debug logging, result merge, beam tracking |
| `coordinator/pipeline/workload_generator.cpp` | workload creation, serialize timing |
| `coordinator/pipeline/debug_utils.cpp` | partition debug output, debug log files |
| `coordinator/workload_runners.cpp` | workload execution, wall-clock timing, phase tracking |
| `coordinator/metrics.cpp` | aggregation and JSON output |
| `server/handlers.cpp` | request handling, queue wait, ZeroMQ I/O, lock wait, INSERT timing, resize statistics |
| `server/graph_ops.cpp` | distance computations, local expansion, RobustPrune |
| `include/protocol/search_protocol.h` | search reply metrics (`StateResponseHeader`) |
| `include/protocol/insert_protocol.h` | INSERT RPC reply timing (`InsertResponseHeader`) |
| `include/protocol/message_types.h` | server statistics reply (`StatsResponseHeader`) |
| `include/types/search_context.h` | per-query state (latency, hop count) |
| `include/types/pending_insert.h` | insert state |
| `include/types/timing_types.h` | detailed timing, `pure_network_us()` |
| `include/metrics/experiment_result.h` | the structures behind the JSON output |

## Appendix B: how the per-hop terms add up

```
hop_rtt (avg_latency_ms)
  = network (request, coordinator -> server)
  + worker_queue_ms            I/O thread enqueue -> worker pickup
  + server_ms.total            lock_wait_ms (included here; do not add it again)
                               + search processing
                               + ZeroMQ, serialize, I/O processing, ...
  + response_queue_ms          worker -> I/O thread
  + network (reply, server -> coordinator)
  + coord_total_ms             serialize + response-queue hand-off + recv + deserialize

network_ms = total - worker_queue - measured_server_total - measured_coord_total
```

Locks on a server:

- `g_resize_mutex` (shared mutex): guards resizing the outer arrays; shared normally, exclusive only to resize;
- `node_locks[local_idx]` (per-node mutex): guards `adj[local_idx]`; taken only to read or change that node;
- `g_alloc_mutex`: serialises the `n_base` / `n_local` counters.

| Operation | Locks |
|---|---|
| search | shared `g_resize_mutex` + per-node lock (adjacency snapshot) |
| insert, allocation | shared, or exclusive when a resize is needed |
| insert, prune | shared (read-only) |
| insert, edges | shared + per-node locks in ascending `local_idx` order |
| backlink | shared (exclusive to resize) + per-node lock |
| broadcast | shared (exclusive to resize) |
| reset | exclusive |
