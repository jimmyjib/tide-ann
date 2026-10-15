# Result metrics

`recall`, `insert` and `mixed` answer with a JSON result. Field names are defined in
`src/tide/include/metrics/json_output.h`. The result is printed between two markers:

```
===JSON_RESULT===
{json content}
===JSON_END===
```

## How the latency fields relate

A hop is one server visit, as defined in [architecture.md](architecture.md#search).

Without `--p2p` the coordinator times every hop:

- `hop_rtt` runs from the coordinator's send to the decoded reply.
- `query_latency` runs from the moment the query is issued to its final reply. It also includes the wait
  before the first send and the gaps between hops, which grow when the coordinator or the servers are
  backlogged. So `query_latency.avg_ms` is roughly `hop_rtt.avg_ms` x `search.avg_hops` plus those waits.
- `per_hop_breakdown` splits `hop_rtt` into five terms: `server_ms.total` (which includes `lock_wait_ms`,
  so do not add it again), `worker_queue_ms`, `response_queue_ms`, `coord_total_ms` and `network_ms`.
  `network_ms` is the remainder, so the five terms add up to `avg_latency_ms`.
- The send times (`server_timing_us.zmq_send`, `coord_timing_per_hop_us.send`) overlap work timed on the
  other side, so they are not subtracted. They stay in `network_ms`.
- `per_query_breakdown` holds the same terms summed per search, which is the per-hop value x `avg_hops`.
  It does not include the gaps between hops.

With `--p2p` the servers forward states to each other and reply to the coordinator once per query, so the
coordinator times the route as one hop:

- `hop_rtt` runs from the coordinator's send to the final reply.
- `server_ms`, `worker_queue_ms` and `response_queue_ms` are the last server's. `network_ms` is the rest of
  `hop_rtt`, so it includes the earlier servers' work.
- `search.avg_hops` and `hop_distribution` count every server visit, and `search.avg_dist_comp` counts the
  distance computations of the whole route.
- The coordinator-side terms are unavailable: `coord_total_ms_available` and
  `coord_timing_per_hop_us.available` are `false`. Use `coord_timing_totals` for coordinator load.

### Inserts

`unified_timing` gives the wall time of the four phases of an insert:

| Field | Phase |
|---|---|
| `search_phase_ms` | the insert's search, timed like a query |
| `insert_rpc_ms` | the INSERT request to the new node's owner: allocation, pruning and edges |
| `backlink_ms` | BACKLINK to each server that owns a new neighbor, until all acknowledge |
| `broadcast_ms` | BROADCAST of the new id's owner and vector to every server, until all acknowledge |

`insert.query_latency.avg_ms` is the search plus the INSERT request. Backlinks and broadcasts run in
parallel across servers, so their summed lock wait can exceed their wall time. `insert.per_query_breakdown`
covers only the server's work on the INSERT request. It leaves out the coordinator and transport time, so
its sum is below `insert_rpc_ms`.

## Search metrics

| Section | Fields |
|---|---|
| top level | `timestamp`, `dataset`, `mode`, `parameters.*` |
| `search` | `count`, `recall`, `qps`, `avg_hops`, `avg_dist_comp`, `total_time_ms`, and `result_digest` with `--digest` |
| `serving` | `elapsed_ms`, `qps`, `workload_setup_ms`, `head_index_ms` (`recall` only) |
| `wall_clock` | `search_ms`, `insert_ms`, `search_only_qps`, `insert_only_tps`, `*_with_metrics_*` |
| `merge_search` | `calls`, `pending_total`, `pending_eligible`, `pending_topk`, `result_changed`, `wall_ms` |
| `observability` | `diagnostic_mode`, set by `insert --search-only`, `--no-backlink`, `--no-broadcast` |
| `hop_distribution` | `{"hops": queries, ...}` |
| `query_latency` | `avg_ms`, `min/p50/p95/p99/max_ms`, issue to final response, queueing included |
| `latency` | `end_to_end.*` (queueing included), `service.*` (per hop) |
| `per_query_breakdown` | `server_ms.total`, `server_ms.lock_wait_ms`, `worker_queue_ms`, `response_queue_ms`, `network_ms`, `coord_total_ms`, `coord_total_ms_available` |
| `per_hop_breakdown` | the same terms per hop, `avg_latency_ms` |
| `hop_rtt` | `avg_ms`, `min/p50/p95/p99/max_ms` |
| `route_traces` | `traced_queries`, `routes`, and `path_transport_residual`: path RTT minus the measured server and coordinator time (`avg_per_query_us`, `avg_per_hop_us`, min/p50/p95/p99/max, `count`). `negative_count` should be 0. |
| `server_timing_us` | `zmq_recv`, `zmq_send`, `deserialize`, `serialize`, `queue_depth`, `resp_queue_wait`, `io_processing`, `worker_overhead`, `send_prep` |
| `coord_timing_per_hop_us` | `available`, `serialize`, `send` (diagnostic), `response_queue`, `recv`, `deserialize` |
| `coord_timing_totals` | `serialize_us`, `send_us`, `response_queue_us`, `recv_us`, `deserialize_us`, `process_us` (main-thread response handling), `poll_wait_us` (main-thread idle wait), `send_count`, `poll_count` (main-loop iterations), `recv_count`, `total_hops` |
| `bottleneck` | `overhead_per_hop_us`, `overhead_ratio` |
| `expansion` | `local_ratio`, `local_expanded`, `top_W`, `avg_neighbors_evaluated` |
| `distance` | `avg_pq_dist`, `avg_full_dist`, null under P2P |
| `per_hop_distance` | `per_hop_avg`, `hop0_query_count`, `avg_hop0_dist`, `multihop_*`, null under P2P |
| `server_load` | requests per server |
| `lock_metrics` | lock counts and waits, `phase_specific.*`, `per_server`, `per_server_phase`, `resize_mutex_profile.*`. `resize_mutex_profile` counts every lock acquisition per role, with `wait_us` the time to acquire and `hold_us` the time held. |
| `backlink`, `broadcast` | `count`, `avg_per_insert`, `wall_clock_ms`, `total_lock_wait_ms` |
| `network_baseline` | `ping_count`, `servers[i].*`, `overall.*`, `direct.*` |
| `self_query` | per query: own node found, its rank and distance |

`search_only_qps` measures the search path alone, and the `*_with_metrics_*` fields and
`merge_search.wall_ms` include the metric thread's merge of pending inserts. `latency` covers searches,
or insert searches in an insert-only command, where its percentiles add the average INSERT RPC time.
`bottleneck.overhead_per_hop_us` sums the coordinator's serialize, send, response-queue, receive and
deserialize time.

`serving.elapsed_ms` runs from just after the ground truth is read to the last completed search. It
includes reading `--query-file`, preparing the queries and the head-index search, which
`workload_setup_ms` and `head_index_ms` time. `serving.qps` is the queries divided by that time, so it is
at most `wall_clock.search_only_qps`, which starts at the first dispatch. The four values are null when
one of the times is missing or not every query completed.

With `--digest`, `search.result_digest.value` is a 64-bit FNV-1a hash, written as 16 hex digits, of a
format tag, K, the number of completed searches and each search's query index and top-K ids in
query-index order. A missing rank counts as 0xffffffff. Two runs with the same value returned the same
results. `query_count` and `ranked_neighbors_hashed` count the searches and ids it covers, and
`algorithm` names the format.

### Distance counts

`search.avg_dist_comp` is the distance computations per query over every server visit. `distance` splits
it into ADC (`avg_pq_dist`) and exact evaluations with re-ranks (`avg_full_dist`), and `per_hop_distance`
by visit. Under P2P only the route's total reaches the coordinator, so those two sections are null.

### Distance kinds and W(v)

Server counters in the `stats` reply (`servers[].search.*`), cumulative since start or `reset`:

| Counter | Meaning |
|---|---|
| `distance_count` | `exact_count` + `adc_count` + `rerank_count` |
| `exact_count` | neighbors scored against a vector this server holds |
| `adc_count` | neighbors scored against their PQ code (PQ mode) |
| `rerank_count` | expanded entries re-scored exactly on their owner (PQ mode) |
| `missing_code_count` | neighbors with neither vector nor code, scored FLT_MAX, not counted as work |
| `adc_table_count`, `adc_table_us` | ADC query tables built, at most one per request, and their time |
| `distance_us` | time of the distance work above, ADC table builds excluded |

W(v) counts the same evaluations per expanded node. Ids whose code this server has not finished storing
are scored FLT_MAX and not counted. A backlink can link such an id while another message for it is still
writing the code, and after a `drain` nothing is missing. With `--full-vectors` there is no re-rank, ADC
or table.

## Insert metrics

| Section | Fields |
|---|---|
| `insert` | `count`, `throughput`, `avg_backlinks`, `total_backlinks`, `total_broadcasts` |
| `insert.query_latency` | `avg_ms` = `search_ms` + `insert_rpc_ms` |
| `insert.per_query_breakdown` | `queue_wait_ms`, `server_ms.*`, `deserialize_ms`, `prune_ms`, `memory_ms`, `local_backlink_prune_ms` |
| `insert.search` | `count`, `total_hops`, `avg_hops`, `hop_distribution`, and the search section's latency fields |
| `maintenance.backlink` | `pre_drain_edges`, `drain_exact_edges`, `combined_edges`, `combined_avg_per_insert`, `metric_complete` |
| `maintenance.broadcast` | `pre_drain_acks`, `drain_acks`, `combined_acks` |
| `unified_timing` | `search_phase_ms`, `insert_rpc_ms`, `backlink_ms`, `broadcast_ms` |
| `resize` | `total_count`, `total_ms`, `by_vector.*`, `by_server.*` |
| `beam_tracking` | `beam_*` and `result_*` entries and their inserted share |
| `server_stats` | `per_server_nodes`, `total_nodes` |
| `partition_analysis` | `entry_partition`, `final_partition`, `transitions` |

`insert.throughput` does not include the time until every server has applied the insert. Backlinks and
broadcasts may still be pending when `insert` returns, so `total_backlinks` and `total_broadcasts` count
only the part done during the command. After a `drain`, compare volumes with `maintenance.*.combined_*`.
Insert `server_ms` leaves out the resize time, which the `resize` section reports.
