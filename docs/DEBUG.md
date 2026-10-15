# Debug output

## `--debug`

The coordinator writes per-query CSV logs when it is started with `--debug`, or when a workload command
(`recall` / `baseline`, `insert`, `mixed`) carries `--debug` (it then stays on for later commands):

```bash
tide-coordinator --servers tcp://server0:28800,... --data-dir /data/runtime --p2p --debug
python3 tools/coord_client.py --endpoint tcp://127.0.0.1:<port> recall --queries 1000 --debug
```

`<port>` is the published admin port: under Docker Compose, `docker compose -f docker/compose.yml port
coordinator 29699`; for a cluster started with `python3 -m tools.local_cluster up`, the port `up`
prints (`python3 -m tools.local_cluster send --port <port> -- recall --queries 1000 --debug` sends the
same command). The client's default endpoint, `tcp://localhost:29699`, answers only when the coordinator
runs on the same host without Docker port mapping.

It also writes `debug_results.npy` (the search results of the command) and prints the partition
summary below. Every debug file (`debug_*.csv`, `debug_*_hops.csv`, `debug_results.npy`) goes to the
coordinator's working directory (`/work` in the demo and Compose layouts), which must be writable;
`--data-dir` may be read-only. A file that cannot be written is a `[WARN]` line in the reply, and the
command still runs.

### Search result CSV (`debug_*.csv`)

| Column | Meaning |
|---|---|
| `type` | `QUERY` or `INSERT` |
| `id` | workload id |
| `entry_partition` | partition of the start node the head index chose |
| `final_partition` | partition of the first result (`neighbors[0]`) |
| `hops` | servers that processed the search state, the entry server included (1 = the entry server only) |
| `nb{i}_id` | id of the i-th result |
| `nb{i}_part` | partition of the i-th result |
| `nb{i}_dist` | distance of the i-th result |
| `topK_p0` .. `topK_p{K-1}` | results that belong to partition i (one column per server, K = the server count) |
| `gt{i}_id` | i-th ground-truth id (`QUERY` only) |
| `recall_hits` | results that are in the ground truth (`QUERY` only) |

### Hop state CSV (`debug_*_hops.csv`)

The file is created with the header below, but the coordinator's response path writes no hop rows, so it
holds the header only (under P2P the coordinator sees only each query's final reply).

| Column | Meaning |
|---|---|
| `type` | `QUERY` or `INSERT` |
| `id` | workload id |
| `hop` | hop number, from 0 |
| `cur_part` | partition that ran this hop |
| `next_part` | partition of the next hop |
| `beam_size` | beam size |
| `beam_p0` .. `beam_p{K-1}` | beam entries per partition (one column per server) |
| `top1_id`, `top1_part`, `top1_dist` | the best beam entry |

### Partition summary (stdout)

After a search or insert command the coordinator prints the entry-partition distribution (from the
head index), the final-partition distribution (from the results) and the entry-to-final transition
matrix:

```
=== INSERT PARTITION DEBUG ===
Entry partition distribution (from head index):
  P0: 27091 (27.1%)
  ...
Final partition distribution (from search result):
  P0: 61912 (61.9%)
  ...
Transition matrix (Entry -> Final):
        Final0  Final1  Final2  Final3
Entry0:  92.7%   4.9%   2.0%   0.4%  (n=27091)
...
=== END PARTITION DEBUG ===
```

## Reading the CSV files

```python
import pandas as pd

df = pd.read_csv("debug_run.csv")

# Final partition distribution and the entry -> final transition matrix.
print(df["final_partition"].value_counts(normalize=True).sort_index())
print(pd.crosstab(df["entry_partition"], df["final_partition"], normalize="index"))

# Share of the top-K results on each partition (one topK_p<i> column per server).
top = [c for c in df.columns if c.startswith("topK_p")]
print(df[top].sum() / df[top].sum().sum())
```

## Where the logging runs

Debug logging runs on the coordinator's metric thread, not on the search thread: the search thread
pushes a metric event for each completed search (query or insert search) and continues; the metric thread writes the CSV rows, merges
results and tracks beams. The owner partition of every id a log line names travels in the event: the
search thread resolves it from the membership, which only it reads and grows. File I/O therefore never blocks the search thread or enters its timing
fields. The metric thread runs whether or not `--debug` is set.

| File | Role |
|---|---|
| `src/tide/coordinator/pipeline/metric_thread.cpp` | metric thread loop and event handling |
| `src/tide/coordinator/pipeline/debug_utils.cpp` | `log_search_result`, `log_hop_state`, the partition summary |
| `src/tide/coordinator/pipeline/pipeline_core.cpp` | search thread; creates and enqueues the metric events |
| `src/tide/include/pipeline_types.h` | debug declarations, `DebugState` |
| `src/tide/include/types/workload_types.h` | `MetricEvent`, `MetricEventType` |
