# Debug output

The coordinator writes a per-query CSV log when it is started with `--debug`, or when a workload command
(`recall`, `insert`, `mixed`) carries `--debug`, which then stays on for later commands.

```bash
tide-coordinator --servers tcp://server0:28800,... --data-dir /data/runtime --p2p --debug
python3 tools/coord_client.py --endpoint tcp://127.0.0.1:<port> recall --queries 1000 --debug
```

`<port>` is the coordinator's admin port (29699, or the published port under Docker).

With `--debug` the coordinator also writes `debug_results.npy`, the search results of the command, and
prints a partition summary. Every debug file goes to the coordinator's working directory, which must be
writable, while `--data-dir` may be read-only. A file that cannot be written gives a warning in the reply,
and the command still runs.

## Search result CSV (`debug_*.csv`)

| Column | Meaning |
|---|---|
| `type` | `QUERY` or `INSERT` |
| `id` | workload id |
| `entry_partition` | partition of the start node the head index chose |
| `final_partition` | partition of the first result (`neighbors[0]`) |
| `hops` | server visits ([architecture.md](architecture.md#search)) |
| `nb{i}_id` | id of the i-th result |
| `nb{i}_part` | partition of the i-th result |
| `nb{i}_dist` | distance of the i-th result |
| `topK_p0` .. `topK_p{K-1}` | results on partition i, one column per server |
| `gt{i}_id` | i-th ground-truth id (`QUERY` only) |
| `recall_hits` | results that are in the ground truth (`QUERY` only) |

## Partition summary

After a search or insert command the coordinator prints the entry-partition distribution (from the head
index), the final-partition distribution (from the results) and the entry-to-final transition matrix.

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

Debug logging runs on the coordinator's metric thread, so file I/O does not slow the search thread. The
metric thread runs whether or not `--debug` is set.
