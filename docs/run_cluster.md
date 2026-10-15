# Running a cluster

A cluster is K `tide-server` processes and one `tide-coordinator`. Every server needs `--num-servers K`,
its `--rank` and, for P2P search, `--peers` with all K addresses in rank order, its own included. The
coordinator needs `--servers` with the same list and `--p2p`. For example, with four servers:

```bash
tide-server --data-dir /data --rank 0 --port 28800 --num-servers 4 --threads 4 \
    --insert-batch-size 128 --backlink-batch-size 128 --broadcast-batch-size 128 --pq \
    --peers tcp://server0:28800,tcp://server1:28800,tcp://server2:28800,tcp://server3:28800
# ... the same on server1, server2 and server3 with --rank 1, 2 and 3

tide-coordinator --data-dir /data --p2p --queries 4096 --batch-size 32 --L 50 --K 10 --W 8 \
    --servers tcp://server0:28800,tcp://server1:28800,tcp://server2:28800,tcp://server3:28800
```

The coordinator waits up to `--cluster-check-sec` (default 300) for all servers. It rejects servers with
different storage modes or dimensions, then listens for admin commands on port 29699
([admin_commands.md](admin_commands.md)).

With the layout of [prepare_dataset.md](prepare_dataset.md), every server runs with `--data-dir server` and
the coordinator with `--data-dir runtime`.

`tools/local_cluster.py` starts, queries and stops such a cluster in Docker
(`python3 -m tools.local_cluster up|send|down`).

## tide-server options

| Option | Default | Meaning |
|---|---|---|
| `--data-dir PATH` | required | the server's data directory ([data_format.md](data_format.md)) |
| `--rank N` | required | this server's rank, 0 to K-1 |
| `--num-servers K` | required | number of servers. Every owner rank in `membership.npy` must be below K |
| `--port PORT` | 29600 | ZeroMQ ROUTER port |
| `--peers LIST` | none | P2P: all K addresses in rank order |
| `--threads N` | 8 | search worker threads |
| `--pq`, `--full-vectors` | `--pq` | storage mode ([pq_storage.md](pq_storage.md)) |
| `--insert-batch-size N` | 16 | inserts applied per batch |
| `--backlink-batch-size N` | 16 | backlinks applied per batch |
| `--broadcast-batch-size N` | 16 | broadcasts applied per batch |
| `--edge-overload N` | 16 | prune a row that grows beyond R + N |
| `--io-recv-budget N`, `--io-send-budget N` | 1, 0 (all) | messages received and sent per I/O loop iteration |
| `--help` | | print the options and exit |

## tide-coordinator options

| Option | Default | Meaning |
|---|---|---|
| `--data-dir PATH` | required | the coordinator's data directory |
| `--servers LIST` | required | all K server addresses in rank order |
| `--p2p` | off | P2P search, the servers need `--peers` |
| `--queries N` | 200 | rows of `query.npy` to load, at most its row count |
| `--L N` | 50 | search list size |
| `--K N` | 10 | result size |
| `--W N` | 8 | beam width: entries a server expands per visit, sent with each search |
| `--batch-size N` | 8 | queries in flight per server thread, changeable by admin commands |
| `--insert-L N`, `--insert-W N` | 64, 8 | search list size and beam width of insert searches |
| `--insert-R N`, `--insert-alpha F` | 64, 1.2 | maximum out-degree and alpha of insert pruning |
| `--debug` | off | per-query debug files ([debug.md](debug.md)) |
| `--cluster-check-sec S` | 300 | seconds to wait for all servers at start |
| `--help` | | print the options and exit |

Both binaries reject an unknown option, print the options and exit 1. Workloads, inserts, ground truth files
and the controller are set by admin commands, not process options.
