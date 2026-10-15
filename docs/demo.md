# Demo

`examples/demo.py` runs tide-ann end to end on synthetic data in a local Docker cluster.

```bash
docker build -f docker/Dockerfile -t tide-ann:latest .
docker build --platform linux/amd64 -f docker/Dockerfile.tools -t tide-ann-tools:latest .
python3 examples/demo.py --workdir /tmp/tide-demo
```

It needs Docker with BuildKit (Docker Desktop, or Docker Engine 23 or newer with buildx) and Python 3.10 or
newer with pyzmq. The tools image is linux/amd64 only because diskannpy publishes x86-64 wheels only, so a
Linux arm64 host needs amd64 emulation, for example
`docker run --privileged --rm tonistiigi/binfmt --install amd64`. Docker Desktop includes it.
`--tools-python PATH` runs the data steps with a local Python (numpy, scipy, pymetis, diskannpy) instead of
the tools image.

## Steps

The demo generates 20,000 x 64 vectors in 32 overlapping Gaussian clusters and 4,096 queries (70% from
clusters 0, 1 and 2). It builds a Vamana graph with one thread, partitions it over four servers with METIS,
trains PQ, and runs one cycle with search list size 200:

1. search all queries on the base graph
2. insert 4,000 vectors from the hot cluster that causes the most work on the busiest server, taken from
   that server's part of the cluster, then drain and search again
3. run a controller phase (`controller-start`, 196,608 queries, `controller-stop`) and search again

The controller settings are passed explicitly (`REPAIR` and `WINDOW` in `examples/demo.py`) and printed.
The demo exits 1 if recall@10 drops below 90% (`--min-recall`) or a membership check fails.
`python3 examples/demo.py --help` lists the options.

## Sample output

A run with the default settings (seed 1). The numbers vary with the seed and slightly from run to run.

```
data preparation (seed 1)
  data: 20,000 x 64 base vectors, center scale 0.3, 4,096 queries (70% from clusters 0,1,2), insert pool 16,000 per hot cluster
  synthetic: 0.2 s
  pool-0: 0.2 s
  pool-1: 0.2 s
  pool-2: 0.2 s
  vamana: 3.5 s
  runtime: 14.4 s
  pq: 0.7 s
  partition-vectors: 0.4 s
  hot-queries: 0.1 s
starting 4 servers and the coordinator
search on the base graph (L = 200)
insert phase: busiest server 1 (33% of distance computations)
  cluster 0: 955 queries, 1,945,047 distance computations on server 1 (39%)
  cluster 1: 964 queries, 1,097,310 distance computations on server 1 (22%)
  cluster 2: 948 queries, 638,557 distance computations on server 1 (13%)
  inserting from cluster 0
  batch: 0.7 s
  gt: 7.8 s
insert phase: 4,000 vectors from cluster 0 (4,000 nearest to server 1), then drain
controller phase: controller-start, the query stream, controller-stop
  settings: window_sec=0.25 cooldown_sec=0.25 min_window_queries=8192 proposal_query_limit=172032 repair=refine:0.01,balance:0.004,tau:0.2,eps:0.03,skew:1.02,cooldown:1,cum:1.0,slack:0.10,hop_slack:0.05,ema:0.5,converge:3,guard:0.10,recall_guard:off,balance_target:underloaded
  query stream: 196,608 queries (48 shuffled passes)

                            recall@10   hops  imbalance   work per server (%)  nodes owned per server
base graph                     94.96%  18.90      1.302   [9, 33, 30, 28]      [5001, 4999, 5000, 5000]
after the insert phase         94.45%  14.48      1.920   [8, 48, 22, 23]      [5001, 8999, 5000, 5000]
after the controller phase     94.50%  15.17      1.345   [22, 34, 22, 22]     [8804, 5092, 5088, 5016]

insert phase:     imbalance 1.302 -> 1.920 (+47.5%), 4,000 vectors from cluster 0, 4,000 joined server 1
controller phase: imbalance 1.920 -> 1.345 (-29.9%), 18 windows, 18 batches committed (18 kept, 0 reverted), 4,339 node moves, 1,521,240 of 8,618,664 bytes moved
memory per server (vectors + PQ codes + PQ tables, MB): 31.9, 31.9, 31.9, 31.9
imbalance = max / mean distance computations per server
done in 92.3 s, replies and logs in /tmp/tide-demo
```

Each indented time is one data step, and its output is in `<workdir>/logs/<step>.log`. `synthetic` makes the
base vectors and queries, `pool-<c>` the insert pool of hot cluster c, `vamana` the graph, `runtime` the METIS
partitions, ground truth and query stream, `pq` the PQ codes, `partition-vectors` each server's own vectors,
`hot-queries` one query file per hot cluster, `batch` the insert batch and `gt` the ground truth after the
inserts.

hops: server visits per query, including the entry server.

## Docker Compose

`docker/compose.yml` serves a demo's work directory with the same four servers, through the Compose v2
plugin (`docker compose`). `TIDE_IMAGE` selects another server image than `tide-ann:latest`.

```bash
export TIDE_WORK=/tmp/tide-demo
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml port coordinator 29699      # -> 127.0.0.1:<port>
python3 tools/coord_client.py --endpoint tcp://127.0.0.1:<port> recall --queries 1000
docker compose -f docker/compose.yml down
```
