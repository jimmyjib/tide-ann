# tide-ann

**A dynamic multi-partition graph ANN index: online inserts and workload-adaptive repartitioning on one
running cluster.**

tide-ann cuts one Vamana graph into K partitions, one per in-memory index server. Two things change
while the cluster serves:

- **The data grows.** New vectors are inserted online; each one joins the partition of the server that
  owns its nearest neighbour.
- **The partitioning follows the query work.** A controller measures, per node v, the search work W(v)
  that the real queries cause on v's server, and moves nodes between servers in small bursts to spread
  that work. A burst that made the measured cost or the hop rate worse by more than a tolerance is
  undone.

Insert phases and repartitioning (controller) phases alternate on the same running cluster. An operator
starts and ends each controller phase with admin commands (`controller-start`, `controller-stop`), and the
controller acts only while a search load runs. Every controller phase starts with a drain, inserts are
refused while it runs, and it ends only when its last burst is judged, so a node move never races an
insert, and inserted nodes are measured and moved like the others.

The repository holds the servers, the coordinator, the data tools, a local end-to-end demo and the
documentation in [`docs/`](docs/). It makes no performance claim
([Limitations and evidence status](#limitations-and-evidence-status)).

## Why

A partitioning is computed once, when the index is loaded, to give every server the same number of
nodes. Equal node counts are not equal work, and the data keeps changing after the partitioning was
computed:

- **Queries concentrate work.** Queries that favour some regions of the data make the servers owning
  those regions do most of the distance computations, even when every server owns the same number of
  nodes. In the demo, 70 % of the queries come from 3 of 32 clusters; in its sample run
  ([What the demo prints](#what-the-demo-prints)) the busiest server does 1.30 times the mean distance
  work on the freshly loaded graph.
- **Inserts move the data, and can move the work with it.** An insert joins the partition of its
  nearest neighbour, so an insert stream that is denser in some regions grows the partitions that own
  them. When those are regions the queries favour, the new nodes add work to servers that are already
  busy. The demo inserts where the work is: it measures which server does the most distance
  computations and which hot cluster's queries cause most of them there, and inserts 4,000 vectors of
  that cluster into that server's part of it. In the sample run this raises the work imbalance from
  1.30 to 1.92.

The controller of tide-ann balances the measured query work. A controller phase measures W(v) for every
node the coordinator knows and moves a few nodes at a time at runtime: *balance* moves nodes away from
the busiest server, *refine* moves nodes towards the server that owns most of their neighbours (so a
search crosses servers less often), and a burst that made the per-server cost or the hops per query
worse by more than a tolerance is undone. Node counts and memory per server are not a goal, and the
controller leaves them uneven. In the demo's sample run the controller phase lowers the work imbalance
from 1.92 to 1.34 with recall unchanged. That is the demo's synthetic data, built so that the
controller has work to move; it is not a performance measurement. The alternative, re-partitioning
offline, means rebuilding the partitions and reloading the servers; tide-ann moves nodes on the running
cluster instead, in controller phases that an operator starts between insert phases (there is no
background loop).

## How it works

- **Partitioned storage.** Each index server (`tide-server`) owns the adjacency rows of its nodes. By
  default it holds the original vectors of its own nodes only, plus a product-quantization (PQ) code of
  every node (hybrid PQ own-only storage); `--full-vectors` replicates every vector instead. No server
  holds the whole graph.
- **State-passing search.** Following the BatANN design [1], a query's search state (beam and visited
  set) is forwarded from server to server over ZeroMQ (P2P), so each step runs where the next
  neighbourhood is stored. The coordinator (`tide-coordinator`) picks the entry point, dispatches the
  queries and collects the results.
- **Online inserts.** For each new vector the coordinator runs a search; the owner of the nearest result
  links the new node into its partition (alpha-RNG pruning, as in FreshDiskANN [3]); backlink and
  broadcast messages update the other servers. Servers apply them in batches; `drain` waits until every
  one is applied.
- **Window controller.** In a controller phase the coordinator opens short statistics windows on every
  server while a search command runs. From each window it reads W(v) for every node and how many of
  v's out-neighbours each server owns, and plans a burst: *refine* moves nodes towards the server that
  owns most of their neighbours, *balance* moves nodes away from the busiest server. The burst is
  applied as one stop-the-world MOVE (PREPARE, COMMIT, ACCEPT): PREPARE sends every server the whole
  move list with every moved node's row and digests, and under PQ storage each node's original vector to
  its new owner only. The next window judges it; a burst that raised the per-server cost or the hops per
  query beyond a tolerance is reverted by the inverse MOVE. Byte budgets, caps and a rejection memory
  bound how much moves.
- **Phases.** `controller-start` drains pending insert work and covers every node the coordinator knows,
  inserted ones included; `insert` is refused until `controller-stop`, which is refused while a burst
  waits for its verdict. Inserts after a phase reuse the row slots that accepted moves freed.

## What it is not

- Not a replicated service: no server holds the whole graph, and a search crosses servers.
- Not a benchmark: the demo data is synthetic and shows that the pieces work together, not how fast or
  how accurate the system is.
- Not disk-based: every server keeps its partition in memory. There are no deletes and no replicas.
- Distances are squared L2 only (cosine through normalisation, below).

## Architecture at a glance

```
              admin client (tools/coord_client.py: one ZeroMQ REQ per command, port 29699)
                                        |
                                 tide-coordinator
      head index, query dispatch, insert pipeline, drain, runtime repartitioning controller
            |                     |                     |                       |
      tide-server 0         tide-server 1         tide-server 2     ...    tide-server K-1
      partition 0           partition 1           partition 2              partition K-1
            \___________ P2P search-state forwarding, backlinks, broadcasts ___________/
```

Time on one cluster runs in phases:

```
 load ──> insert phase ──> controller phase ──> insert phase ──> controller phase ──> ...
          insert, drain     controller-start (drains), search load,
                            controller-stop
```

Searches (`recall`) run in any phase; the controller acts only while a search command is running,
because its windows measure the work of that search load. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
describes the processes, messages and data flow; [docs/CONTROLLER.md](docs/CONTROLLER.md) the
repartitioning controller.

## Quick start (Docker demo)

Requirements:

- Docker Desktop, or Docker Engine 23 or newer with the buildx plugin (BuildKit), and the Compose v2
  plugin (`docker compose`) for [Docker Compose](#docker-compose). `DOCKER_BUILDKIT` must not be 0:
  the tools image's platform check reads the target platform that BuildKit passes, and the legacy
  builder leaves it empty, so the check refuses the build.
- On a Linux arm64 host, amd64 emulation registered first. The usual way to register it with QEMU
  is `docker run --privileged --rm tonistiigi/binfmt --install amd64`; Docker Desktop includes it.
- Python 3.10+ with `pyzmq` on the host. The data steps run in the tools image; with `--tools-python`
  they run in that interpreter instead, which then needs numpy, scipy and pymetis (and diskannpy,
  unless `--graph-image` names an image for the Vamana step).

The two image builds download Ubuntu packages, the cppzmq header and Python wheels (about 0.4 GB in all,
plus the Ubuntu base image if it is not already local).

```bash
docker build -f docker/Dockerfile -t tide-ann:latest .
docker build --platform linux/amd64 -f docker/Dockerfile.tools -t tide-ann-tools:latest .
python3 examples/demo.py --workdir /tmp/tide-demo
```

The tools image needs `--platform linux/amd64` on every host (its Dockerfile refuses any other target):
diskannpy publishes x86-64 wheels only, and the demo runs the image as linux/amd64. On an arm64 host
Docker Desktop builds and runs it through its x86-64 emulation; Docker Engine on Linux arm64 needs the
QEMU emulation above. The server image's build stage compiles every target and runs the C++ tests.
These three commands were run on an arm64 macOS host with Docker Desktop, with other image tags (passed
to the demo with `--image` and `--tools-image`), from a pre-release snapshot of this code with an
earlier version of the demo ([Testing](#testing)); the demo described below has not run in these
images.

`examples/demo.py` generates a synthetic dataset from a seed (`--seed`, default 1): 20,000 x 64 vectors
from 32 overlapping Gaussian clusters, 4,096 queries (70 % of them from three hot clusters 0, 1 and 2)
and, for each hot cluster, a pool of vectors from that cluster only. It builds a Vamana graph with one
thread (on one machine with the same Python packages, the same seed gives the same graph and the same
partitions; another platform or other package versions can give other ones), partitions it with METIS,
trains PQ, starts four servers and a coordinator with P2P search, and runs one cycle, with search list
size 200 in every search command:

1. a search of all queries on the base graph;
2. the insert phase: 4,000 vectors chosen where the query work is are inserted, then a drain, then a
   search of all queries. The busiest server is the one with the most distance computations in the
   search on the base graph; the demo searches each hot cluster's queries alone, takes the hot cluster
   that causes the most distance computations on that server, and from that cluster's pool the vectors
   whose nearest existing vector that server owns (then the nearest to that server's part);
3. the controller phase: `controller-start`, a 196,608-query search load and `controller-stop`, then a
   search of all queries.

The controller settings are passed explicitly and printed (they are in the sample below): the default
byte budgets per window (refine 1 %, balance 0.4 % of the bytes) and the default `tau` (0.2); balance
only above a max/mean load of 1.02, towards any server below the mean load
(`balance_target:underloaded`); and verdict tolerances of 10 % for the peak and total cost and 5 % for
the hops per query ([Runtime repartitioning](#runtime-repartitioning)). The demo exits 1 when recall@10
falls below 90 % (`--min-recall`) at any of its three searches of all queries.
`python3 examples/demo.py --help` lists the options that change the workload (`--seed`,
`--center-scale`, `--noise`, `--hot-clusters`, `--inserts`, ...), the search list size (`--search-l`),
the stream length (`--stream-repeats`) and the controller token (`--repair`), and those that run the
data steps with a local Python (`--tools-python`) or use other images (`--image`, `--tools-image`,
`--bin-dir`, `--graph-image`, `--volume`).

### What the demo prints

A sample run of the default command (seed 1), every line it prints, with the work directory shown as
`/tmp/tide-demo`. It ran with the Python data steps on the host (`--tools-python`), the Vamana step in a
separate linux/amd64 image (`--graph-image`) and the binaries of this tree's build in linux/arm64
containers (`--bin-dir`, `--volume`, `--image`); the steps and the table layout are the same as in the
Quick start.

```
data preparation (seed 1)
  workload: 20,000 x 64 base vectors from 32 Gaussian clusters (centre scale 0.3, 12 intrinsic dimensions, noise 0.3); 4,096 queries, 70% from the hot clusters 0,1,2; per hot cluster an insert pool of 16,000 vectors from that cluster only
  synthetic: 0.2 s
  pool-0: 0.1 s
  pool-1: 0.1 s
  pool-2: 0.1 s
  vamana: 2.2 s
  runtime: 8.0 s
  pq: 0.4 s
  partition-vectors: 0.2 s
  hot-queries: 0.1 s
starting 4 servers and the coordinator
search on the base graph (search list size L = 200 in every search command)
insert phase: choosing the insert cluster. Busiest server in the search of all queries: 1 (33 % of the distance computations); each hot cluster's queries searched alone:
    hot cluster 0 (955 queries): 1,945,047 distance computations on server 1 (39 % of its work in the search of all queries)
    hot cluster 1 (964 queries): 1,097,310 distance computations on server 1 (22 % of its work in the search of all queries)
    hot cluster 2 (948 queries): 638,557 distance computations on server 1 (13 % of its work in the search of all queries)
  -> the insert batch comes from hot cluster 0
  batch: 0.4 s
  gt: 4.1 s
insert phase: 4,000 inserts from hot cluster 0, the vectors of its pool of 16,000 in or nearest to server 1's part of it (nearest existing vector on server 1: 4,000), then a drain
controller phase: controller-start, the query stream, controller-stop
  controller settings (the demo's, not recommendations): window_sec=0.25 cooldown_sec=0.25 min_window_queries=8192 proposal_query_limit=172032 repair=algo:local,refine:0.01,balance:0.004,tau:0.2,eps:0.03,skew:1.02,cooldown:1,cum:1.0,slack:0.10,hop_slack:0.05,ema:0.5,converge:3,guard:0.10,recall_guard:off,balance_target:underloaded
  query stream: 196,608 queries (48 shuffled passes over the 4,096)

                            recall@10   hops  imbalance   work per server (%)  nodes owned per server     membership
base graph                     94.96%  18.90      1.302   [9, 33, 30, 28]      [5001, 4999, 5000, 5000]   consistent
after the insert phase         94.45%  14.47      1.919   [8, 48, 22, 23]      [5001, 8999, 5000, 5000]   consistent
after the controller phase     94.51%  15.15      1.344   [22, 34, 22, 22]     [8797, 5098, 5099, 5006]   consistent

insert phase:     4,000 vectors of hot cluster 0 at the busiest server 1 (4,000 of them joined it); imbalance 1.302 -> 1.919 (+47.4 %)
controller phase: imbalance 1.919 -> 1.344 (-30.0 %); 18 planning windows; 18 bursts committed: 18 kept, 0 rejected and reverted; 4335 node moves committed; 1,519,912 of 8,617,720 bytes moved (weight w)
memory per server (vectors + PQ codes + PQ tables, MB): 31.9, 31.9, 31.9, 31.9
imbalance = max / mean over the servers of the distance computations of that search of all queries (1.0 = even); work per server = each server's share of them
insert batch = vectors of the hot cluster whose queries, searched alone, cause the most distance computations on the server that was busiest in the search on the base graph, taken from that server's part of the cluster
membership = every server's ownership and own vectors equal the coordinator's membership (membership-check)
done in 63.5 s; replies and logs in /tmp/tide-demo
```

- **recall@10**: the share of each query's 10 true nearest neighbours that the search returned, over
  the 4,096 queries; after the insert phase, against the exact ground truth of the base plus every
  inserted vector.
- **hops**: the average number of server visits of a query's search state, the entry server included
  (1 = the search finished on its entry server). With overlapping clusters and search list size 200 a
  search state moves between servers many times.
- **imbalance**: the busiest server's distance computations during that search divided by the mean
  over the servers (1.0 = even); **work per server**: each server's share of them, in percent.
- **nodes owned per server**: the coordinator's membership. The 20,000 base nodes start split evenly
  (METIS can leave a server one node over or under).
- **membership**: `consistent` means `membership-check` passed: every server's owner labels, rows and
  own original vectors (byte for byte) agree with the coordinator's membership.
- **insert phase** line: the hot cluster the batch came from, the busiest server, how many of the
  4,000 new nodes joined that server (the owner of each new node's nearest neighbour), and the change of
  the imbalance.
- **controller phase** line: the planning windows of the phase; the bursts committed, of which those
  the verdict accepted are kept and those it rejected were undone by an inverse MOVE ("rejected and
  reverted"); the node moves committed (forward moves only); and the bytes moved (forward MOVEs plus the
  inverse MOVEs of reverted bursts) out of the total bytes (adjacency rows plus vectors) of all nodes.
- **memory**: each server's allocated vector, PQ-code and PQ-table bytes. The stores grow in chunks of
  100,000 rows, so on data this small every server allocates the same.

How to read it:

- The skewed queries make server 1 the busiest from the start (33 % of the distance work). Searched
  alone, hot cluster 0's queries cause 39 % of server 1's work, more than the other two hot clusters', so
  the batch comes from hot cluster 0. All 4,000 vectors join server 1, which then does 48 % of the work:
  the imbalance rises from 1.302 to 1.919. Recall is 0.51 points lower (the ground truth now includes
  the new vectors).
- The controller phase moves 4,335 nodes in 18 accepted bursts (17.6 % of the bytes) and lowers the
  imbalance to 1.344, with recall unchanged (94.45 % and 94.51 %) and 5 % more hops per query. Server 1
  is still the busiest (34 %), now with 5,098 nodes; server 0 has grown to 8,797.
- `membership-check` is consistent at all three points.

The data is synthetic and the numbers vary with the seed; the demo shows the mechanism (inserts, a
controller phase, node moves and the consistency checks on one running cluster), not a performance
result.

### Docker Compose

`docker/compose.yml` serves the demo's data with the same four-server layout (`TIDE_IMAGE` selects
another server image than `tide-ann:latest`):

```bash
export TIDE_WORK=/tmp/tide-demo        # compose reads it for every subcommand
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml port coordinator 29699      # -> 127.0.0.1:<port>
python3 tools/coord_client.py --endpoint tcp://127.0.0.1:<port> recall --queries 1000
docker compose -f docker/compose.yml down
```

These commands were run against a demo's work directory, on a server image built from a pre-release
snapshot of this code ([Testing](#testing)), each with `-p <name>` added so the run had a project of its
own: the servers loaded the base graph, `recall` answered on the published port, `membership-check`
passed and `down` left no container or network behind.

## Build from source

Linux, a C++17 compiler, CMake 3.20 or newer, libzmq and the cppzmq header `zmq.hpp`
(Debian/Ubuntu: `build-essential cmake libzmq3-dev`, plus `zmq.hpp` from cppzmq 4.10.0).

```bash
cmake -S . -B build                          # Release is the default build type
cmake --build build --parallel 6
ctest --test-dir build --output-on-failure   # 20 C++ test programs
```

These commands were run on this tree on linux/arm64 (GCC 11.4), and `cmake -S . -B build` was also
checked without a build type (the cache gets `CMAKE_BUILD_TYPE=Release`). The binaries are
`tide-server`, `tide-coordinator` and `tide-build-pq`; `cmake --install build --prefix <dir>` installs
them into `<dir>/bin` (`docker/Dockerfile` does this). The distance kernels use AVX2 and FMA on x86-64
(`-mavx2 -mfma`, so the CPU must support both) and NEON on arm64. **x86-64 has been built and tested
only under emulation**, from a pre-release snapshot of this code: `docker build --platform linux/amd64
-f docker/Dockerfile .` on an arm64 host compiled it with these flags and its 20 C++ tests passed under
Docker Desktop's x86-64 emulation. This tree has not been built for x86-64, no native x86-64 CPU has run
the code, and the CI workflow (`.github/workflows/ci.yml`), which builds and tests on x86-64, has not
run yet. `tide-build-pq` uses OpenMP when the compiler provides it.

## Prepare a dataset

The data tools are Python modules under `tools/prepare` (numpy; scipy and pymetis for partitioning;
diskannpy for the Vamana build). [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) describes every file.

Distances are squared L2 only. For cosine data, normalise every vector set to unit length first
(`python3 -m tools.prepare.normalize_vectors --input A.npy --output A_unit.npy ...`; covered by unit
tests only) and compute ground truth on the normalised files.

The sequence the demo runs, from an empty dataset directory (every path below is relative to it) with
the repository on `PYTHONPATH`; `<build>` is the CMake build directory:

```bash
export PYTHONPATH=<repository root>
python3 -m tools.prepare.make_synthetic --output source --seed 1 --n-base 20000 --dim 64 --center-scale 0.3 \
    --noise 0.3 --intrinsic-dim 12 --n-query 4096 --hot-clusters 0,1,2 --hot-share 0.7 --n-insert 1 \
    --insert-cluster 0 --insert-share 1
for c in 0 1 2; do
  python3 -m tools.prepare.make_synthetic --output pool-$c --seed 1 --n-base 20000 --dim 64 --center-scale 0.3 \
      --noise 0.3 --intrinsic-dim 12 --n-query 4096 --hot-clusters 0,1,2 --hot-share 0.7 --n-insert 16000 \
      --insert-cluster $c --insert-share 1
done
python3 -m tools.prepare.build_vamana --input source/base.npy --output index --metric l2 --threads 1
python3 -m tools.prepare.prepare_demo_runtime --dataset-root . --parts 4 --stream-repeats 48 --name demo --checkpoint demo
mkdir pq
<build>/tide-build-pq runtime/base.npy pq/pq_pivots.bin 16 15 --codes pq/pq_codes.bin
python3 -m tools.prepare.write_partition_vectors --runtime runtime --pq-dir pq
```

- `make_synthetic` writes `source/` (base, queries, inserts and base + inserts; the demo uses the base
  and the queries, and its one insert vector is not used); use `tools.prepare.slice_dataset` to cut the
  same layout from your own float32 vectors. Each `pool-<c>/` run writes the same base and queries (they
  have their own random streams) and 16,000 insert vectors of hot cluster c only, from which the insert
  phase chooses its batch.
- The demo chooses its insert batch while the cluster runs, from the measured work, with its own
  numpy steps (`examples/demo.py --data-step hot-queries` and `--data-step insert-batch`), and computes
  the ground truth of the base plus the batch with `tools.prepare.generate_groundtruth_mkdir`.
- `build_vamana` builds the graph (R 64, L 100, alpha 1.2) into `index/`; with `--threads 1` the same
  input gives the same graph (with more threads the build is faster but the graph varies between runs).
- `prepare_demo_runtime` partitions the graph (`--parts` = K, 2 to 64), picks the head-index sample,
  computes exact top-10 ground truth for the base and for base + inserts, builds the query stream of the
  controller phase (`stream/`, 48 shuffled passes over the queries) and seals `runtime/`, the
  coordinator's data directory.
- `tide-build-pq <base.npy> <pivots> M iters --codes <codes>` trains PQ (M must divide the dimension;
  the demo uses M = dim / 4) and encodes every base vector.
- `write_partition_vectors` writes each partition's original vectors (`pq/partition_<r>/vectors.npy`),
  checks them against the codes and seals the PQ directory.

Then give the servers their data directory. A server reads one flat layout and only its own
`partition_<r>/`, so one directory can serve every rank (use `cp` instead of `ln` across filesystems):

```bash
mkdir -p server
ln runtime/membership.npy pq/pq_pivots.bin pq/pq_codes.bin server/
for r in 0 1 2 3; do
  mkdir -p server/partition_$r
  ln runtime/partition_$r/adj runtime/partition_$r/local_to_global.npy pq/partition_$r/vectors.npy server/partition_$r/
done
```

Every server then runs with `--data-dir server` and the coordinator with `--data-dir runtime`. (The demo
links one directory per server, `views/server-<r>/`, holding only that server's files, and serves the
coordinator `stream/query.npy` and `stream/groundtruth.npy` as its `query.npy` and `groundtruth.npy`.)

## Run a K-server cluster

Every server needs `--num-servers K`, its `--rank` and, for P2P search, `--peers` with all K addresses
in rank order (its own included). The coordinator needs `--servers` with the same list and `--p2p`.
The demo started its four servers and the coordinator with these arguments (in Docker, through
`tools/local_cluster.py`):

```bash
tide-server --data-dir /data --rank 0 --port 28800 --num-servers 4 --threads 4 \
    --insert-batch-size 128 --backlink-batch-size 128 --broadcast-batch-size 128 --pq \
    --peers tcp://server0:28800,tcp://server1:28800,tcp://server2:28800,tcp://server3:28800
# ... the same for ranks 1, 2, 3, each with its own data directory

tide-coordinator --data-dir /data --p2p --queries 196608 --batch-size 32 --L 50 --K 10 --W 8 \
    --servers tcp://server0:28800,tcp://server1:28800,tcp://server2:28800,tcp://server3:28800
```

The coordinator waits up to `--cluster-check-sec` (default 300) for every server, refuses a cluster
whose servers run different storage modes or dimensions, and then listens for admin commands on
`tcp://*:29699`. The beam width `--W` is the number of beam entries a server expands per visit; it
travels in every search state, so every server uses it. Notation used throughout: K is the number of
servers, while the coordinator flag `--K` is the result size (top-k, default 10); W(v) is the search work
of node v and W[p] the load of server p, while the flag `--W` is the beam width.
`tools/local_cluster.py` starts, drives and removes such a cluster in Docker. Its docker argv builders
and its reply parser have unit tests; its command line (`python3 -m tools.local_cluster up|send|down`)
has not been run. The demo uses the same code as a module.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#process-options) lists every option of both binaries.

## Admin commands

Send one command per request with `tools/coord_client.py` (or any ZeroMQ REQ socket). Everything after
the client's own options is sent as written (a `--` before the command is accepted and dropped). Most
commands answer with a JSON result between the lines `===JSON_RESULT===` and `===JSON_END===`. A refusal
is either a line starting `Error:` (workload and migration argument errors, and commands refused during
a controller phase or an open migration) or the command's JSON result with `"status": "error"`
(`controller-start`, `-stop`, `-abort`, `membership-check`, and a failed migration, `drain` or `reset`);
[docs/ADMIN_COMMANDS.md](docs/ADMIN_COMMANDS.md#replies) lists which. The client exits 0 for a reply that
is not an error; 1 for an error reply (JSON `"status": "error"`, an `Error:` line, an unknown command),
an endpoint ZeroMQ rejects or a ZeroMQ send or receive error; 2 for no reply within `--timeout` and for a
usage error. ZeroMQ connects in the background, so a coordinator that is stopped or unreachable, a wrong
port or a host name that does not resolve shows up as a timeout (exit 2) after `--timeout` (default
300000 ms, 5 minutes); pass a short `--timeout`, for example `--timeout 5000`, for a liveness check.

The snippets below define a shell function `c` (it works in bash and zsh) and replace `<port>` with the
coordinator's admin port:

```bash
c() { python3 tools/coord_client.py --endpoint "tcp://127.0.0.1:<port>" "$@"; }
c stats
c recall --queries 4096 --query-file /control/query.npy --groundtruth /control/groundtruth.npy \
    --batch-size 32 --dispatch-window 256 --no-route-trace --no-merge --search-l 200
c membership-check --write /work/membership-recall-base.npy
c recall --queries 955 --query-file /control/query-hot-0.npy --no-groundtruth \
    --batch-size 32 --dispatch-window 256 --no-route-trace --no-merge --search-l 200
c insert 4000 --file /control/insert.npy --start-id 0 --batch-size 32 --dispatch-window 256
c drain
```

(The demo sends this command text through `tools/local_cluster.py`'s `send`, not through
`coord_client.py`; the paths are inside the coordinator's container. The demo runs `stats` before and
after every search and takes each server's distance computations from the difference; the search of
one hot cluster's queries alone, without ground truth, is how it finds the cluster that causes the
busiest server the most work. `--start-id 0` is the first row of the file; the new ids continue after
the known ones.)

| Command | Purpose |
|---|---|
| `recall` (`baseline`) | search a query set; recall@10 against ground truth, hops, timing |
| `insert N --file F [--start-id S]` | insert N vectors from rows S.. of F (ids continue after the known ones) |
| `mixed ...` | interleaved searches and inserts in one command |
| `drain` | wait until every insert, backlink and broadcast is applied on every server |
| `controller-start <journal_dir> [options]` | start a controller phase (drains first) |
| `controller-status`, `controller-stop`, `controller-abort` | inspect, end, or end a failed controller phase |
| `migration-apply`, `migration-accept`, `migration-rollback` | apply a node-move plan by hand |
| `membership-check [--write F.npy]` | compare every server's ownership and own vectors with the coordinator's membership |
| `status`, `stats`, `ping` | coordinator state; every server's counters; a network round-trip baseline |
| `reset`, `recreate-head` | reload the servers' load-time state; rebuild the head index |
| `shutdown` | stop every server, then the coordinator |
| `quit`, `exit` | stop the coordinator only (the servers keep running) |

[docs/ADMIN_COMMANDS.md](docs/ADMIN_COMMANDS.md) gives every argument, reply and refusal;
[docs/METRICS.md](docs/METRICS.md) explains the result fields and [docs/DEBUG.md](docs/DEBUG.md) the
`--debug` output.

## Storage modes

- `--pq` (the default): hybrid PQ own-only storage. A server holds the original vectors of the nodes it
  owns and a PQ code of every node. Owned neighbours are scored exactly, other neighbours with
  asymmetric PQ distances, and every expanded node is re-scored exactly on its owner; entries without an
  exact distance are dropped at the end, so the returned results carry exact distances.
- `--full-vectors`: every server loads every vector (`base.npy`); every distance is exact.

All servers of a cluster must run the same mode. The coordinator keeps `base.npy` and every inserted
vector in both modes (head index, `recreate-head`, `reset`, `membership-check`). The PQ mode has not
been measured on a serving cluster; two earlier implementations of it were, and were slower than full
replication ([Limitations](#limitations-and-evidence-status)). [docs/STORAGE_PQ.md](docs/STORAGE_PQ.md)
describes the scoring, the memory and the files.

## Runtime repartitioning

A controller phase:

```bash
c controller-start /work/controller window_sec=0.25 cooldown_sec=0.25 min_window_queries=8192 \
    proposal_query_limit=172032 \
    repair=algo:local,refine:0.01,balance:0.004,tau:0.2,eps:0.03,skew:1.02,cooldown:1,cum:1.0,slack:0.10,hop_slack:0.05,ema:0.5,converge:3,guard:0.10,recall_guard:off,balance_target:underloaded
c recall --queries 196608 --no-groundtruth --batch-size 128 --dispatch-window 256 --no-route-trace --no-merge \
    --search-l 200
c controller-status
c controller-stop
```

(The demo's commands, with `c` as above and sent the same way; it runs them once, after its insert
phase. The journal directory must exist and be writable by the coordinator. The demo's coordinator loaded its 196,608-row query stream as `query.npy`, and
`proposal_query_limit` keeps the last 24,576 queries for verdicts only.)

- **Phases.** `controller-start` refuses while a controller or a migration is active, drains pending
  insert maintenance, and covers every id the coordinator knows. `insert`, `mixed`, `recreate-head`,
  `reset` and `migration-apply` are refused until `controller-stop`. `controller-stop` is refused while
  a committed burst still waits for its verdict: keep searching until it resolves, then stop again.
- **Weights.** `weight:w` (the default) is W(v); `weight:wnb` adds the work of v's in-neighbours (W^nb),
  which makes the harvest (the statistics every server sends the coordinator at the end of a window)
  about 3.7 times larger at K = 4.
- **Budgets.** `refine` and `balance` are per-window byte budgets and `cum` the cumulative cap, all as
  fractions of the total bytes (adjacency row plus, under `--pq`, the vector of every node).
- **Verdict.** A burst is reverted when the peak or total per-server cost grew by more than `slack`, or
  the hops per query by more than `hop_slack`. `recall_guard:<pp>` (off by default) also reverts it
  when recall fell by more than `pp` percentage points; it needs ground truth on the search load.

The demo's settings suit its short controller phase on a small synthetic dataset (the default budgets
and `tau`; balance to any server below the mean load; looser verdict tolerances than the defaults); they
are not recommended settings.
[docs/CONTROLLER.md](docs/CONTROLLER.md) gives the window, the planner rules, the fence and MOVE
protocol, the budgets, caps and history, the guard and every option with its default.

## Testing

```bash
ctest --test-dir build --output-on-failure                   # C++ unit tests (assert stays on in Release)
PYTHONPATH=$PWD python3 -m unittest discover -s tools -t .   # Python unit tests (numpy, scipy, pymetis, pyzmq)
```

What has been run on this tree:

- **C++**: a clean Release build and its 20 C++ test programs, in a linux/arm64 Docker container
  (GCC 11.4, NEON kernels).
- **ThreadSanitizer**: the six C++ test programs that run concurrent code, built with
  `-fsanitize=thread` in the same image, ran without a report; one of them runs a coordinator workload
  with `--debug` against fake servers while inserts grow the membership
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#race-detection) has what was covered and the limits).
- **Python unit tests**: its 58 tests, on an arm64 macOS host with Python 3.13 (numpy 2.4, scipy 1.17,
  pymetis 2025.2, pyzmq 27.1).
- **`examples/demo.py`**, end to end, with its Python data steps on the macOS host Python (numpy 2.4.1,
  pymetis 2025.2.2), its Vamana step in a separate linux/amd64 image with Python 3.10 and diskannpy
  0.7.0, and binaries byte-identical to the build above in linux/arm64 containers
  ([What the demo prints](#what-the-demo-prints) shows a run).
- **Larger local runs** on a slice of real vectors (200,000 x 96, 4 and 8 servers, one machine) of
  inserts, controller phases on clusters holding inserted ids, manual moves, both storage modes and both
  weights, on the binaries of this code. They are regression checks, not measurements.

What has been run only from a pre-release snapshot of this code (not rebuilt from this tree):

- **The Docker path**: the two images of `docker/` were built on an arm64 macOS host with Docker
  Desktop (the server image's build stage ran the 20 C++ tests), an earlier version of
  `examples/demo.py` ran on those two images only, as [Quick start](#quick-start-docker-demo) runs it
  (every data step in the tools image, linux/amd64 under emulation, with Python 3.10, numpy 1.25.0,
  scipy 1.11.4, pymetis 2023.1.1 and diskannpy 0.7.0; `tide-build-pq`, the four servers and the
  coordinator in the server image), and the commands of [Docker Compose](#docker-compose) ran on the
  server image, serving the data of a demo run.
- **x86-64**: the server image built for linux/amd64 under emulation on that arm64 host (AVX2 and FMA
  kernels); its 20 C++ tests passed.

Not run: the CI workflow (`.github/workflows/ci.yml`: x86-64 build and C++ tests, Python tests; no CI
run yet), the Python unit tests in its pinned Python 3.10 environment (numpy 1.25.0, scipy 1.11.4,
pymetis 2023.1.1, pyzmq 25.1.2; the tools image has the same versions but no pyzmq, and ran the data
steps of an earlier version of the demo, not the unit tests), any native x86-64 CPU, the Docker images
built from this tree, and Docker on a Linux host.

## Limitations and evidence status

[docs/LIMITATIONS.md](docs/LIMITATIONS.md) has the full list. In short:

- **Earlier measurements** of earlier research versions of this code, which differ from the code in
  this repository (not reproduced here; no numbers are claimed for them except the controller and PQ
  results below): static placements on 4, 8 and 16 servers; the runtime controller; online inserts into
  the partitioned graph.
- **The runtime controller: two completed runs of an earlier version.** Both used the W^nb weight on 4
  servers and one dataset (GIST, 1,000,000 vectors of 960 dimensions), search only on a fixed graph, with
  full replication, and two planner settings; a third run ended without its final measurement. Each run
  measured the unchanged placement and the controller on the same machines. After the controller phase,
  with the controller off, throughput was 40.5 % and 49.3 % above the unchanged placement's, with
  recall@10 at 86.01 % and 86.02 % against its 85.92 % (at the search setting used); over the controller
  phase itself, its MOVE pauses included, throughput was 1.11 and 1.01 times the unchanged placement's.
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md#evidence-status) has the details.
- **PQ storage: only earlier implementations were measured, and they were slower.** Two earlier
  implementations of the PQ storage ran on 4 servers on one 128-dimensional dataset (SIFT, 500,000
  vectors). Both used about 75 % less memory per server than full replication. The first, with the same
  scoring rule as this code, used about 14 % more CPU per query and more hops per query (about 2.2
  against 1.9, counting every server a query visits). The second, which scored all navigation with PQ,
  lost about 2.4 recall points and searched and inserted more slowly. The hybrid PQ storage in this
  repository (the default `--pq`) has not been measured. Besides the two controller runs above, these
  are the only numbers from those earlier measurements that this repository quotes; they are quoted
  because they bear on the default and are unfavourable.
- **Locally tested only**: the default weight W(v); MOVE with vectors (the measured moves carried
  adjacency only); inserts alternating with controller phases; and the controller at K = 8. End-to-end
  runs used K = 4 and 8 only; other server counts (2 to 64 are accepted) have unit tests only.
- **Inserts with the controller**: the local run that alternates them ran on the binaries of this code
  with the verdict forced to accept every burst (`slack` and `hop_slack` 1.0, so every accepted move of
  an inserted node could be checked at its new owner) and with the demo's verdict settings, where a
  burst holding inserted nodes was rejected and reverted and every one of those nodes was back on its
  previous owner, byte-exact.
- **The demo supports no performance claim**: synthetic data, one machine, small sizes.
- The verdict's hop clause can follow window noise on short windows; the recall guard's reference is
  unpaired; ThreadSanitizer ran on the concurrency tests, not on the server and coordinator programs;
  x86-64 has been built and tested only under emulation, from a pre-release snapshot (build and C++
  tests); there has been no CI run yet.

## References

This work builds on:

1. N. A. Dang, B. Landrum and K. Birman. *Passing the Baton: High Throughput Distributed Disk-Based
   Vector Search with BatANN.* arXiv:2512.09331 (v2), 2026. (State-passing search between partitions;
   the head index.)
2. S. Jayaram Subramanya et al. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a
   Single Node.* NeurIPS 2019. (The Vamana graph; the PQ module is ported from the DiskANN reference
   implementation, see [NOTICE](NOTICE).)
3. A. Singh et al. *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity
   Search.* arXiv:2105.09613, 2021. (Insertion into a Vamana graph.)
4. G. Karypis. *METIS: A Software Package for Partitioning Unstructured Graphs, Partitioning Meshes,
   and Computing Fill-Reducing Orderings of Sparse Matrices*, version 5.1.0 manual, 2013. (Load-time
   partitioning, through pymetis.)

They are cited as the sources of the designs named in parentheses. Third-party code in this repository
is listed in [NOTICE](NOTICE).

## Licence

MIT; see [LICENSE](LICENSE). The PQ module is ported from DiskANN (MIT); see [NOTICE](NOTICE).
