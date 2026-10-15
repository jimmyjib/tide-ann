# Limitations and evidence status

This repository makes no performance claim. This page says what has been measured, what has only been
tested, and what is known to behave in a way you should plan for.

## Evidence status

**Earlier measurements.** Earlier research versions of this code, which differ from the code in this
repository, were measured on several servers. Those measurements are not reproduced by this repository,
and apart from the controller runs and the PQ storage results below no numbers from them are quoted here.
They covered:

- static (offline) placements of the partitions on 4, 8 and 16 servers;
- the runtime controller with the W^nb weight, on 4 servers and one dataset (GIST, 1,000,000 vectors of
  960 dimensions), search only on a fixed graph (no inserts), with full replication. Two runs completed,
  with two planner settings (`tau` 0.2 and `skew` 1.05; `tau` 0.3 and `skew` 1.02); a third run ended
  without its final measurement and gives no result. Each run measured the unchanged placement and the
  controller on the same machines, with probes of 100,000 queries:
  - after the controller phase, with the controller off: 15,896 against 11,316 queries per second for
    the unchanged placement (40.5 % more) in the first run, 18,470 against 12,373 (49.3 % more) in the
    second; recall@10 85.92 % for the unchanged placement and 86.01 % and 86.02 % after the controller
    phase (at the search setting used);
  - over the controller phase itself (about 127 s of serving, with MOVE pauses of 7.6 s and 6.9 s in
    all): 1.11 and 1.01 times the unchanged placement's throughput;
  - the controller's skew went from 1.55 to 1.04 and 1.01 in 57 and 53 bursts, of which 5 and 7 were
    reverted, and about 37 % of the nodes ended on another server;
- online inserts into the partitioned graph;
- two earlier implementations of PQ own-only storage, on 4 servers with P2P search and one
  128-dimensional dataset (SIFT, 500,000 vectors). Both kept the original vectors of a server's own
  nodes and a PQ code of every node, and both used about 75 % less memory per server than full
  replication. Both were slower than full replication:
  - the first used the same scoring rule as this repository (exact distances for owned neighbours, PQ
    distances for the others, an exact re-rank of every expanded node); it used about 14 % more CPU per
    query and more hops per query (about 2.2 against 1.9, counting every server a query visits);
  - the second scored all navigation with PQ distances; it lost about 2.4 recall points (about 96.5 %
    against 99 % recall@10) and searched and inserted more slowly.

  The PQ numbers are quoted because they bear on the default storage mode and are unfavourable. The
  hybrid PQ storage in this repository has not been measured on a serving cluster.

**Tested locally only** (C++ unit tests, and local multi-server runs on one machine with a 200,000 x 96
slice of real vectors and with the synthetic demo data; regression checks, not measurements):

- the default weight W(v) (every measured controller run used W^nb);
- hybrid PQ own-only storage, the default `--pq` (above);
- MOVE shipping the original vector (the measured moves carried adjacency only), and the byte budgets
  that count vectors;
- **server counts**: the local end-to-end runs used K = 4 and K = 8 only (the controller included; the
  measured controller ran at K = 4). K = 2, 3 and 9 to 64 are accepted but covered by unit tests only:
  the planner at K = 2, 3, 4, 8 and 16; the controller history, a 16-frame statistics harvest and a
  migration plan round trip at K = 16; and the refusal of more than 64 servers;
- **inserts alternating with controller phases, and controller phases that move inserted nodes**: a
  local run with two insert phases (2,000 inserts each) and three controller phases, on 4 servers with
  PQ storage, in two settings:
  - with `slack` and `hop_slack` set to 1.0, so that every burst is accepted and every accepted move of
    an inserted node can be checked at its new owner (membership check, exact vector, found by its own
    vector as a query): six bursts, all accepted, moved 730 inserted nodes, and every one was checked
    at its new owner;
  - with the demo's verdict settings (`slack:0.10`, `hop_slack:0.05`): the first burst (3,497 nodes,
    216 of them inserted) was rejected on its hop rate and reverted, and all 216 inserted nodes were
    back on their previous owners, with byte-exact vectors, and found by their own vectors at distance
    0; three later bursts were accepted, and the 529 inserted nodes they moved were checked at their
    new owners.

  Manual moves of inserted nodes and their inverse MOVE ran separately;
- the recall guard, whose reference is unpaired (below).

**The demo supports no performance claim**: synthetic data, one machine, small sizes. Its data is built so
that the controller has something to do: 32 overlapping clusters (neighbouring clusters share graph
edges, so the partitions share long borders across which nodes can move) and 70 % of the queries from
three of them. It runs one cycle: a search on the base graph, an insert phase that inserts 4,000 vectors
of one hot cluster into the busiest server's part of it, chosen from the measured distance work, and a
controller phase with the explicit settings it prints. The README's "What the demo prints" shows a
sample run. The data is synthetic and the numbers vary with the seed; the demo shows the mechanism, not
a performance result.

## Race detection

The six C++ test programs that run concurrent code were built with ThreadSanitizer
(`-fsanitize=thread`, GCC 11.4, linux/arm64) and ran without a report:

- `chunked_array`: the chunked stores, the PQ code-ready flag and the per-node lock store;
- `stats_window`: the window counters, the counters of inserted ids and the snapshot encoder thread;
- `shutdown_protocol`: the server's work queues and its shutdown quiescence;
- `server_concurrency`: searches while insert, backlink and broadcast batches are applied by several
  workers (both storage modes, with freed row slots reused; the test requires searches that overlap the
  batches, measured from both sides' time intervals), inserts whose candidates name ids of other servers
  whose broadcast and backlink are still queued or being applied (the test checks that no row and no
  reported edge names an id the server has not received), then searches while window snapshots are
  taken;
- `window_controller_lifecycle`: the controller's worker thread;
- `coordinator_admin`: the coordinator's pipeline and admin commands against in-process fake servers,
  including a `mixed --debug` command whose searches and inserts the fakes answer: its inserts grow the
  membership while the metric thread writes the debug log and merges pending inserts into results, and
  every owner label in the log is checked against the membership.

The paths these tests exercise include an inserted node's owner label, which its backlink and its
broadcast both write; the per-node lock table, which an insert grows while searches read it; under
`--full-vectors`, an inserted node's replicated vector, which its backlink and its broadcast both carry
and which searches and prunes read only once it is complete; and another server's inserted node that an
insert meets before its broadcast has arrived (the insert drops that candidate, in both storage
modes).

Limits of this evidence:

- the `tide-server` and `tide-coordinator` programs never ran under the sanitizer, only these tests.
  They link the server's worker sources and the coordinator's pipeline, but not the server's main loop
  (`tide_server.cpp`) or real sockets; a race is found only on a path a test runs. No test runs the
  server's statistics reply, which reads rows under their locks (a new row is allocated before it is
  published);
- GCC 11's ThreadSanitizer does not intercept `pthread_cond_clockwait`, which the C++ library uses for
  timed condition-variable waits, and then reports a false "double lock". The test runs preloaded a
  small replacement built on `pthread_cond_timedwait`; a control program showed the false report
  without it and none with it, and a deliberate data race was still reported with it preloaded;
- the test container ran with Docker's seccomp filter off, because the arm64 sanitizer runtime turns off
  address-space randomisation, which the default filter refuses. It had no network and no capabilities.

## Not run

- **x86-64 on a real CPU, and this tree on x86-64 at all.** Every run of the servers and the
  coordinator used linux/arm64 (NEON distance kernels). The x86-64 path (AVX2 and FMA) was built, and
  its 20 C++ tests passed, only under Docker Desktop's x86-64 emulation on an arm64 host
  (`docker build --platform linux/amd64 -f docker/Dockerfile .`), and only from a pre-release snapshot
  of this code; this tree was not built for x86-64. The CI workflow (`.github/workflows/ci.yml`), which
  builds and tests on x86-64, has not run yet. x86-64 binaries require a CPU with AVX2 and FMA.
- **The pinned Python environment for the unit tests.** The Python unit tests ran with Python 3.13 and
  numpy 2.4 on an arm64 macOS host, not with the Python 3.10 / numpy 1.25.0 / scipy 1.11.4 / pymetis
  2023.1.1 / pyzmq 25.1.2 that the CI workflow pins. The data steps of an earlier version of the demo
  did run with Python 3.10 and those numpy, scipy and pymetis versions, in the `docker/Dockerfile.tools`
  image (which has no pyzmq); this demo's data steps have not run there.
- **The CI workflow's inputs.** Its actions are pinned to commit SHAs; the apt packages, the runner image
  and the Python patch release float.
- **The Docker images built from this tree, and the images and Compose file on a Linux host**
  (`docker/`): the two images were built from a pre-release snapshot of this code, and an earlier version
  of the demo and the Compose commands ran on them on one arm64 macOS host with Docker Desktop only (the
  tools image under x86-64 emulation). This demo has not run in these images. The tools image needs
  BuildKit (its platform check reads `TARGETPLATFORM`, which the legacy builder leaves empty). The base
  image is pinned by digest, the cppzmq header by checksum and the Python wheels by version; the apt
  package versions float with the Ubuntu archive.
- **Admin commands** `ping`, `shutdown`, `quit` and `exit`; the `baseline` alias only by a unit test of
  its refusal (a missing default ground truth); `mixed` only in unit tests, against in-process fake
  servers ([ADMIN_COMMANDS.md](ADMIN_COMMANDS.md#what-ran-in-this-release)).
- **Large id spaces.** Every per-server table grows in chunks of 100,000 ids up to 1,000 chunks, so at
  most 100,000,000 ids (load-time plus inserted). The rows a server owns (adjacency and per-node locks)
  grow in chunks of 10,000, so at most 10,000,000 rows per server. A load-time size just below a chunk
  boundary is not covered by a test; the local runs used up to about 210,000 ids.

## Known behaviour

- **The verdict can follow window noise.** With short windows the hop rate varies between windows at a
  fixed placement by a few percent (about 6 % in the local runs), more than the default `hop_slack`
  (0.02) and about as large as the 0.05 the demo uses, so small bursts can be accepted or reverted by
  noise. Longer windows make the verdict more stable.
- **The recall guard's reference is unpaired.** `recall_guard:<pp>` compares the verify window's recall
  with the trigger window's, over different queries of the same load, so a change of the query mix
  between the two windows enters the comparison. Choose `pp` above the window-to-window recall spread.
- **The controller acts only while a search command runs**, and a burst committed near the end of a
  search command keeps `controller-stop` refused until another search command resolves it. The
  proposal limit counts the queries of the current search command, so a search run only to resolve a
  burst can itself plan and commit new bursts until its own count reaches the limit
  ([CONTROLLER.md](CONTROLLER.md#budgets-caps-and-memory)).
- **A later controller phase needs the same configuration** and no manual MOVE since the last phase,
  or `controller-start` is refused; `reset` clears it ([CONTROLLER.md](CONTROLLER.md#phases-with-inserts)).
- **Inserts and the controller never run at the same time.** Each controller phase starts with a drain;
  inserts are refused until `controller-stop`.
- **The controller balances query work, not node counts or memory.** It moves nodes to reduce the
  imbalance of the priced search work W per server (refine also moves nodes towards the server that owns
  most of their neighbours) and leaves the node counts uneven. Node counts and per-server memory (rows,
  and the own original vectors under `--pq`) are neither balanced nor bounded per server; the byte
  budgets limit only how much moves. A server that received most of the inserts keeps its larger share
  after a controller phase unless its work moves nodes off it.
- **Balance can find no admissible move.** Balance moves a node only to a server that owns at least
  `tau` of the node's out-neighbours (the lightest server, or with `balance_target:underloaded` any
  server below the mean load). When the busiest server's work sits in a region whose nodes have their
  neighbours almost only on that server, balance moves nothing and the imbalance stays; refine moves
  only nodes whose neighbours are mostly on another server.
- **Partial insert batches wait.** A server applies inserts, backlinks and broadcasts in batches; a
  partial batch waits for a full one or for `drain`. There is no age-based flush. Search results can lag
  the inserts; `drain` before a check that needs every insert applied. An insert whose candidates
  include another server's new node that has not reached this server yet drops that candidate (counted
  as `insert_neighbors_undelivered` in `stats`), so concurrent inserts can link slightly fewer
  neighbours than a sequential run.
- **Window counters of inserted ids** grow only when a window begins; a server that has not applied
  every insert's broadcast refuses the window, which is why `controller-start` drains first.
- **Self-query misses under `--pq`.** A few nodes (in the local runs, a fixed handful of ids, moved or
  not) are not found in their own top-10 with L = 50 because their PQ estimate keeps them out of the
  beam; their vectors are exact and `recall --search-l 200` finds them at distance 0.
- **Ground truth is static.** After inserts, recall is meaningful only against ground truth computed on
  the base plus the inserted vectors (`base_plus_insert.npy`).
- **Distances are squared L2.** Cosine data must be normalised to unit length before indexing
  (`tools/prepare/normalize_vectors.py`); there is no cosine build.
- **Memory.** An accepted MOVE frees the source row and vector slot into a pool that later inserts and
  moves reuse. Stores grow in chunks of 100,000 rows, so small partitions under `--pq` can use as much
  memory as full replication ([STORAGE_PQ.md](STORAGE_PQ.md#memory)). The coordinator keeps every
  vector of the dataset in every storage mode.
- **No deletes, no replicas, no fault tolerance.** A node lives on exactly one server; there is no
  delete command; a server that fails stops the commands that need it.
- **One admin command at a time.** The coordinator handles commands sequentially. `recall --batch-size`,
  `--dispatch-window` and `--debug` stay set for later commands.
- **Debug output** goes to the coordinator's working directory, which must be writable
  ([DEBUG.md](DEBUG.md)); a file that cannot be written is a warning and the command still runs.
- **Insert throughput** in the insert results measures the search and the dispatch of each insert, not
  the time until every server has applied it ([METRICS.md](METRICS.md)).
- **Harvest timing.** Each controller window collects its statistics from every server (the harvest)
  within a deadline of 25 s x max(1, K / 4). How long a harvest takes on graphs larger than the local
  runs' (about 210,000 ids) has not been measured.
