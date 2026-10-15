# tide-ann

tide-ann is an approximate nearest neighbor (ANN) index for data that keeps growing. One
[Vamana](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf) graph
is split into partitions, one per in-memory index server. New vectors are inserted while the servers answer
queries, and a controller repartitions the graph at runtime by moving nodes between servers to follow the
query work.

- Each index server (`tide-server`) owns the adjacency rows of one partition. The partitions are first
  computed with [METIS](https://github.com/KarypisLab/METIS). By default a server keeps the original vectors
  of its own nodes and a product quantization (PQ) code of every node. `--full-vectors` keeps every vector
  on every server instead.
- A query's search state (beam and visited set) is passed from server to server over ZeroMQ,
  following [the BatANN design](https://arxiv.org/abs/2512.09331). The coordinator (`tide-coordinator`)
  picks the entry point from a small sample of the nodes and collects the results.
- An insert first runs a search. The server that owns the nearest result links the new node with
  alpha-RNG pruning, as in [FreshDiskANN](https://arxiv.org/abs/2105.09613). Backlinks and broadcasts reach
  the other servers in batches, and `drain` waits until they are applied.
- A controller phase runs from `controller-start` to `controller-stop`. While queries run, the controller
  counts the distance computations of every node. It moves small batches of nodes away from the busiest
  server, or towards the server that owns most of their neighbors, and reverts a batch that raised the
  peak or summed server cost, or the hops per query, by more than a set tolerance. Inserts are rejected
  during a controller phase.

## Build

Requirements: Linux, a C++17 compiler, CMake 3.20 or newer, libzmq and the header `zmq.hpp` from cppzmq
4.10.0. On Debian or Ubuntu:

```bash
sudo apt install build-essential cmake libzmq3-dev
sudo curl -fsSL -o /usr/local/include/zmq.hpp https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp
```

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix <dir>
```

The binaries are `tide-server`, `tide-coordinator` and `tide-build-pq`. On x86-64 the CPU must support AVX2
and FMA. On arm64 the distance code uses NEON.

The Python tools in `tools/` need numpy, scipy and pymetis, diskannpy for the Vamana build and pyzmq for
the admin client. Their tests run with:

```bash
PYTHONPATH=$PWD python3 -m unittest discover -s tools -t .
```

## Quick start

The demo generates a small synthetic dataset, starts four servers and a coordinator in Docker, inserts
4,000 vectors and runs one controller phase. It needs Docker with BuildKit and Python 3.10 or newer with
pyzmq.

```bash
docker build -f docker/Dockerfile -t tide-ann:latest .
docker build --platform linux/amd64 -f docker/Dockerfile.tools -t tide-ann-tools:latest .
python3 examples/demo.py --workdir /tmp/tide-demo
```

The tools image is linux/amd64 only. On a Linux arm64 host, register amd64 emulation first, as shown on
[the demo page](docs/demo.md), which also explains the steps and shows a sample run.

## Documentation

- [Running the demo](docs/demo.md)
- [Preparing a dataset](docs/prepare_dataset.md)
- [Running a cluster](docs/run_cluster.md)
- [Admin commands](docs/admin_commands.md)
- [The controller](docs/controller.md)
- [Architecture](docs/architecture.md)
- [Storage modes](docs/pq_storage.md)
- [Data formats](docs/data_format.md)
- [Result metrics](docs/metrics.md)
- [Debug output](docs/debug.md)

## Status

tide-ann is a research prototype, tested on one machine with Docker. Distances are squared L2, and there
are no deletes, replicas or fault tolerance. See the docs for details.

## License

MIT, see [LICENSE](LICENSE). The PQ code is ported from DiskANN (MIT), see [NOTICE](NOTICE).
