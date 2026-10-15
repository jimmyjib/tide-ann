# Preparing a dataset

The tools under `tools/prepare` build a dataset directory for a K-server cluster. They need numpy, scipy,
pymetis and diskannpy. Run them with the repository root on `PYTHONPATH`.

1. `make_synthetic` writes base, query and insert vectors to `source/`. Use `slice_dataset` to cut the
   same layout from your own float32 vectors.
2. `build_vamana` builds the graph (R 64, L 100, alpha 1.2) into `index/`. With `--threads 1` the same
   input gives the same graph on one platform with the same package versions. More threads build faster,
   but the graph then changes from run to run.
3. `prepare_runtime` partitions the graph into `--parts` servers with METIS, picks the head-index
   sample, computes the ground truth, writes `runtime/`, the coordinator's data directory, and builds a
   query stream for a controller phase.
4. `tide-build-pq` trains PQ and encodes the base vectors. M must divide the dimension.
5. `write_partition_vectors` writes each partition's original vectors to `pq/partition_<r>/vectors.npy`.

These are the commands for a 4-server cluster on synthetic data, run from an empty dataset directory.
`<build>` is the CMake build directory.

```bash
export PYTHONPATH=<repository root>
python3 -m tools.prepare.make_synthetic --output source --seed 1 --n-base 20000 --dim 64 --center-scale 0.3 \
    --noise 0.3 --intrinsic-dim 12 --n-query 4096 --hot-clusters 0,1,2 --hot-share 0.7 --n-insert 4000
python3 -m tools.prepare.build_vamana --input source/base.npy --output index --metric l2 --threads 1
python3 -m tools.prepare.prepare_runtime --dataset-root . --parts 4 --stream-repeats 48
mkdir pq
<build>/tide-build-pq runtime/base.npy pq/pq_pivots.bin 16 15 --codes pq/pq_codes.bin
python3 -m tools.prepare.write_partition_vectors --runtime runtime --pq-dir pq
```

Then give the servers their data directory. A server reads one flat layout and only its own
`partition_<r>/`, so one directory can serve every rank. Use `cp` instead of `ln` across filesystems.

```bash
mkdir -p server
ln runtime/membership.npy pq/pq_pivots.bin pq/pq_codes.bin server/
for r in 0 1 2 3; do
  mkdir -p server/partition_$r
  ln runtime/partition_$r/adj runtime/partition_$r/local_to_global.npy pq/partition_$r/vectors.npy server/partition_$r/
done
```

Every server then runs with `--data-dir server` and the coordinator with `--data-dir runtime`
([run_cluster.md](run_cluster.md)). [data_format.md](data_format.md) describes every file.

For cosine similarity, normalize the vectors to unit length with `python3 -m tools.prepare.normalize_vectors`
before building the index and the ground truth.
