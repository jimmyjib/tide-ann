#!/usr/bin/env python3
"""End-to-end demo: synthetic data -> Vamana graph -> METIS partitions -> PQ -> a local 4-server cluster, then
new data arriving where the query work is, followed by a controller phase.

Steps (everything is written under ``--workdir``, which must not exist yet):

1. ``make_synthetic`` (seed ``--seed``): 20,000 x 64 vectors from 32 overlapping Gaussian clusters
   (centre scale 0.3, isotropic noise 0.3: neighbouring clusters share graph edges, so the partitions
   share long borders across which the controller can move work) and 4,096 queries (70 % from the three
   hot clusters 0, 1 and 2); then, once per hot cluster, an insert pool of 4 x ``--inserts`` vectors
   drawn from that cluster only (make_synthetic draws the base vectors and the queries from their own
   random streams, so every pool run writes the same base and queries; the demo checks their hashes and
   keeps only the pool's insert vectors);
2. a Vamana graph (R 64, L 100, alpha 1.2; diskannpy) built with one thread, so on one platform with
   the same package versions a seed gives the same graph and the same METIS partitions run to run
   (another platform or other versions of numpy and its BLAS/LAPACK, pymetis or diskannpy can give
   other ones, and with them other numbers);
3. ``prepare_demo_runtime``: node-balanced METIS partitions (one per server), head ids, exact ground
   truth for the base, the sealed runtime and a query stream;
4. PQ pivots and codes (``tide-build-pq``, M = dim / 4) and every partition's original vectors, then one
   data directory per server holding only what that server loads (hybrid PQ own-only storage);
5. a 4-server cluster with P2P state passing (``tools/local_cluster.py``), then, with search list
   size ``--search-l`` (200) in every search command:
   a. a search of all queries on the base graph;
   b. the insert phase: an insert batch of ``--inserts`` (4,000) vectors chosen from that search (rule
      below) is inserted, then a drain; a search of all queries (ground truth of the base plus every
      inserted vector);
   c. the controller phase: ``controller-start`` with the demo's controller settings (printed before
      the phase), a stream of ``--stream-repeats`` (48) shuffled passes over the queries,
      ``controller-stop``; a search of all queries.

Insert batch rule (measured, not configured). The busiest server is the one with the most distance
computations in the search of all queries on the base graph. The demo searches the queries of each hot
cluster alone and counts the distance computations each hot cluster's queries cause on that server; the
batch comes from the hot cluster that causes the most of them (the hot cluster whose region carries the
largest share of the busiest server's work; a tie goes to the cluster listed first). From that cluster's
pool it takes the vectors that lie in the busiest server's part of the cluster: the vectors whose
nearest existing vector the busiest server owns (the coordinator's membership, written by
``membership-check --write`` at that search), then the ones nearest to that server's part. A new vector
joins the partition of its nearest neighbour, so the new data lands where the query work already
concentrates, as in a workload whose popular region keeps growing; the controller phase then rebalances
the measured work.

For every search of all queries the demo prints recall@10, the average hops per query, the
distance-work imbalance (max / mean over the servers of the distance computations that search caused),
each server's share of that work and the nodes each server owns, and checks (``membership-check``)
that every server's ownership and own vectors equal the coordinator's membership. The data is
synthetic: the numbers show that the pieces work together, not how fast or how good the system is.

Requirements on the host: Docker, Python 3.10+ with pyzmq. The data steps run in the tools image
(``docker/Dockerfile.tools``: numpy, scipy, pymetis, diskannpy) unless ``--tools-python`` names a local
interpreter that has those packages; the cluster runs in the image of ``docker/Dockerfile``::

    docker build -f docker/Dockerfile -t tide-ann:latest .
    docker build --platform linux/amd64 -f docker/Dockerfile.tools -t tide-ann-tools:latest .
    python3 examples/demo.py --workdir /tmp/tide-demo

The tools image is linux/amd64 only (diskannpy publishes x86-64 wheels only); the data steps run it with
``--platform linux/amd64`` (``--tools-platform``), so it must be built with that platform. The demo's own
small numpy steps (one query file per hot cluster, the insert batch) run the same way, as
``demo.py --data-step ...``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.local_cluster import LABEL, ClusterError, LocalCluster, docker, send, user_args  # noqa: E402

# The demo's controller settings, passed explicitly and printed before the controller phase. They make a
# short phase on a small synthetic dataset show its effect; they are not recommended settings. The planner
# keys (docs/CONTROLLER.md): the default per-window byte budgets (refine 1 %, balance 0.4 % of all bytes),
# tau (0.2) and weight (W(v)); balance moves to any server below the mean load, not only to the lightest
# (balance_target:underloaded), and only above a skew of 1.02 (default 1.05); a moved node rests one
# planning window (default 2); a phase may move forward up to the total bytes (default 0.45 of them); the
# verdict tolerates 10 % more cost and 5 % more hops per query per burst (defaults 3 % and 2 %).
REPAIR = ("algo:local,refine:0.01,balance:0.004,tau:0.2,eps:0.03,skew:1.02,cooldown:1,cum:1.0,slack:0.10,"
          "hop_slack:0.05,ema:0.5,converge:3,guard:0.10,recall_guard:off,balance_target:underloaded")
WINDOW = ["window_sec=0.25", "cooldown_sec=0.25", "min_window_queries=8192"]
# The last rows of the controller phase's query stream plan nothing new; they only judge the last burst.
VERIFY_TAIL_ROWS = 24576
SEARCH = ["--batch-size", "32", "--dispatch-window", "256", "--no-route-trace", "--no-merge"]
GROUNDTRUTH = ["--metric", "l2", "--k", "10", "--l2-kernel", "norm-dot", "--l2-float64"]
CHECKPOINTS = ("base", "after_insert", "after_controller")
# Rows of each hot cluster's insert pool, in batches: the insert phase picks its batch from this many batches.
POOL_FACTOR = 4


def log(message: str) -> None:
    print(message, flush=True)


class Steps:
    """Runs the Python data steps on the host or in the tools image, and the C++ PQ step in the server image."""

    def __init__(self, args: argparse.Namespace, work: Path):
        self.args, self.work = args, work

    def _container(self, step: str, image: str, platform: str | None, argv: Sequence[str],
                   repo: bool = True) -> list[str]:
        mounts = ["-v", f"{self.work}:/w"] + (["-v", f"{REPO}:/repo:ro"] if repo else [])
        return ["run", "--rm", "--pull", "never", "--name", f"{self.args.name}-{step}", "--label",
                f"{LABEL}={self.args.name}", "--network", "none", *user_args(),
                *(["--platform", platform] if platform else []), "-e", "PYTHONPATH=/repo",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-w", "/w", *mounts, image, *argv]

    def python(self, step: str, argv: Sequence[str], *, graph: bool = False) -> str:
        """``python <argv>`` in the work directory; ``{repo}`` in argv names this tree where the step runs.
        The Vamana step (``graph``) runs in ``--graph-image`` when one is given. Returns the step's output."""
        if graph and self.args.graph_image:
            image = self.args.graph_image
        else:
            image = None if self.args.tools_python else self.args.tools_image
        argv = [item.replace("{repo}", "/repo" if image else str(REPO)) for item in argv]
        started = time.monotonic()
        if image is None:
            env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([self.args.tools_python, "-B", *argv], cwd=self.work, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        else:
            result = docker(self._container(step, image, self.args.tools_platform, ["python3", "-B", *argv]),
                            timeout=3600, check=False)
        output = result.stdout
        (self.work / "logs").mkdir(exist_ok=True)
        (self.work / "logs" / f"{step}.log").write_text(output, encoding="utf-8")
        if result.returncode != 0:
            raise ClusterError(f"step {step} failed ({result.returncode}):\n{output[-3000:]}")
        log(f"  {step}: {time.monotonic() - started:.1f} s")
        return output

    def build_pq(self, m: int) -> None:
        started = time.monotonic()
        argv = [f"{self.args.bin_dir}/tide-build-pq", "/w/runtime/base.npy", "/w/pq/pq_pivots.bin", str(m), "15",
                "--codes", "/w/pq/pq_codes.bin"]
        command = self._container("pq", self.args.image, None, argv, repo=False)
        index = command.index(self.args.image)
        extra = [item for spec in self.args.volume for item in ("-v", spec)]
        command[index:index] = extra
        result = docker(command, timeout=3600, check=False)
        (self.work / "logs" / "pq.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            raise ClusterError(f"tide-build-pq failed ({result.returncode}):\n{result.stdout[-3000:]}")
        log(f"  pq: {time.monotonic() - started:.1f} s")


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def server_views(work: Path, servers: int) -> list[Path]:
    """One directory per server with only the files it loads under hybrid PQ storage (no base.npy)."""
    views = []
    for rank in range(servers):
        view = work / "views" / f"server-{rank}"
        link(work / "runtime" / "membership.npy", view / "membership.npy")
        for name in ("pq_pivots.bin", "pq_codes.bin"):
            link(work / "pq" / name, view / name)
        for leaf in ("adj", "local_to_global.npy"):
            link(work / "runtime" / f"partition_{rank}" / leaf, view / f"partition_{rank}" / leaf)
        link(work / "pq" / f"partition_{rank}" / "vectors.npy", view / f"partition_{rank}" / "vectors.npy")
        views.append(view)
    return views


def source_revision() -> str:
    try:
        top = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, capture_output=True)
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", "examples/demo.py"],
                                 text=True, capture_output=True)
    except OSError:
        return "unknown"
    return top.stdout.strip() if top.returncode == 0 and tracked.returncode == 0 else "unknown"


def hot_clusters(args: argparse.Namespace) -> list[int]:
    return [int(item) for item in args.hot_clusters.split(",") if item.strip()]


def pool_rows(args: argparse.Namespace) -> int:
    """Rows of a hot cluster's insert pool, which the insert phase chooses its batch from."""
    return POOL_FACTOR * args.inserts


def workload_argv(args: argparse.Namespace, cluster: int, inserts: int) -> list[str]:
    """The make_synthetic options of the demo's workload, with ``inserts`` insert vectors of one cluster."""
    return ["--seed", str(args.seed), "--n-base", str(args.n_base), "--dim", str(args.dim),
            "--center-scale", f"{args.center_scale:g}", "--noise", f"{args.noise:g}",
            "--intrinsic-dim", str(args.intrinsic_dim), "--n-query", str(args.queries),
            "--hot-clusters", args.hot_clusters, "--hot-share", f"{args.hot_share:g}",
            "--n-insert", str(inserts), "--insert-cluster", str(cluster), "--insert-share", "1"]


def prepare(args: argparse.Namespace, work: Path) -> dict[int, int]:
    """The data steps; returns the number of queries of each hot cluster."""
    steps = Steps(args, work)
    hot = hot_clusters(args)
    log(f"data preparation (seed {args.seed})")
    log(f"  workload: {args.n_base:,} x {args.dim} base vectors from 32 Gaussian clusters (centre scale "
        f"{args.center_scale:g}, {args.intrinsic_dim} intrinsic dimensions, noise {args.noise:g}); {args.queries:,} "
        f"queries, {args.hot_share:.0%} from the hot clusters {args.hot_clusters}; per hot cluster an insert pool "
        f"of {pool_rows(args):,} vectors from that cluster only")
    # source: the base vectors and the queries (make_synthetic needs an insert set; its one vector is not used).
    steps.python("synthetic", ["-m", "tools.prepare.make_synthetic", "--output", "source",
                               *workload_argv(args, hot[0], 1)])
    for cluster in hot:
        pool = work / f"pool-{cluster}"
        steps.python(f"pool-{cluster}", ["-m", "tools.prepare.make_synthetic", "--output", pool.name,
                                         *workload_argv(args, cluster, pool_rows(args))])
        for name in ("base.npy", "query.npy", "query_labels.npy", "base_labels.npy"):
            if sha256_file(work / "source" / name) != sha256_file(pool / name):
                raise ClusterError(f"{pool.name}/{name} differs from source/{name}")
        for name in ("base.npy", "query.npy", "base_plus_insert.npy"):
            (pool / name).unlink()  # base and queries are source's (checked above); the pool keeps its inserts
    steps.python("vamana", ["{repo}/tools/prepare/build_vamana.py", "--input", "source/base.npy", "--output", "index",
                            "--metric", "l2", "--threads", str(args.graph_threads)], graph=True)
    steps.python("runtime", ["-m", "tools.prepare.prepare_demo_runtime", "--dataset-root", ".",
                             "--parts", str(args.servers), "--stream-repeats", str(args.stream_repeats),
                             "--name", "demo", "--checkpoint", "demo", "--producer-commit", source_revision()])
    (work / "pq").mkdir()
    steps.build_pq(args.pq_m or args.dim // 4)
    steps.python("partition-vectors", ["-m", "tools.prepare.write_partition_vectors", "--runtime", "runtime",
                                       "--pq-dir", "pq"])
    control = work / "control"
    for source, name in (("source/query.npy", "query.npy"), ("gt/base/groundtruth.npy", "groundtruth.npy")):
        link(work / source, control / name)
    steps.python("hot-queries", ["{repo}/examples/demo.py", "--data-step", "hot-queries", "--source", "source",
                                 "--hot-clusters", args.hot_clusters, "--output", "control"])
    counts = json.loads((control / "hot-queries.json").read_text(encoding="utf-8"))
    server_views(work, args.servers)
    return {cluster: int(counts[str(cluster)]) for cluster in hot}


class Session:
    """The admin commands of the demo; every reply is saved under <workdir>/replies."""

    def __init__(self, port: int, work: Path, servers: int, search_l: int):
        self.port, self.replies, self.servers, self.search_l = port, work / "replies", servers, search_l
        self.replies.mkdir(exist_ok=True)
        self.count = 0

    def run(self, name: str, command: Sequence[str], timeout: float = 600.0, ok: bool = True) -> dict[str, Any]:
        self.count += 1
        result, raw = send(self.port, command, timeout_seconds=timeout)
        (self.replies / f"{self.count:02d}-{name}.reply").write_text(raw, encoding="utf-8")
        if ok and result.get("status", "ok") not in ("ok", "success"):
            raise ClusterError(f"{name}: {json.dumps(result)[:2000]}")
        return result

    def stats(self, name: str) -> list[dict[str, Any]]:
        rows = self.run(name, ["stats"]).get("servers") or []
        by_rank = {row.get("partition_id"): row for row in rows if isinstance(row, dict)}
        if sorted(k for k in by_rank if isinstance(k, int)) != list(range(self.servers)):
            raise ClusterError(f"{name}: not every server answered")
        return [by_rank[rank] for rank in range(self.servers)]

    def search_work(self, name: str, command: Sequence[str]) -> tuple[dict[str, Any], list[int], list[dict]]:
        """One search command, the distance computations it caused on each server, and the stats after it."""
        before = self.stats(f"{name}-stats-before")
        reply = self.run(name, command)
        after = self.stats(f"{name}-stats-after")
        work = [a["search"]["distance_count"] - b["search"]["distance_count"] for a, b in zip(after, before)]
        return reply, work, after

    def measured_recall(self, name: str, queries: int, groundtruth: str) -> dict[str, Any]:
        reply, work, after = self.search_work(name, ["recall", "--queries", str(queries), "--query-file",
                                              "/control/query.npy", "--groundtruth", groundtruth, *SEARCH,
                                              "--search-l", str(self.search_l)])
        search = reply.get("search", {})
        # Every server's ownership and own vectors against the coordinator's membership.
        check = self.run(f"{name}-membership", ["membership-check", "--write", f"/work/membership-{name}.npy"],
                         ok=False)
        servers = check.get("servers") or []
        exact = (check.get("status") == "ok" and check.get("consistent") is True and len(servers) == self.servers
                 and all(row.get("consistent") is True and row.get("vectors_exact", True) is True
                         for row in servers))
        return {"recall_at_10_percent": search.get("recall"), "avg_hops": search.get("avg_hops"),
                "distance_work_by_server": work,
                "imbalance_max_over_mean": max(work) / (sum(work) / len(work)) if sum(work) > 0 else None,
                "work_share_by_server": [w / sum(work) for w in work] if sum(work) > 0 else None,
                "owned_nodes_by_server": check.get("coordinator_owned_by_rank"),
                "membership_consistent_and_vectors_exact": exact,
                "membership_file": f"work/coordinator/membership-{name}.npy",
                "memory_bytes_by_server": [{key: row["storage"][key] for key in
                                            ("vector_bytes", "code_bytes", "pq_table_bytes")} for row in after]}


def choose_insert_cluster(args: argparse.Namespace, session: Session, last: dict[str, Any],
                          hot_counts: dict[int, int]) -> dict[str, Any]:
    """The insert cluster rule: the hot cluster whose queries cause the most distance computations on the server
    that was busiest in the search of all queries on the base graph."""
    work = last["distance_work_by_server"]
    busiest = max(range(len(work)), key=lambda rank: (work[rank], -rank))
    share = work[busiest] / sum(work) if sum(work) > 0 else None
    log(f"insert phase: choosing the insert cluster. Busiest server in the search of all queries: "
        f"{busiest} ({100.0 * (share or 0.0):.0f} % of the distance computations); each hot cluster's queries "
        f"searched alone:")
    by_cluster: dict[int, list[int]] = {}
    for cluster in hot_clusters(args):
        _, by_cluster[cluster], _ = session.search_work(
            f"insert-probe-cluster-{cluster}",
            ["recall", "--queries", str(hot_counts[cluster]), "--query-file", f"/control/query-hot-{cluster}.npy",
             "--no-groundtruth", *SEARCH, "--search-l", str(args.search_l)])
        log(f"    hot cluster {cluster} ({hot_counts[cluster]:,} queries): {by_cluster[cluster][busiest]:,} distance "
            f"computations on server {busiest} ({100.0 * by_cluster[cluster][busiest] / max(1, work[busiest]):.0f} % "
            f"of its work in the search of all queries)")
    order = hot_clusters(args)
    chosen = max(order, key=lambda c: (by_cluster[c][busiest], -order.index(c)))
    log(f"  -> the insert batch comes from hot cluster {chosen}")
    return {"busiest_server": busiest, "busiest_server_work_share": share,
            "hot_cluster_queries": {str(c): hot_counts[c] for c in order},
            "hot_cluster_work_by_server": {str(c): by_cluster[c] for c in order}, "cluster": chosen,
            "membership_file": last["membership_file"]}


def insert_phase(args: argparse.Namespace, session: Session, steps: Steps, choice: dict[str, Any]) -> dict[str, Any]:
    """Write the batch (the rule's rows of the cluster's pool) and the ground truth of base + batch, insert the
    batch and drain."""
    cluster, server = choice["cluster"], choice["busiest_server"]
    batch, combined = "control/insert.npy", "gt-input/base-plus-inserts.npy"
    output = steps.python("batch", [
        "{repo}/examples/demo.py", "--data-step", "insert-batch", "--points", "source/base.npy",
        "--owners", choice["membership_file"], "--server", str(server), "--servers", str(args.servers),
        "--pool", f"pool-{cluster}/insert.npy", "--first-row", "0", "--pool-rows", str(pool_rows(args)),
        "--rows", str(args.inserts), "--output", batch, "--base-plus-inserts", combined])
    chosen = json.loads(output.strip().splitlines()[-1])
    steps.python("gt", ["-m", "tools.prepare.generate_groundtruth_mkdir", "--base", combined, "--query",
                        "source/query.npy", "--output", "gt/after-insert/groundtruth.npy", *GROUNDTRUTH])
    link(steps.work / "gt" / "after-insert" / "groundtruth.npy",
         steps.work / "control" / "groundtruth-after-insert.npy")
    log(f"insert phase: {args.inserts:,} inserts from hot cluster {cluster}, the vectors of its pool of "
        f"{pool_rows(args):,} in or nearest to server {server}'s part of it (nearest existing vector on server "
        f"{server}: {chosen['nearest_on_server']:,}), then a drain")
    insert = session.run("insert", ["insert", str(args.inserts), "--file", "/control/insert.npy",
                                    "--start-id", "0", "--batch-size", "32", "--dispatch-window", "256"],
                         timeout=3600)
    count = insert.get("insert", {}).get("count")
    if count != args.inserts:
        raise ClusterError(f"the insert phase inserted {count} of {args.inserts} vectors")
    drain = session.run("drain", ["drain"], timeout=600)
    return {"count": count, "pool_rows": pool_rows(args), "batch": chosen,
            "drain_complete": drain.get("drain_barrier_complete")}


def controller_phase(args: argparse.Namespace, session: Session, start: list[str], stream_rows: int) -> dict[str, Any]:
    log("controller phase: controller-start, the query stream, controller-stop")
    log(f"  controller settings (the demo's, not recommendations): {' '.join(start[2:])}")
    log(f"  query stream: {stream_rows:,} queries ({args.stream_repeats} shuffled passes over the {args.queries:,})")
    session.run("controller-start", start, timeout=600)
    stream = ["recall", "--queries", str(stream_rows), "--no-groundtruth", "--batch-size", "128",
              "--dispatch-window", "256", "--no-route-trace", "--no-merge", "--search-l", str(args.search_l)]
    session.run("stream", stream, timeout=3600)
    status = session.run("controller-status", ["controller-status"])
    stop = session.run("controller-stop", ["controller-stop"], ok=False)
    rounds = 0
    # A burst committed near the stream's end still waits for its verdict: keep searching until it resolves.
    while stop.get("status") != "ok" and stop.get("unresolved_action") is True and rounds < 4:
        rounds += 1
        session.run(f"resolve-{rounds}", ["recall", "--queries", str(min(16384, stream_rows)),
                                          "--no-groundtruth", *stream[4:]], timeout=3600)
        status = session.run(f"controller-status-{rounds}", ["controller-status"])
        stop = session.run(f"controller-stop-{rounds}", ["controller-stop"], ok=False)
    if stop.get("status") != "ok":
        raise ClusterError(f"controller-stop refused: {json.dumps(stop)[:1000]}")
    repair = status.get("repair", {}) if isinstance(status.get("repair"), dict) else {}
    # commits counts every burst committed, a later reverted one included; reverts the bursts the verdict
    # rejected and the inverse MOVE undid. accepted_migrations is the coordinator's count of membership
    # migrations accepted since load (forward and inverse MOVEs alike), not a verdict count.
    result = {key: status.get(key) for key in ("cycles", "commits", "reverts", "accepted_migrations",
                                               "actions_committed", "moved_bytes", "total_bytes", "verify_pending")}
    result.update({"windows": repair.get("windows"), "bursts": repair.get("bursts")})
    pending = 1 if status.get("verify_pending") is True else 0
    result.update({"kept": (result.get("commits") or 0) - (result.get("reverts") or 0) - pending,
                   "weight": (status.get("repair_config") or {}).get("weight"), "resolve_rounds": rounds})
    return result


def workload(args: argparse.Namespace, session: Session, steps: Steps, stream_rows: int,
             hot_counts: dict[int, int]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    log(f"search on the base graph (search list size L = {args.search_l} in every search command)")
    results["base"] = session.measured_recall("recall-base", args.queries, "/control/groundtruth.npy")
    tail = min(VERIFY_TAIL_ROWS, stream_rows // 4)
    start = ["controller-start", "/work/controller", *WINDOW, f"proposal_query_limit={stream_rows - tail}",
             f"repair={args.repair}"]
    results["controller_start"] = start
    choice = choose_insert_cluster(args, session, results["base"], hot_counts)
    results["insert"] = {**choice, **insert_phase(args, session, steps, choice)}
    groundtruth = "/control/groundtruth-after-insert.npy"
    results["after_insert"] = session.measured_recall("recall-after-insert", args.queries, groundtruth)
    results["controller"] = controller_phase(args, session, start, stream_rows)
    results["after_controller"] = session.measured_recall("recall-after-controller", args.queries, groundtruth)
    return results


def identity(work: Path, args: argparse.Namespace) -> dict[str, Any]:
    """The hashes that show whether two runs built the same data, graph, partitions and insert batch."""
    files = {"base": "source/base.npy", "query": "source/query.npy", "graph": "index/vamana",
             "membership": "runtime/membership.npy"}
    files.update({f"insert_pool_cluster_{c}": f"pool-{c}/insert.npy" for c in hot_clusters(args)})
    files["insert_batch"] = "control/insert.npy"
    return {name: sha256_file(work / path) for name, path in files.items()}


def plural(count: Any, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    return f"{before:.3f} -> {after:.3f} ({100.0 * (after - before) / before:+.1f} %)"


def report(results: dict[str, Any]) -> None:
    labels = {"base": "base graph", "after_insert": "after the insert phase",
              "after_controller": "after the controller phase"}
    log("")
    log(f"{'':26s} {'recall@10':>10s} {'hops':>6s} {'imbalance':>10s}   {'work per server (%)':20s} "
        f"{'nodes owned per server':26s} membership")
    for key in CHECKPOINTS:
        row = results[key]
        imbalance = row["imbalance_max_over_mean"]
        check = "consistent" if row["membership_consistent_and_vectors_exact"] else "MISMATCH"
        shares = row["work_share_by_server"]
        shares = "[" + ", ".join(f"{100.0 * x:.0f}" for x in shares) + "]" if shares else "n/a"
        log(f"{labels[key]:26s} {row['recall_at_10_percent']:9.2f}% {row['avg_hops']:6.2f} "
            f"{imbalance if imbalance is not None else float('nan'):10.3f}   {shares:20s} "
            f"{str(row['owned_nodes_by_server']):26s} {check}")
    imbalance = {key: results[key]["imbalance_max_over_mean"] for key in CHECKPOINTS}
    log("")
    insert, c = results["insert"], results["controller"]
    server = insert["busiest_server"]
    gained = (results["after_insert"]["owned_nodes_by_server"][server]
              - results["base"]["owned_nodes_by_server"][server])
    log(f"insert phase:     {insert['count']:,} vectors of hot cluster {insert['cluster']} at the busiest server "
        f"{server} ({gained:,} of them joined it); imbalance {change(imbalance['base'], imbalance['after_insert'])}")
    awaiting = ", 1 awaiting its verdict" if c.get("verify_pending") is True else ""
    log(f"controller phase: imbalance {change(imbalance['after_insert'], imbalance['after_controller'])}; "
        f"{plural(c['windows'], 'planning window')}; {plural(c['commits'], 'burst')} committed: {c['kept']} kept, "
        f"{c['reverts']} rejected and reverted{awaiting}; {plural(c['actions_committed'], 'node move')} "
        f"committed; {c['moved_bytes']:,} of {c['total_bytes']:,} bytes moved (weight {c['weight']})")
    memory = results["after_controller"]["memory_bytes_by_server"]
    log("memory per server (vectors + PQ codes + PQ tables, MB): "
        + ", ".join(f"{(m['vector_bytes'] + m['code_bytes'] + m['pq_table_bytes']) / 1e6:.1f}" for m in memory))
    log("imbalance = max / mean over the servers of the distance computations of that search of all queries "
        "(1.0 = even); work per server = each server's share of them")
    log("insert batch = vectors of the hot cluster whose queries, searched alone, cause the most distance "
        "computations on the server that was busiest in the search on the base graph, taken from that server's "
        "part of the cluster")
    log("membership = every server's ownership and own vectors equal the coordinator's membership "
        "(membership-check)")


def data_step(argv: list[str]) -> int:
    """The demo's own numpy steps, run like the other data steps (tools image or --tools-python)."""
    import numpy as np

    parser = argparse.ArgumentParser(prog="demo.py --data-step")
    sub = parser.add_subparsers(dest="step", required=True)
    hot = sub.add_parser("hot-queries", help="one query file per hot cluster (query-hot-<cluster>.npy)")
    hot.add_argument("--source", type=Path, required=True, help="make_synthetic output (query.npy, query_labels.npy)")
    hot.add_argument("--hot-clusters", required=True)
    hot.add_argument("--output", type=Path, required=True)
    batch = sub.add_parser("insert-batch", help="the batch of rows of an insert pool slice that lie in, or nearest "
                                                "to, one server's part of the data; and every vector plus the batch")
    batch.add_argument("--points", type=Path, action="append", required=True,
                       help="the existing vectors in id order (the demo passes the base)")
    batch.add_argument("--owners", type=Path, required=True, help="the owner of every existing id (uint32 .npy)")
    batch.add_argument("--server", type=int, required=True)
    batch.add_argument("--servers", type=int, required=True)
    batch.add_argument("--pool", type=Path, required=True)
    batch.add_argument("--first-row", type=int, required=True)
    batch.add_argument("--pool-rows", type=int, required=True)
    batch.add_argument("--rows", type=int, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--base-plus-inserts", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.step == "hot-queries":
        query = np.load(args.source / "query.npy")
        labels = np.load(args.source / "query_labels.npy")
        counts = {}
        for cluster in (int(item) for item in args.hot_clusters.split(",") if item.strip()):
            rows = np.ascontiguousarray(query[labels == cluster], dtype=np.float32)
            if not len(rows):
                raise SystemExit(f"no query comes from hot cluster {cluster}")
            np.save(args.output / f"query-hot-{cluster}.npy", rows)
            counts[str(cluster)] = int(len(rows))
        (args.output / "hot-queries.json").write_text(json.dumps(counts, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(counts, sort_keys=True))
        return 0
    pool = np.load(args.pool)
    end = args.first_row + args.pool_rows
    if args.first_row < 0 or not 1 <= args.rows <= args.pool_rows or end > len(pool):
        raise SystemExit(f"pool rows {args.first_row}..{end} lie outside the pool's {len(pool)} or hold no batch")
    points = np.ascontiguousarray(np.concatenate([np.load(path) for path in args.points]), dtype=np.float32)
    owners = np.load(args.owners).astype(np.int64)
    if owners.shape != (len(points),) or not 0 <= args.server < args.servers or owners.max() >= args.servers:
        raise SystemExit(f"{len(owners)} owners for {len(points)} existing vectors, or a server outside the cluster")
    candidates = np.ascontiguousarray(pool[args.first_row:end], dtype=np.float32)
    # Exact squared L2 to every existing vector (without the candidate's own norm, which ranks nothing): the
    # nearest one overall and the nearest one the server owns. gap = 0 when the nearest one is the server's.
    norms = np.einsum("ij,ij->i", points, points)
    on_server = owners == args.server
    nearest_owner = np.empty(len(candidates), dtype=np.int64)
    gap = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), 512):
        block = norms[None, :] - 2.0 * (candidates[start:start + 512] @ points.T)
        nearest = block.argmin(axis=1)
        nearest_owner[start:start + 512] = owners[nearest]
        gap[start:start + 512] = block[:, on_server].min(axis=1) - block[np.arange(len(block)), nearest]
    # The rows in the server's part first, then the nearest to it; ties by pool row. The batch keeps pool order.
    take = np.sort(np.lexsort((np.arange(len(candidates)), gap))[:args.rows])
    rows = candidates[take]
    # Insert row i of the batch becomes id N + i (N = the existing vectors), so row r of this file is id r.
    combined = np.concatenate([points, rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.base_plus_inserts.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, rows)
    np.save(args.base_plus_inserts, np.ascontiguousarray(combined, dtype=np.float32))
    print(json.dumps({"batch_rows": int(len(rows)), "base_plus_inserts_rows": int(len(combined)),
                      "pool_rows_on_server": int(np.sum(nearest_owner == args.server)),
                      "nearest_on_server": int(np.sum(nearest_owner[take] == args.server)),
                      "nearest_owner_counts": np.bincount(nearest_owner[take], minlength=args.servers).tolist()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", type=Path, required=True, help="new directory for data, logs and replies")
    parser.add_argument("--name", default="tide-demo", help="prefix of the Docker containers and network")
    parser.add_argument("--image", default="tide-ann:latest", help="image with the binaries")
    parser.add_argument("--bin-dir", default="/usr/local/bin", help="directory of the binaries in --image")
    parser.add_argument("--volume", action="append", default=[],
                        help="extra docker -v spec for the binary containers (e.g. a volume holding the binaries)")
    parser.add_argument("--tools-image", default="tide-ann-tools:latest")
    parser.add_argument("--tools-platform", default="linux/amd64", help="platform of the tools / graph images")
    parser.add_argument("--tools-python", help="run the Python data steps with this local interpreter instead")
    parser.add_argument("--graph-image", help="image for the Vamana step only (needs diskannpy)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the synthetic data")
    parser.add_argument("--graph-threads", type=int, default=1,
                        help="Vamana build threads (1: the same graph for the same seed on one platform with the "
                             "same package versions; more is faster but the graph, the partitions and the printed "
                             "numbers then vary between runs)")
    parser.add_argument("--servers", type=int, default=4)
    parser.add_argument("--server-threads", type=int, default=4)
    parser.add_argument("--n-base", type=int, default=20000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--center-scale", type=float, default=0.3,
                        help="spread of the cluster centres (make_synthetic; larger = more separated clusters)")
    parser.add_argument("--intrinsic-dim", type=int, default=12, help="dimensions of each cluster's own subspace")
    parser.add_argument("--noise", type=float, default=0.3, help="isotropic noise, relative to the cluster spread")
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--hot-clusters", default="0,1,2",
                        help="clusters the hot queries come from; the insert batch comes from one of them")
    parser.add_argument("--hot-share", type=float, default=0.7, help="share of the queries from the hot clusters")
    parser.add_argument("--inserts", type=int, default=4000,
                        help="vectors of the insert phase, all from one hot cluster: the one whose queries, searched "
                             "alone, cause the most distance computations on the server that was busiest in the "
                             "search of all queries on the base graph; of that cluster's pool (4 x this many "
                             "vectors), the vectors whose nearest existing vector that server owns, then the "
                             "nearest to its part")
    parser.add_argument("--stream-repeats", type=int, default=48,
                        help="passes over the queries in the controller phase's stream")
    parser.add_argument("--search-l", type=int, default=200, help="search list size of every search command")
    parser.add_argument("--repair", default=REPAIR, help="the controller phase's repair token (docs/CONTROLLER.md)")
    parser.add_argument("--pq-m", type=int, default=0, help="PQ sub-quantizers (default dim / 4)")
    parser.add_argument("--min-recall", type=float, default=90.0, help="fail below this recall@10 (percent)")
    parser.add_argument("--keep", action="store_true", help="leave the cluster running")
    args = parser.parse_args(argv)
    work = args.workdir.resolve()
    if work.exists():
        parser.error(f"{work} exists; give a new directory")
    if args.dim % (args.pq_m or max(1, args.dim // 4)):
        parser.error("--pq-m must divide --dim")
    try:
        hot = hot_clusters(args)
    except ValueError:
        parser.error("--hot-clusters is a comma-separated list of cluster ids")
    if not hot or len(set(hot)) != len(hot) or args.inserts < 1:
        parser.error("--hot-clusters needs distinct cluster ids and --inserts a positive count")
    work.mkdir(parents=True)
    started = time.monotonic()
    hot_counts = prepare(args, work)
    stream_rows = args.stream_repeats * args.queries
    cluster = LocalCluster(name=args.name, image=args.image, coordinator_data=work / "runtime",
                           server_data=[work / "views" / f"server-{rank}" for rank in range(args.servers)],
                           servers=args.servers, bin_dir=args.bin_dir, storage="pq", threads=args.server_threads,
                           coordinator_queries=stream_rows, control=work / "control", work=work / "work",
                           query_override=work / "stream" / "query.npy",
                           groundtruth_override=work / "stream" / "groundtruth.npy", volumes=args.volume)
    (work / "work" / "coordinator" / "controller").mkdir(parents=True)
    log(f"starting {args.servers} servers and the coordinator")
    try:
        port = cluster.up()
        results = workload(args, Session(port, work, args.servers, args.search_l), Steps(args, work), stream_rows,
                           hot_counts)
    except BaseException:
        (work / "logs").mkdir(exist_ok=True)
        for container in cluster.names:
            (work / "logs" / f"{container}.log").write_text(cluster.logs(container), encoding="utf-8")
        if not args.keep:
            cluster.down()
        raise
    for container in cluster.names:
        (work / "logs" / f"{container}.log").write_text(cluster.logs(container), encoding="utf-8")
    if not args.keep:
        cluster.down()
    results["seconds"] = round(time.monotonic() - started, 1)
    results["settings"] = {"seed": args.seed, "graph_threads": args.graph_threads,
                           "workload": {"source": workload_argv(args, hot[0], 1),
                                        **{f"pool-{c}": workload_argv(args, c, pool_rows(args)) for c in hot}},
                           "inserts": args.inserts, "pool_rows": pool_rows(args),
                           "search_l": args.search_l, "stream_rows": stream_rows,
                           "controller_start": results.pop("controller_start")}
    results["identity"] = identity(work, args)
    report(results)
    low = [label for label in CHECKPOINTS
           if not (results[label]["recall_at_10_percent"] or 0) >= args.min_recall
           or not results[label]["membership_consistent_and_vectors_exact"]]
    results["status"] = "FAIL" if low else "PASS"
    (work / "DEMO_RESULT.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if low:
        log(f"recall below {args.min_recall} % or a membership mismatch at: {', '.join(low)}")
        return 1
    log(f"done in {results['seconds']} s; replies and logs in {work}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--data-step"]:
        sys.exit(data_step(sys.argv[2:]))
    sys.exit(main())
