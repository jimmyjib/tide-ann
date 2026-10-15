#!/usr/bin/env python3
"""Run tide-ann end to end on synthetic data in a local 4-server Docker cluster.

Builds the dataset, Vamana graph, METIS partitions and PQ files, starts the cluster, inserts a batch next to
the busiest server, runs one controller phase and prints recall, hops and per-server load after each step.

    docker build -f docker/Dockerfile -t tide-ann:latest .
    docker build --platform linux/amd64 -f docker/Dockerfile.tools -t tide-ann-tools:latest .
    python3 examples/demo.py --workdir /tmp/tide-demo
"""
from __future__ import annotations

import argparse
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

# Controller settings for a short controller phase on small data (keys: docs/controller.md).
REPAIR = ("refine:0.01,balance:0.004,tau:0.2,eps:0.03,skew:1.02,cooldown:1,cum:1.0,slack:0.10,"
          "hop_slack:0.05,ema:0.5,converge:3,guard:0.10,recall_guard:off,balance_target:underloaded")
WINDOW = ["window_sec=0.25", "cooldown_sec=0.25", "min_window_queries=8192"]
# queries at the end of the stream that only check the last batch of moves
VERIFY_TAIL_ROWS = 24576
SEARCH = ["--batch-size", "32", "--dispatch-window", "256", "--no-route-trace", "--no-merge"]
GROUNDTRUTH = ["--metric", "l2", "--k", "10"]
CHECKPOINTS = ("base", "after_insert", "after_controller")
# insert pool size per hot cluster, in multiples of --inserts
POOL_FACTOR = 4


def log(message: str) -> None:
    print(message, flush=True)


class Steps:
    """Run the data steps on the host or in the tools image."""

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
        """Run python with argv in the work dir ({repo} = the repo path) and return its output."""
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


def server_views(work: Path, servers: int) -> list[Path]:
    """One directory per server with only the files it loads in PQ mode."""
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


def hot_clusters(args: argparse.Namespace) -> list[int]:
    return [int(item) for item in args.hot_clusters.split(",") if item.strip()]


def pool_rows(args: argparse.Namespace) -> int:
    return POOL_FACTOR * args.inserts


def workload_argv(args: argparse.Namespace, cluster: int, inserts: int) -> list[str]:
    """make_synthetic options of the demo data, with `inserts` insert vectors from one cluster."""
    return ["--seed", str(args.seed), "--n-base", str(args.n_base), "--dim", str(args.dim),
            "--center-scale", f"{args.center_scale:g}", "--noise", f"{args.noise:g}",
            "--intrinsic-dim", str(args.intrinsic_dim), "--n-query", str(args.queries),
            "--hot-clusters", args.hot_clusters, "--hot-share", f"{args.hot_share:g}",
            "--n-insert", str(inserts), "--insert-cluster", str(cluster), "--insert-share", "1"]


def prepare(args: argparse.Namespace, work: Path) -> dict[int, int]:
    """Run the data steps. Returns the number of queries of each hot cluster."""
    steps = Steps(args, work)
    hot = hot_clusters(args)
    log(f"data preparation (seed {args.seed})")
    log(f"  data: {args.n_base:,} x {args.dim} base vectors, center scale {args.center_scale:g}, "
        f"{args.queries:,} queries ({args.hot_share:.0%} from clusters {args.hot_clusters}), "
        f"insert pool {pool_rows(args):,} per hot cluster")
    # source/: base and queries (its one insert vector is not used)
    steps.python("synthetic", ["-m", "tools.prepare.make_synthetic", "--output", "source",
                               *workload_argv(args, hot[0], 1)])
    for cluster in hot:
        pool = work / f"pool-{cluster}"
        steps.python(f"pool-{cluster}", ["-m", "tools.prepare.make_synthetic", "--output", pool.name,
                                         *workload_argv(args, cluster, pool_rows(args))])
        # Same seed: the pool's base and queries are those of source/ (they have their own random streams).
        for name in ("base.npy", "query.npy", "base_plus_insert.npy"):
            (pool / name).unlink()  # the pool keeps only its inserts
    steps.python("vamana", ["{repo}/tools/prepare/build_vamana.py", "--input", "source/base.npy", "--output", "index",
                            "--metric", "l2", "--threads", str(args.graph_threads)], graph=True)
    steps.python("runtime", ["-m", "tools.prepare.prepare_runtime", "--dataset-root", ".",
                             "--parts", str(args.servers), "--stream-repeats", str(args.stream_repeats)])
    (work / "pq").mkdir()
    steps.build_pq(args.pq_m or args.dim // 4)
    steps.python("partition-vectors", ["-m", "tools.prepare.write_partition_vectors", "--runtime", "runtime",
                                       "--pq-dir", "pq"])
    control = work / "control"
    for source, name in (("source/query.npy", "query.npy"), ("gt/base/groundtruth.npy", "groundtruth.npy")):
        link(work / source, control / name)
    output = steps.python("hot-queries", ["-m", "tools.prepare.split_queries_by_cluster", "--source", "source",
                                          "--hot-clusters", args.hot_clusters, "--output", "control"])
    counts = json.loads(output.strip().splitlines()[-1])
    server_views(work, args.servers)
    return {cluster: int(counts[str(cluster)]) for cluster in hot}


class Session:
    """Admin commands of the demo. Every reply is saved under <workdir>/replies."""

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
        return {"recall": search.get("recall"), "avg_hops": search.get("avg_hops"),
                "distance_work_by_server": work,
                "imbalance_max_over_mean": max(work) / (sum(work) / len(work)) if sum(work) > 0 else None,
                "work_share_by_server": [w / sum(work) for w in work] if sum(work) > 0 else None,
                "owned_nodes_by_server": check.get("coordinator_owned_by_rank"),
                "membership_ok": exact,
                "membership_file": f"work/coordinator/membership-{name}.npy",
                "memory_bytes_by_server": [{key: row["storage"][key] for key in
                                            ("vector_bytes", "code_bytes", "pq_table_bytes")} for row in after]}


# Insert from the hot cluster whose queries cause the most distance computations on the busiest server,
# taking the pool rows nearest to that server's nodes.
def choose_insert_cluster(args: argparse.Namespace, session: Session, last: dict[str, Any],
                          hot_counts: dict[int, int]) -> dict[str, Any]:
    work = last["distance_work_by_server"]
    busiest = max(range(len(work)), key=lambda rank: (work[rank], -rank))
    share = work[busiest] / sum(work) if sum(work) > 0 else None
    log(f"insert phase: busiest server {busiest} ({100.0 * (share or 0.0):.0f}% of distance computations)")
    by_cluster: dict[int, list[int]] = {}
    for cluster in hot_clusters(args):
        _, by_cluster[cluster], _ = session.search_work(
            f"insert-probe-cluster-{cluster}",
            ["recall", "--queries", str(hot_counts[cluster]), "--query-file", f"/control/query-hot-{cluster}.npy",
             "--no-groundtruth", *SEARCH, "--search-l", str(args.search_l)])
        log(f"  cluster {cluster}: {hot_counts[cluster]:,} queries, {by_cluster[cluster][busiest]:,} distance "
            f"computations on server {busiest} "
            f"({100.0 * by_cluster[cluster][busiest] / max(1, work[busiest]):.0f}%)")
    order = hot_clusters(args)
    chosen = max(order, key=lambda c: (by_cluster[c][busiest], -order.index(c)))
    log(f"  inserting from cluster {chosen}")
    return {"busiest_server": busiest, "cluster": chosen, "membership_file": last["membership_file"]}


def insert_phase(args: argparse.Namespace, session: Session, steps: Steps, choice: dict[str, Any]) -> dict[str, Any]:
    """Write the insert batch and its ground truth, insert it and drain."""
    cluster, server = choice["cluster"], choice["busiest_server"]
    batch, combined = "control/insert.npy", "gt-input/base-plus-inserts.npy"
    output = steps.python("batch", [
        "-m", "tools.prepare.select_insert_batch", "--points", "source/base.npy",
        "--owners", choice["membership_file"], "--server", str(server), "--servers", str(args.servers),
        "--pool", f"pool-{cluster}/insert.npy", "--first-row", "0", "--pool-rows", str(pool_rows(args)),
        "--rows", str(args.inserts), "--output", batch, "--base-plus-inserts", combined])
    chosen = json.loads(output.strip().splitlines()[-1])
    steps.python("gt", ["-m", "tools.prepare.generate_groundtruth", "--base", combined, "--query",
                        "source/query.npy", "--output", "gt/after-insert/groundtruth.npy", *GROUNDTRUTH])
    link(steps.work / "gt" / "after-insert" / "groundtruth.npy",
         steps.work / "control" / "groundtruth-after-insert.npy")
    log(f"insert phase: {args.inserts:,} vectors from cluster {cluster} "
        f"({chosen['nearest_on_server']:,} nearest to server {server}), then drain")
    insert = session.run("insert", ["insert", str(args.inserts), "--file", "/control/insert.npy",
                                    "--start-id", "0", "--batch-size", "32", "--dispatch-window", "256"],
                         timeout=3600)
    count = insert.get("insert", {}).get("count")
    if count != args.inserts:
        raise ClusterError(f"the insert phase inserted {count} of {args.inserts} vectors")
    session.run("drain", ["drain"], timeout=600)
    return {"count": count}


def controller_phase(args: argparse.Namespace, session: Session, start: list[str], stream_rows: int) -> dict[str, Any]:
    log("controller phase: controller-start, the query stream, controller-stop")
    log(f"  settings: {' '.join(start[2:])}")
    log(f"  query stream: {stream_rows:,} queries ({args.stream_repeats} shuffled passes)")
    session.run("controller-start", start, timeout=600)
    stream = ["recall", "--queries", str(stream_rows), "--no-groundtruth", "--batch-size", "128",
              "--dispatch-window", "256", "--no-route-trace", "--no-merge", "--search-l", str(args.search_l)]
    session.run("stream", stream, timeout=3600)
    status = session.run("controller-status", ["controller-status"])
    stop = session.run("controller-stop", ["controller-stop"], ok=False)
    rounds = 0
    # A batch committed near the end of the stream may still be pending. Search more until it resolves.
    while stop.get("status") != "ok" and stop.get("unresolved_action") is True and rounds < 4:
        rounds += 1
        session.run(f"resolve-{rounds}", ["recall", "--queries", str(min(16384, stream_rows)),
                                          "--no-groundtruth", *stream[4:]], timeout=3600)
        status = session.run(f"controller-status-{rounds}", ["controller-status"])
        stop = session.run(f"controller-stop-{rounds}", ["controller-stop"], ok=False)
    if stop.get("status") != "ok":
        raise ClusterError(f"controller-stop failed: {json.dumps(stop)[:1000]}")
    repair = status.get("repair", {}) if isinstance(status.get("repair"), dict) else {}
    # commits includes batches that were later reverted
    result = {key: status.get(key) for key in ("commits", "reverts", "actions_committed", "moved_bytes",
                                               "total_bytes", "verify_pending")}
    result["windows"] = repair.get("windows")
    pending = 1 if status.get("verify_pending") is True else 0
    result["kept"] = (result.get("commits") or 0) - (result.get("reverts") or 0) - pending
    return result


def workload(args: argparse.Namespace, session: Session, steps: Steps, stream_rows: int,
             hot_counts: dict[int, int]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    log(f"search on the base graph (L = {args.search_l})")
    results["base"] = session.measured_recall("recall-base", args.queries, "/control/groundtruth.npy")
    tail = min(VERIFY_TAIL_ROWS, stream_rows // 4)
    start = ["controller-start", "/work/controller", *WINDOW, f"proposal_query_limit={stream_rows - tail}",
             f"repair={args.repair}"]
    choice = choose_insert_cluster(args, session, results["base"], hot_counts)
    results["insert"] = {**choice, **insert_phase(args, session, steps, choice)}
    groundtruth = "/control/groundtruth-after-insert.npy"
    results["after_insert"] = session.measured_recall("recall-after-insert", args.queries, groundtruth)
    results["controller"] = controller_phase(args, session, start, stream_rows)
    results["after_controller"] = session.measured_recall("recall-after-controller", args.queries, groundtruth)
    return results


def plural(count: Any, word: str, many: str | None = None) -> str:
    shown = f"{count:,}" if isinstance(count, int) else str(count)
    return f"{shown} {word if count == 1 else many or word + 's'}"


def change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    return f"{before:.3f} -> {after:.3f} ({100.0 * (after - before) / before:+.1f}%)"


def report(results: dict[str, Any]) -> None:
    labels = {"base": "base graph", "after_insert": "after the insert phase",
              "after_controller": "after the controller phase"}
    log("")
    log(f"{'':26s} {'recall@10':>10s} {'hops':>6s} {'imbalance':>10s}   {'work per server (%)':20s} "
        "nodes owned per server")
    for key in CHECKPOINTS:
        row = results[key]
        imbalance = row["imbalance_max_over_mean"]
        shares = row["work_share_by_server"]
        shares = "[" + ", ".join(f"{100.0 * x:.0f}" for x in shares) + "]" if shares else "n/a"
        log(f"{labels[key]:26s} {row['recall']:9.2f}% {row['avg_hops']:6.2f} "
            f"{imbalance if imbalance is not None else float('nan'):10.3f}   {shares:20s} "
            f"{row['owned_nodes_by_server']}")
    imbalance = {key: results[key]["imbalance_max_over_mean"] for key in CHECKPOINTS}
    log("")
    insert, c = results["insert"], results["controller"]
    server = insert["busiest_server"]
    gained = (results["after_insert"]["owned_nodes_by_server"][server]
              - results["base"]["owned_nodes_by_server"][server])
    log(f"insert phase:     imbalance {change(imbalance['base'], imbalance['after_insert'])}, "
        f"{insert['count']:,} vectors from cluster {insert['cluster']}, {gained:,} joined server {server}")
    pending = ", 1 pending" if c.get("verify_pending") is True else ""
    log(f"controller phase: imbalance {change(imbalance['after_insert'], imbalance['after_controller'])}, "
        f"{plural(c['windows'], 'window')}, {plural(c['commits'], 'batch', 'batches')} committed ({c['kept']} kept, "
        f"{c['reverts']} reverted{pending}), {plural(c['actions_committed'], 'node move')}, "
        f"{c['moved_bytes']:,} of {c['total_bytes']:,} bytes moved")
    memory = results["after_controller"]["memory_bytes_by_server"]
    log("memory per server (vectors + PQ codes + PQ tables, MB): "
        + ", ".join(f"{(m['vector_bytes'] + m['code_bytes'] + m['pq_table_bytes']) / 1e6:.1f}" for m in memory))
    log("imbalance = max / mean distance computations per server")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the tide-ann demo on a local Docker cluster.")
    parser.add_argument("--workdir", type=Path, required=True, help="new directory for data, logs and replies")
    parser.add_argument("--name", default="tide-demo", help="prefix of the Docker containers and network")
    parser.add_argument("--image", default="tide-ann:latest", help="image with the binaries")
    parser.add_argument("--bin-dir", default="/usr/local/bin", help="directory of the binaries in --image")
    parser.add_argument("--volume", action="append", default=[],
                        help="extra docker -v spec for the cluster containers")
    parser.add_argument("--tools-image", default="tide-ann-tools:latest")
    parser.add_argument("--tools-platform", default="linux/amd64", help="platform of the tools / graph images")
    parser.add_argument("--tools-python", help="local Python for the data steps instead of the tools image")
    parser.add_argument("--graph-image", help="image for the Vamana step only (needs diskannpy)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the synthetic data")
    parser.add_argument("--graph-threads", type=int, default=1,
                        help="Vamana build threads (1: same graph for the same seed and package versions)")
    parser.add_argument("--servers", type=int, default=4)
    parser.add_argument("--server-threads", type=int, default=4)
    parser.add_argument("--n-base", type=int, default=20000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--center-scale", type=float, default=0.3,
                        help="spread of the cluster centers (larger = more separated)")
    parser.add_argument("--intrinsic-dim", type=int, default=12, help="dimensions of each cluster's own subspace")
    parser.add_argument("--noise", type=float, default=0.3, help="isotropic noise, relative to the cluster spread")
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--hot-clusters", default="0,1,2",
                        help="comma-separated ids of the hot clusters")
    parser.add_argument("--hot-share", type=float, default=0.7, help="share of the queries from the hot clusters")
    parser.add_argument("--inserts", type=int, default=4000,
                        help="vectors to insert from one hot cluster")
    parser.add_argument("--stream-repeats", type=int, default=48,
                        help="passes over the queries in the controller phase")
    parser.add_argument("--search-l", type=int, default=200, help="search list size of every search command")
    parser.add_argument("--repair", default=REPAIR, help="repair= settings for controller-start (docs/controller.md)")
    parser.add_argument("--pq-m", type=int, default=0, help="PQ sub-quantizers (default dim / 4)")
    parser.add_argument("--min-recall", type=float, default=90.0, help="fail below this recall@10 (percent)")
    parser.add_argument("--keep", action="store_true", help="leave the cluster running")
    args = parser.parse_args(argv)
    work = args.workdir.resolve()
    if work.exists():
        parser.error(f"--workdir {work} already exists")
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
    seconds = round(time.monotonic() - started, 1)
    report(results)
    low = [label for label in CHECKPOINTS if not (results[label]["recall"] or 0) >= args.min_recall]
    mismatch = [label for label in CHECKPOINTS if not results[label]["membership_ok"]]
    if low:
        log(f"error: recall@10 below {args.min_recall}% at: {', '.join(low)}")
    if mismatch:
        log(f"error: a server's nodes or vectors differ from the coordinator's membership at: {', '.join(mismatch)} "
            f"(membership-check replies in {work / 'replies'})")
    if low or mismatch:
        return 1
    log(f"done in {seconds} s, replies and logs in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
