#!/usr/bin/env python3
"""Start, drive and stop a local K-server cluster in Docker.

One Docker network, K index servers (``tide-server``, P2P state passing: every server gets
``--peers`` with all K addresses) and one coordinator (``tide-coordinator --p2p``), whose admin port
29699 is published on an ephemeral port of 127.0.0.1. Every container and the network carry the label
``tide.run=<name>``, so ``down`` removes exactly what ``up`` created.

Data layout (host directories, mounted read-only):

* each server's data directory: ``membership.npy`` and ``partition_<rank>/{adj,local_to_global.npy}``,
  plus ``pq_pivots.bin``, ``pq_codes.bin`` and ``partition_<rank>/vectors.npy`` under hybrid PQ
  storage (the default) or ``base.npy`` under ``--full-vectors``; one directory for every server or one
  per server;
* the coordinator's data directory: the whole runtime (``base.npy``, ``membership.npy``,
  ``query.npy``, ``groundtruth.npy``, the head ids);
* an optional control directory (``/control``: insert files, extra query and ground-truth files) and a
  writable work directory (``/work``: the controller journal).

Use it as a module (``LocalCluster``) or from the command line::

    python3 -m tools.local_cluster up --name demo --image tide-ann:latest \\
        --coordinator-data runtime --server-data views/server-0 ... --work work
    python3 -m tools.local_cluster send --port 49153 -- recall --queries 1000
    python3 -m tools.local_cluster down --name demo --servers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

SERVER_BINARY = "tide-server"
COORDINATOR_BINARY = "tide-coordinator"
SERVER_PORT = 28800
ADMIN_PORT = 29699
LABEL = "tide.run"
RESULT_BEGIN, RESULT_END = "===JSON_RESULT===", "===JSON_END==="


class ClusterError(RuntimeError):
    pass


def docker(args: Sequence[str], *, timeout: float = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["docker", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout, check=False)
    if check and result.returncode != 0:
        raise ClusterError(f"docker {' '.join(args[:3])} ... failed ({result.returncode}): {result.stdout[-2000:]}")
    return result


def resource_names(name: str, servers: int) -> tuple[str, list[str]]:
    """The network and the containers (servers in rank order, coordinator last) of a cluster."""
    return f"{name}-net", [f"{name}-server-{rank}" for rank in range(servers)] + [f"{name}-coordinator"]


def user_args() -> list[str]:
    """Run as the calling user where that is defined, so files written into mounts stay the user's."""
    if hasattr(os, "getuid"):
        return ["--user", f"{os.getuid()}:{os.getgid()}"]
    return []


@dataclass
class LocalCluster:
    name: str
    image: str
    coordinator_data: Path
    server_data: list[Path]
    servers: int = 4
    bin_dir: str = "/usr/local/bin"
    storage: str = "pq"                     # "pq" (hybrid PQ own-only) or "full" (full replication)
    threads: int = 4
    coordinator_queries: int = 1000
    batch_size: int = 32
    control: Path | None = None
    work: Path | None = None
    query_override: Path | None = None      # mounted over the coordinator's query.npy
    groundtruth_override: Path | None = None
    volumes: list[str] = field(default_factory=list)  # extra "-v" specs, e.g. a volume holding the binaries
    server_args: list[str] = field(default_factory=lambda: ["--insert-batch-size", "128",
                                                            "--backlink-batch-size", "128",
                                                            "--broadcast-batch-size", "128"])
    readiness_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not 2 <= self.servers <= 64:
            raise ClusterError("a cluster has 2..64 servers")
        if self.storage not in ("pq", "full"):
            raise ClusterError("storage is pq or full")
        if len(self.server_data) == 1:
            self.server_data = self.server_data * self.servers
        if len(self.server_data) != self.servers:
            raise ClusterError("give one server data directory, or one per server")
        self.network, self.names = resource_names(self.name, self.servers)

    def peers(self) -> str:
        return ",".join(f"tcp://server{rank}:{SERVER_PORT}" for rank in range(self.servers))

    def _labels(self) -> list[str]:
        return ["--label", f"{LABEL}={self.name}"]

    def _extra_volumes(self) -> list[str]:
        return [item for spec in self.volumes for item in ("-v", spec)]

    def server_command(self, rank: int) -> list[str]:
        mounts = ["-v", f"{Path(self.server_data[rank]).resolve()}:/data:ro", *self._extra_volumes()]
        if self.work is not None:
            # Each server gets its own empty writable directory; migration data travels in messages.
            scratch = Path(self.work).resolve() / f"server-{rank}"
            scratch.mkdir(parents=True, exist_ok=True)
            mounts += ["-v", f"{scratch}:/work"]
        return ["run", "-d", "--name", self.names[rank], "--network", self.network, "--network-alias",
                f"server{rank}", *self._labels(), *user_args(), *mounts, self.image,
                f"{self.bin_dir}/{SERVER_BINARY}", "--data-dir", "/data", "--rank", str(rank),
                "--port", str(SERVER_PORT), "--peers", self.peers(), "--num-servers", str(self.servers),
                "--threads", str(self.threads), *self.server_args,
                "--pq" if self.storage == "pq" else "--full-vectors"]

    def coordinator_command(self) -> list[str]:
        mounts = ["-v", f"{Path(self.coordinator_data).resolve()}:/data:ro", *self._extra_volumes()]
        if self.query_override is not None:
            mounts += ["-v", f"{Path(self.query_override).resolve()}:/data/query.npy:ro"]
        if self.groundtruth_override is not None:
            mounts += ["-v", f"{Path(self.groundtruth_override).resolve()}:/data/groundtruth.npy:ro"]
        if self.control is not None:
            mounts += ["-v", f"{Path(self.control).resolve()}:/control:ro"]
        workdir: list[str] = []
        if self.work is not None:
            coordinator_work = Path(self.work).resolve() / "coordinator"
            coordinator_work.mkdir(parents=True, exist_ok=True)
            mounts += ["-v", f"{coordinator_work}:/work"]
            workdir = ["-w", "/work"]
        return ["run", "-d", "--name", self.names[-1], "--network", self.network, "--network-alias", "coordinator",
                "-p", f"127.0.0.1::{ADMIN_PORT}", *self._labels(), *user_args(), *workdir, *mounts, self.image,
                f"{self.bin_dir}/{COORDINATOR_BINARY}", "--servers", self.peers(), "--data-dir", "/data",
                "--queries", str(self.coordinator_queries), "--p2p", "--batch-size", str(self.batch_size),
                "--L", "50", "--K", "10", "--W", "8"]

    def up(self) -> int:
        """Create the network and the containers, wait until every process is ready; returns the admin port."""
        docker(["network", "create", *self._labels(), self.network])
        for rank in range(self.servers):
            docker(self.server_command(rank))
        docker(self.coordinator_command())
        deadline = time.monotonic() + self.readiness_seconds
        while time.monotonic() < deadline:
            for container in self.names:
                state = docker(["inspect", "--format", "{{.State.Running}}", container], check=False).stdout.strip()
                if state != "true":
                    raise ClusterError(f"{container} exited during start-up:\n{self.logs(container)[-3000:]}")
            logs = [self.logs(container) for container in self.names]
            if (all(f"Listening on tcp://*:{SERVER_PORT}" in text for text in logs[:-1])
                    and "Entering daemon mode." in logs[-1]):
                port = docker(["port", self.names[-1], f"{ADMIN_PORT}/tcp"]).stdout.strip().splitlines()[0]
                return int(port.rsplit(":", 1)[-1])
            time.sleep(0.25)
        raise ClusterError(f"the cluster was not ready within {self.readiness_seconds:g} s")

    def logs(self, container: str) -> str:
        return docker(["logs", container], check=False).stdout

    def down(self) -> None:
        down(self.name, self.servers)


def down(name: str, servers: int) -> None:
    """Remove the cluster's containers and network: exact names, and only when they carry its label."""
    network, names = resource_names(name, servers)
    for container in names:
        found = docker(["inspect", "--format", "{{index .Config.Labels \"" + LABEL + "\"}}", container], check=False)
        if found.returncode == 0:
            if found.stdout.strip() != name:
                raise ClusterError(f"{container} exists but is not labelled {LABEL}={name}; not removed")
            docker(["rm", "--force", container])
    found = docker(["network", "inspect", "--format", "{{index .Labels \"" + LABEL + "\"}}", network], check=False)
    if found.returncode == 0:
        if found.stdout.strip() != name:
            raise ClusterError(f"network {network} is not labelled {LABEL}={name}; not removed")
        docker(["network", "rm", network])


def parse_reply(text: str) -> dict[str, Any]:
    """The JSON result of a coordinator reply (between its markers)."""
    if RESULT_BEGIN not in text:
        raise ClusterError(f"no JSON result in the reply: {text[-2000:]}")
    body = text.split(RESULT_BEGIN, 1)[1]
    body = body.split(RESULT_END, 1)[0] if RESULT_END in body else body
    result, _ = json.JSONDecoder().raw_decode(body.strip())
    if not isinstance(result, dict):
        raise ClusterError("the JSON result is not an object")
    return result


def send(port: int, command: Sequence[str] | str, *, timeout_seconds: float = 600.0,
         host: str = "127.0.0.1") -> tuple[dict[str, Any], str]:
    """Send one admin command (a REQ/REP round trip) and return (JSON result, raw reply)."""
    import zmq

    text = command if isinstance(command, str) else " ".join(command)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_seconds * 1000))
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(f"tcp://{host}:{port}")
        socket.send_string(text)
        raw = socket.recv_string()
    except zmq.Again as error:
        raise ClusterError(f"no reply to {text.split()[0]!r} within {timeout_seconds:g} s") from error
    finally:
        socket.close()
        context.term()
    return parse_reply(raw), raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    up = sub.add_parser("up", help="start a cluster and print its admin port")
    up.add_argument("--name", required=True)
    up.add_argument("--image", required=True)
    up.add_argument("--bin-dir", default="/usr/local/bin")
    up.add_argument("--servers", type=int, default=4)
    up.add_argument("--threads", type=int, default=4)
    up.add_argument("--storage", choices=("pq", "full"), default="pq")
    up.add_argument("--coordinator-data", type=Path, required=True)
    up.add_argument("--server-data", type=Path, action="append", required=True,
                    help="once for every server, or once per server in rank order")
    up.add_argument("--coordinator-queries", type=int, default=1000)
    up.add_argument("--control", type=Path)
    up.add_argument("--work", type=Path)
    up.add_argument("--volume", action="append", default=[], help="extra docker -v spec")
    stop = sub.add_parser("down", help="remove a cluster's containers and network")
    stop.add_argument("--name", required=True)
    stop.add_argument("--servers", type=int, default=4)
    command = sub.add_parser("send", help="send one admin command and print the reply")
    command.add_argument("--port", type=int, required=True)
    command.add_argument("--timeout", type=float, default=600.0)
    command.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "up":
        cluster = LocalCluster(name=args.name, image=args.image, coordinator_data=args.coordinator_data,
                               server_data=args.server_data, servers=args.servers, bin_dir=args.bin_dir,
                               storage=args.storage, threads=args.threads,
                               coordinator_queries=args.coordinator_queries, control=args.control, work=args.work,
                               volumes=args.volume)
        try:
            print(cluster.up())
        except BaseException:
            cluster.down()
            raise
    elif args.action == "down":
        down(args.name, args.servers)
    else:
        words = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not words:
            parser.error("send needs a command")
        _, raw = send(args.port, words, timeout_seconds=args.timeout)
        print(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
