"""The local cluster's docker argv (nothing is started) and the reply parser."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.local_cluster import ClusterError, LocalCluster, parse_reply, resource_names


class LocalClusterTest(unittest.TestCase):
    def cluster(self, root: Path, **kwargs) -> LocalCluster:
        return LocalCluster(name="t", image="img", coordinator_data=root / "runtime",
                            server_data=[root / f"view-{rank}" for rank in range(4)], work=root / "work", **kwargs)

    def test_p2p_on_every_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cluster = self.cluster(Path(tmp))
            peers = "tcp://server0:28800,tcp://server1:28800,tcp://server2:28800,tcp://server3:28800"
            for rank in range(4):
                command = cluster.server_command(rank)
                self.assertEqual(command[command.index("--peers") + 1], peers)
                self.assertEqual(command[command.index("--rank") + 1], str(rank))
                self.assertEqual(command[command.index("--num-servers") + 1], "4")
                self.assertEqual(command[-1], "--pq")
                self.assertIn(f"{Path(tmp).resolve() / f'view-{rank}'}:/data:ro", command)
                self.assertIn("tide.run=t", command)
            coordinator = cluster.coordinator_command()
            self.assertIn("--p2p", coordinator)
            self.assertEqual(coordinator[coordinator.index("--servers") + 1], peers)
            # The admin port is published on an ephemeral port of the loopback interface only.
            self.assertEqual(coordinator[coordinator.index("-p") + 1], "127.0.0.1::29699")

    def test_names_storage_and_counts(self) -> None:
        network, names = resource_names("demo", 8)
        self.assertEqual(network, "demo-net")
        self.assertEqual(names[0], "demo-server-0")
        self.assertEqual(names[-1], "demo-coordinator")
        self.assertEqual(len(names), 9)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = LocalCluster(name="t", image="img", coordinator_data=root, server_data=[root], storage="full")
            self.assertEqual(full.server_command(3)[-1], "--full-vectors")
            with self.assertRaises(ClusterError):
                LocalCluster(name="t", image="img", coordinator_data=root, server_data=[root, root])
            with self.assertRaises(ClusterError):
                LocalCluster(name="t", image="img", coordinator_data=root, server_data=[root], servers=1)

    def test_parse_reply(self) -> None:
        self.assertEqual(parse_reply('log line\n===JSON_RESULT===\n{"status": "ok", "n": 2}\n===JSON_END===\n'),
                         {"status": "ok", "n": 2})
        with self.assertRaises(ClusterError):
            parse_reply("[ERROR] no result")


if __name__ == "__main__":
    unittest.main()
