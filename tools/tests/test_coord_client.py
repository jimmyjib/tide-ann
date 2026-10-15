"""Unit tests of the admin client coord_client.py: the command forms it accepts and its exit codes.

The client runs as a subprocess against a stand-in ``zmq`` module that records the command it was sent
and answers with a fixed reply, so no coordinator and no pyzmq are needed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLIENT = Path(__file__).resolve().parents[1] / "coord_client.py"

FAKE_ZMQ = '''
import os
REQ = 1; RCVTIMEO = 2; SNDTIMEO = 3; LINGER = 4
class Again(Exception): pass
class ZMQError(Exception): pass
class Socket:
    def setsockopt(self, *a): pass
    def connect(self, *a): pass
    def send_string(self, value):
        with open(os.environ["FAKE_SENT"], "w", encoding="utf-8") as out:
            out.write(value)
    def recv_string(self):
        return os.environ["FAKE_REPLY"]
    def close(self): pass
class Context:
    def socket(self, *a): return Socket()
    def term(self): pass
'''

OK_JSON = "===JSON_RESULT===\n{\"status\": \"ok\", \"action\": \"recall\"}\n===JSON_END==="
ERROR_COMPACT = "===JSON_RESULT===\n{\"status\":\"error\",\"action\":\"controller-stop\"}\n===JSON_END==="
ERROR_SPACED = "===JSON_RESULT===\n{\"status\": \"error\", \"action\": \"search\"}\n===JSON_END==="


class CoordClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "zmq.py").write_text(FAKE_ZMQ, encoding="utf-8")
        self.sent = root / "sent.txt"
        self.root = root

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_client(self, *argv: str, reply: str = OK_JSON) -> tuple[int, str]:
        env = dict(os.environ, PYTHONPATH=str(self.root), FAKE_SENT=str(self.sent), FAKE_REPLY=reply)
        if self.sent.exists():
            self.sent.unlink()
        result = subprocess.run([sys.executable, str(CLIENT), *argv], capture_output=True, text=True, env=env,
                                timeout=60)
        return result.returncode, self.sent.read_text(encoding="utf-8") if self.sent.exists() else ""

    def test_options_after_the_command_are_sent_without_a_separator(self) -> None:
        code, sent = self.run_client("--endpoint", "tcp://127.0.0.1:1", "recall", "--queries", "1000")
        self.assertEqual((code, sent), (0, "recall --queries 1000"))
        code, sent = self.run_client("--endpoint", "tcp://127.0.0.1:1", "insert", "4000", "--file", "/c/i.npy",
                                     "--start-id", "0")
        self.assertEqual((code, sent), (0, "insert 4000 --file /c/i.npy --start-id 0"))

    def test_a_separator_before_the_command_is_dropped(self) -> None:
        code, sent = self.run_client("--endpoint", "tcp://127.0.0.1:1", "--timeout", "1000", "--", "recall",
                                     "--queries", "1000", "--debug")
        self.assertEqual((code, sent), (0, "recall --queries 1000 --debug"))

    def test_a_command_is_required(self) -> None:
        for argv in ((), ("--endpoint", "tcp://127.0.0.1:1"), ("--",)):
            code, sent = self.run_client(*argv)
            self.assertEqual((code, sent), (2, ""), argv)

    def test_error_replies_exit_one(self) -> None:
        for reply in (ERROR_COMPACT, ERROR_SPACED, "Error: cannot read the ground truth: Cannot open: /x.npy",
                      "Unknown command: bogus\nAvailable commands: recall, insert"):
            code, _ = self.run_client("status", reply=reply)
            self.assertEqual(code, 1, reply)

    def test_ok_replies_exit_zero(self) -> None:
        for reply in (OK_JSON, "[ERROR] a server log line\n" + OK_JSON, "pong"):
            code, _ = self.run_client("status", reply=reply)
            self.assertEqual(code, 0, reply)

    def test_exit_zero_on_reply_reads_every_reply_as_delivered(self) -> None:
        for reply in (ERROR_COMPACT, "Error: refused"):
            code, sent = self.run_client("--exit-zero-on-reply", "--", "controller-stop", reply=reply)
            self.assertEqual((code, sent), (0, "controller-stop"))


if __name__ == "__main__":
    unittest.main()
