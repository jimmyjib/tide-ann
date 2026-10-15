#!/usr/bin/env python3
"""Send one admin command to the coordinator over ZeroMQ and print its reply.

The coordinator listens on a REQ/REP socket (port 29699 by default). The reply is printed whole.

Usage:
    python3 coord_client.py [--endpoint tcp://host:port] [--timeout ms] [--exit-zero-on-reply] [--] command args...

The client's own options come first; everything from the command on is sent to the coordinator
as written, its options included (a ``--`` before the command is accepted and dropped).

Exit codes: 0 for a reply that is not an error; 1 for an error reply, an endpoint ZeroMQ rejects
(bad syntax or an unknown transport) or a ZeroMQ send or receive error; 2 for no reply within
--timeout and for a usage error. An error reply is a JSON result whose "status" is "error" (in
any spacing), or a reply without a JSON result that contains "error" in any case (an
``Error: ...`` refusal) or answers an unknown command. ``--exit-zero-on-reply`` exits 0 for every
reply the coordinator sent, error replies included, for callers that read the status themselves.
ZeroMQ connects in the background: a coordinator that is stopped or unreachable, a wrong port or a
host name that does not resolve shows up as a timeout (exit 2) after --timeout (default 300000 ms);
use a short --timeout (for example 5000) for a liveness check.

Examples:
    python3 coord_client.py status
    python3 coord_client.py recall --queries 1000
    python3 coord_client.py insert 1000 --file /data/insert.npy --start-id 0
    python3 coord_client.py -- insert 1000 --file /data/insert.npy --start-id 0
"""

import argparse
import json
import re
import sys

try:
    import zmq
except ImportError:
    print("ERROR: pyzmq not installed. Run: pip install pyzmq", file=sys.stderr)
    sys.exit(1)

RESULT_BEGIN = "===JSON_RESULT==="
RESULT_END = "===JSON_END==="
ERROR_STATUS = re.compile(r'"status"\s*:\s*"error"')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Coordinator admin client (ZeroMQ)')
    parser.add_argument('--endpoint', default='tcp://localhost:29699',
                        help='ZMQ endpoint (default: tcp://localhost:29699)')
    parser.add_argument('--timeout', type=int, default=300000,
                        help='Receive timeout in ms (default: 300000 = 5 min)')
    parser.add_argument('--exit-zero-on-reply', action='store_true',
                        help='exit 0 for every reply, error replies included (read the reply status yourself)')
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='the command and its arguments, sent as written')
    args = parser.parse_args(argv)
    words = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not words:
        parser.error('a command is required')
    args.command_text = ' '.join(words)
    return args


def reply_is_error(response):
    """True for an error reply: the JSON result's status is "error" (any spacing), or, without a JSON result,
    the reply contains "error" in any case or answers an unknown command."""
    if RESULT_BEGIN in response and RESULT_END in response:
        body = response.rsplit(RESULT_BEGIN, 1)[1].split(RESULT_END, 1)[0].strip()
        try:
            result = json.loads(body)
        except ValueError:
            return ERROR_STATUS.search(body) is not None
        return isinstance(result, dict) and result.get('status') == 'error'
    return 'ERROR' in response.upper() or response.lstrip().startswith('Unknown command')


def main(argv=None):
    args = parse_args(argv)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout)
    socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5s send timeout
    socket.setsockopt(zmq.LINGER, 0)

    try:
        socket.connect(args.endpoint)
    except zmq.ZMQError as e:
        print(f"ERROR: Failed to connect to {args.endpoint}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        socket.send_string(args.command_text)
    except zmq.ZMQError as e:
        print(f"ERROR: Failed to send command: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        response = socket.recv_string()
        print(response)
        if args.exit_zero_on_reply:
            sys.exit(0)
        sys.exit(1 if reply_is_error(response) else 0)
    except zmq.Again:
        print(f"ERROR: Timeout waiting for response (>{args.timeout}ms)", file=sys.stderr)
        sys.exit(2)
    except zmq.ZMQError as e:
        print(f"ERROR: Failed to receive response: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        socket.close()
        context.term()


if __name__ == '__main__':
    main()
