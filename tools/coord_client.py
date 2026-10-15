#!/usr/bin/env python3
"""Send one admin command to the coordinator over ZeroMQ and print its reply.

Usage:
    python3 coord_client.py [--endpoint tcp://host:port] [--timeout ms] [--] command args...

Options before the command belong to the client. Everything after it is sent as is.
Exit status: 0 ok, 1 error reply or ZeroMQ error, 2 timeout or bad usage.
An unreachable coordinator shows up as a timeout. Use --timeout 5000 for a quick check.

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
    print("error: pyzmq is not installed (pip install pyzmq)", file=sys.stderr)
    sys.exit(1)

RESULT_BEGIN = "===JSON_RESULT==="
RESULT_END = "===JSON_END==="
ERROR_STATUS = re.compile(r'"status"\s*:\s*"error"')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Send one admin command to the coordinator.')
    parser.add_argument('--endpoint', default='tcp://localhost:29699',
                        help='coordinator endpoint (default: tcp://localhost:29699)')
    parser.add_argument('--timeout', type=int, default=300000,
                        help='receive timeout in ms (default: 300000 = 5 min)')
    parser.add_argument('--exit-zero-on-reply', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='command and its arguments, sent as is')
    args = parser.parse_args(argv)
    words = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not words:
        parser.error('a command is required')
    args.command_text = ' '.join(words)
    return args


def reply_is_error(response):
    """True if the reply reports an error."""
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
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    socket.setsockopt(zmq.LINGER, 0)

    try:
        socket.connect(args.endpoint)
    except zmq.ZMQError as e:
        print(f"error: cannot connect to {args.endpoint}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        socket.send_string(args.command_text)
    except zmq.ZMQError as e:
        print(f"error: cannot send the command: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        response = socket.recv_string()
        print(response)
        if args.exit_zero_on_reply:
            sys.exit(0)
        sys.exit(1 if reply_is_error(response) else 0)
    except zmq.Again:
        print(f"error: no reply within {args.timeout} ms", file=sys.stderr)
        sys.exit(2)
    except zmq.ZMQError as e:
        print(f"error: cannot receive the reply: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        socket.close()
        context.term()


if __name__ == '__main__':
    main()
