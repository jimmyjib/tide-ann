# Admin commands

The coordinator takes one text command per request on a ZeroMQ REP socket (port 29699) and sends one
reply. The examples use this shell function, with `<port>` the coordinator's admin port:

```bash
c() { python3 tools/coord_client.py --endpoint "tcp://127.0.0.1:<port>" "$@"; }
c stats
c recall --queries 1000
c insert 1000 --file /data/insert.npy
c drain
```

The client's own options (`--endpoint`, `--timeout`) come before the command.
Everything from the command on is sent as written, and a `--` before the command is dropped. Any ZeroMQ
REQ socket can send the same text. The coordinator runs one admin command at a time.

## Commands

- `recall`: search a query set, report recall, hops and timing
- `insert N --file F`: insert N vectors from a file
- `mixed`: interleaved searches and inserts in one command
- `drain`: wait until every insert, backlink and broadcast is applied
- `controller-start`, `controller-status`, `controller-stop`, `controller-abort`: the controller phase
- `migration-apply`, `migration-accept`, `migration-rollback`: apply a node-move plan by hand
- `membership-check`: compare every server's ownership and vectors with the coordinator's
- `status`, `stats`, `ping`: coordinator state, server counters, round-trip times
- `recreate-head`, `reset`: rebuild the head index, reload the load-time state
- `shutdown`, `quit`, `exit`: stop every server and the coordinator, or the coordinator only

## Replies

Most commands print a JSON object between `===JSON_RESULT===` and `===JSON_END===`. An error is a line
starting with `Error:` or a JSON result with `"status": "error"`. `tools/coord_client.py` exits 0 on
success, 1 on an error reply or a ZeroMQ error, and 2 on a timeout or a usage error. The default timeout
is 5 minutes, and an unreachable coordinator shows up as a timeout. Use `--timeout 5000` for a quick
check.

While a controller phase is active, `insert`, `mixed`, `recreate-head`, `reset` and `migration-apply` are
rejected. While a migration is applied but not yet accepted or rolled back, `insert`, `mixed`,
`recreate-head` and `migration-apply` are rejected.

## recall

`recall` searches N queries and reports recall@`--K`, hops ([architecture.md](architecture.md#search)),
timing and work ([metrics.md](metrics.md)).

| Argument | Default | Meaning |
|---|---|---|
| `--queries N` | the coordinator's `--queries` | queries to run, cycling through the loaded rows |
| `--query-file F`, `--query-offset O` | `query.npy`, 0 | float32 queries from another file, from row O |
| `--groundtruth F` | `<data-dir>/groundtruth.npy` | int32, uint32 or int64 ground truth, at least `--K` ids per row |
| `--no-groundtruth` | | unscored: recall is 0 and the controller sees none |
| `--search-l L` | the coordinator's `--L` | search list size of this command, `--K` to 4096 |
| `--self-ids F` | | row i of `--query-file` is the vector of id `F[i]` |
| `--self-recall`, `--self-recall-base-id B` | | query i should find id B + i |
| `--batch-size N`, `--dispatch-window N` | the coordinator's settings | queries in flight per server thread, dispatch window (sticky) |
| `--no-merge` | merge on | do not merge pending inserts into the results |
| `--no-route-trace` | trace on | do not record each query's server route |
| `--digest` | off | add `search.result_digest`, a hash of the returned ids for comparing runs ([metrics.md](metrics.md)) |
| `--debug` | off | debug output ([debug.md](debug.md)) (sticky) |
| `--ping-baseline N`, `--ping-batch N` | | run `ping N` first and embed its result |

Sticky options apply to later commands too. Query i is scored against ground-truth row i modulo the
loaded queries. `--self-ids` needs `--query-file` and `--queries` equal to its number of ids, and reports
the rank and distance at which each id is found. `--self-recall-base-id` defaults to the load-time node
count. Bad arguments or unreadable files are rejected with an `Error:` line before any query is sent.
Unknown arguments are ignored. After inserts, use ground truth computed on `base_plus_insert.npy`. During
a controller phase, `recall` is the search load the controller measures.

## insert

```
insert N --file F [--start-id S] [--batch-size N] [--dispatch-window N]
       [--no-backlink] [--no-broadcast] [--search-only] [--debug]
```

Inserts N vectors, rows S onward of F, fewer if the file ends first. The new ids continue after the ids
the coordinator knows, whatever S is. The reply is the workload result with an `insert` section. Inserts
are applied in batches. Run `drain` before a check that needs all of them. `--no-backlink`,
`--no-broadcast` and `--search-only` skip parts of the insert for diagnosis
(`observability.diagnostic_mode`) and leave the index incomplete. Each `insert` resets them. Bad
arguments, an unreadable insert file and an insert file of another dimension are rejected with an `Error:`
line.

## mixed

```
mixed [--queries N] [--insert-ratio P] [--groundtruth F] [--insert-gt F] [--insert-file F]
      [--insert-start-id S] [--inserted-query-ratio PCT] [--no-backlink] [--no-broadcast]
      [--batch-size N] [--dispatch-window N] [--no-merge] [--digest] [--debug]
```

One workload of N operations, P% of them inserts (default 50, 1 to 99) and the rest searches. A share of
the searches (`--inserted-query-ratio`, default 10%) uses already inserted vectors as queries, scored
against `--insert-gt`. The searches cycle through the coordinator's loaded queries. Bad arguments and
files are rejected with an `Error:` line. `--digest` adds `search.result_digest` over the searches, as in
`recall`.

## drain

No arguments. Waits until every pending insert reply, backlink and broadcast is applied and acknowledged
on every server. Reply: `status`, `action`, `drain_seconds`, `pending_at_start` (`insert_responses`,
`backlink_batches`, `broadcasts`), `maintenance` and `drain_barrier_complete`.

## controller-start

```
controller-start <journal_dir> [window_sec=S] [cooldown_sec=S] [min_window_queries=N]
                 [proposal_query_limit=N] [repair=key:value[,key:value...]]
```

Drains, then starts a controller phase over every id the coordinator knows. The reply reports the drain
(`drained`, `drain_seconds`), the node counts, `ownership_epoch`, `membership_sha256` and, on success, the
controller state as in `controller-status`. `membership_sha256` is the SHA-256 of the owners as uint32. An
error reply has `"status":"error"` and an `error` string. The options and the controller's behavior are in
[controller.md](controller.md).

## controller-status

No arguments. Reply: `active`, `ownership_epoch`, `accepted_migrations` and, while a controller is active,
its state: cycle, commit and revert counts, moved bytes, `unresolved_action`, `stopped_on_error`,
`terminal_reason`, the window and repair settings and the history. After a stop it reports the stop's
result.

## controller-stop

No arguments. Closes the servers' statistics windows and ends the phase. Reply: `status`,
`unresolved_action`, `was_active` and the statistics stop (`stats_off_confirmed`, `stats_off_epoch`,
`stats_off_ranks`, `stats_off_acks`). It answers `"status":"error"` while a committed batch is pending
(`"unresolved_action":true`), with an argument, or when the statistics stop fails. Without an active
controller it answers `"status":"ok"` with `"was_active":false`.

## controller-abort

No arguments. Ends a controller that stopped on error, without a coordinator restart. Its last batch stays
in place and the controller history is dropped. Reply: `status`, `error`, `was_active`, `terminal_reason`,
`ownership_epoch`. It is rejected for a controller that has not stopped on error, while a migration is
active, and while ownership is ambiguous. `migration-accept`, or `migration-rollback` if ACCEPT was never
sent, settles the ownership.

## migration-apply

`migration-apply [--refresh] <plan>` drains, then runs PREPARE and COMMIT for the MOVE in the plan file
([data_format.md](data_format.md#migration-plan-files)). The move stays open until `migration-accept` or
`migration-rollback`. `--refresh` takes the moved nodes' current rows, and under `--pq` their vectors, from
their owners instead of the file. Under `--pq` a plan without vectors needs it. A plan's vectors must match
the original vectors, and each source checks the vector digest. Reply: `status`, `refresh`,
`ownership_epoch`, `move_count`, `payload` (bytes sent per server and the deadlines used). A plan that does
not load or does not match the membership and epoch is rejected. A PREPARE or COMMIT that fails on a server
is rolled back.

## migration-accept, migration-rollback

Neither takes arguments, and both drain first. `migration-accept` makes the committed MOVE final on every
server, and the sources free the moved rows and vectors. An incomplete ACCEPT is retried on every server.
Reply: `status`, `ownership_epoch`, `accepted_epoch`, `history_depth`. After a controller stopped on error
with ambiguous ownership, `migration-accept` settles the ownership.

`migration-rollback` restores the state before the open MOVE. Reply: `status`, `ownership_epoch`. It
answers `"status":"error"` when no MOVE is open, and it is rejected once ACCEPT was sent.

## membership-check

`membership-check [--write <path.npy>]` drains, then asks every server to check its state against the
coordinator's membership: the owner label of every id, a row for each owned id, under `--pq` the original
vector of every owned id and a PQ code of every id, and the free row-slot pool. `--write` also saves the
coordinator's membership (uint32, one owner per id). The reply has `consistent` overall and per server,
plus `vectors_exact` and mismatch counts per server. Under `--pq`, `vectors_exact` is true when the digest
of a server's original vectors matches the coordinator's. `own_only_storage`, and `own_only` per server,
are true in PQ mode ([pq_storage.md](pq_storage.md)). `missing_codes` counts ids a server has no PQ code
for. The command is rejected while a migration is active.

## State and maintenance

- `status` reports `status`, `n_base`, `total_inserted`, `n_samples` (head-index size) and `maintenance`:
  backlinks sent, pending insert replies, backlinks and broadcasts, orphan replies.
- `stats` asks every server for its counters: graph, search work and distance kinds, storage bytes and
  queues. Reply: `status`, `action` and `servers`, one object per server ([metrics.md](metrics.md)).
  `insert_neighbors_undelivered` in each server's `graph` object counts insert candidates skipped because
  they had not reached the server yet ([architecture.md](architecture.md#insert)).
- `ping [count] [--batch N]` measures `count` (default 100) round trips to every server, one at a time
  and, with `--batch N`, also in batches of N. The result is also embedded in the next workload result.
- `recreate-head` rebuilds the head index from the base and the inserted vectors with the current
  membership. Reply: `status`, `action`, `n_samples`, `total_nodes`.
- `reset` makes every server reload its load-time state from disk. The coordinator drops every inserted
  vector and pending operation, reloads `membership.npy`, sets the ownership epoch to 0, restores the
  load-time head index and drops the controller history. Reply: `status`, `action`, `n_base`,
  `membership_size`, `total_inserted`, with `status` `"error"` when a server did not confirm or the
  membership file cannot be reloaded.
- `shutdown` drains, stops every server and ends the coordinator. `quit` and `exit` end the coordinator
  only.

An unknown command answers with the list of commands, which the coordinator also prints at start-up.
