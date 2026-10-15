# Admin commands

The coordinator reads one text command per request on a ZeroMQ REP socket (`tcp://*:29699`) and answers
with one reply. The command is split on whitespace; the first word selects it. The source is
`process_daemon_command` in `src/tide/coordinator/admin_commands.cpp`.

```bash
python3 tools/coord_client.py --endpoint tcp://127.0.0.1:<port> [--timeout ms] [--] <command> [arguments...]
python3 tools/coord_client.py --endpoint tcp://127.0.0.1:<port> insert 1000 --file /data/insert.npy
```

The client's own options come first; everything from the command on is sent as written, its options
included. A `--` before the command is accepted and dropped. The default timeout is 300,000 ms. Any
ZeroMQ REQ socket can send the same text (the demo uses `tools/local_cluster.py`'s `send`).

## Replies

A reply is the command's text output. Most commands end it with one JSON object between the lines
`===JSON_RESULT===` and `===JSON_END===`:

- `controller-*`, `migration-*`, `membership-check`: `"status":"ok"` or `"status":"error"` (no space);
- `drain`, `reset`, `status`, `recreate-head`: `"status": "ok"` or `"status": "error"` (with a space);
- `stats`: `"status": "ok"`, or `"partial"` when not every server answered;
- `recall`, `baseline`, `insert`, `mixed`: the workload result ([METRICS.md](METRICS.md)), without a
  `status` field. A workload that a server or a frozen dispatch ended early answers
  `{"status": "error", "action": ..., "error": ..., "completed_queries": ..., "completed_inserts": ...}`
  instead.

How a command refuses depends on the command:

- a line starting `Error:` and no JSON result: argument and input errors of `recall`/`baseline`,
  `insert` and `mixed`, the argument errors of `migration-apply`, `migration-accept` and
  `migration-rollback`, and the shared refusals in the table below (a controller phase is active, or a
  migration is open);
- the command's JSON result with `"status":"error"` and an `error` string: `controller-start`,
  `controller-abort` and `membership-check`, their argument errors included;
- `controller-stop`: `"status":"error"` with `unresolved_action` and no `error` field (an extra argument,
  or a burst still awaiting its verdict);
- `migration-apply`, `migration-accept` and `migration-rollback`: a MOVE that failed or cannot run (for
  example a rollback with no open MOVE) answers `"status":"error"`;
- `drain` and `reset`: a failure answers `"status": "error"`;
- a workload that ended early: an `Error:` line and then a JSON result with `"status": "error"` (below);
- an unknown command: `Unknown command: <name>` and the command list.

Some internal failures print `[ERROR] ...` lines before the JSON. `tools/coord_client.py` prints the whole
reply and exits 0 for a reply that is not an error; 1 for an error reply (a JSON result whose `status`
is `"error"`, in either spacing, or a reply without a JSON result that contains "error" in any case or
answers an unknown command), for an endpoint ZeroMQ rejects (bad syntax or an unknown transport) and for a
ZeroMQ send or receive error; 2 for no reply within `--timeout` and for a usage error. ZeroMQ connects in
the background, so a coordinator that is stopped or unreachable, a wrong port or a host name that does not
resolve shows up as a timeout (exit 2) after `--timeout` (default 300000 ms, 5 minutes); pass a short
`--timeout`, for example `--timeout 5000`, for a liveness check. `--exit-zero-on-reply` makes it exit 0 for
every reply, for callers that read the status themselves.

## Refusals that apply to several commands

| Condition | Refused commands | Reply |
|---|---|---|
| a controller phase is active | `insert`, `mixed`, `recreate-head` | `Error: <command> requires a stopped controller (controller-stop; controller-abort ends a terminal one): inserts and the runtime controller alternate` |
| a controller phase is active | `migration-apply` | `Error: migration-apply requires a stopped controller (...): the controller's fence is the only MOVE of a controller phase` |
| a controller phase is active | `reset` | `Error: reset requires a stopped controller (controller-stop; controller-abort ends a terminal one)` |
| a migration is applied but not accepted or rolled back | `insert`, `mixed`, `recreate-head` | `Error: <command> requires no active migration (migration-accept or migration-rollback first): the plan of epoch A -> B is unresolved` |

## Search and insert workloads

### `recall` (alias `baseline`)

```
recall [--queries N] [--query-file F] [--query-offset O] [--groundtruth F | --no-groundtruth]
       [--search-l L] [--self-ids F] [--self-recall] [--self-recall-base-id B]
       [--batch-size N] [--dispatch-window N] [--no-merge] [--no-route-trace] [--debug]
       [--ping-baseline N] [--ping-batch N]
```

Searches N queries and reports recall@`--K` (the result size), hops (servers visited per query, the entry
server included), timing and work ([METRICS.md](METRICS.md)).

| Argument | Default | Meaning |
|---|---|---|
| `--queries N` | the coordinator's `--queries` (capped at `query.npy`'s rows) | queries to run; more than the loaded rows cycle through them |
| `--query-file F`, `--query-offset O` | `query.npy`, 0 | float32 queries from another file, starting at row O |
| `--groundtruth F` | `<data-dir>/groundtruth.npy` | ground truth for recall: a 2D int32, uint32 or int64 array with a row for each query this command scores, that is the first min(`--queries`, loaded queries) rows (query i is scored against row i mod the number of loaded queries), and at least `--K` ids per row |
| `--no-groundtruth` | | unscored: recall is reported as 0 and the controller sees no recall |
| `--search-l L` | the coordinator's `--L` | search list size of this command only, from `--K` to 4096 |
| `--self-ids F` | | self-queries: row i of `--query-file` is the vector of id `F[i]`; reports the rank and distance at which each id is found. Needs `--query-file` and `--queries` equal to the number of ids |
| `--self-recall`, `--self-recall-base-id B` | | older self-recall: query i is expected to find id B + i (B defaults to the load-time node count) |
| `--batch-size N`, `--dispatch-window N` | the coordinator's settings | queries in flight per server thread and the dispatch window; **they stay set for later commands** |
| `--no-merge` | merge on | do not merge pending inserts into the results |
| `--no-route-trace` | trace on | do not record the server route of each query |
| `--debug` | | debug output ([DEBUG.md](DEBUG.md)); **stays on for later commands** |
| `--ping-baseline N`, `--ping-batch N` | | run `ping N` first and embed its result |

Unknown arguments are ignored. Errors (the same for `baseline`), each before any query is sent: `Error:
cannot read the ground truth: ...` (an unreadable `--groundtruth` file, a missing
`<data-dir>/groundtruth.npy` without `--no-groundtruth`, or a malformed one: not an NPY file, another
dtype than int32, uint32 or int64, Fortran order, not 2D, or shorter than its shape); `Error: ground
truth <rows> x <cols> does not cover the <n> queries x K=<K> this command scores` (fewer rows than the
loaded queries the command uses, or fewer than `--K` ids per row); `Error: cannot read the query file:
...` (an unreadable `--query-file`, or one whose header shape cannot be parsed; its dtype, memory order and
length are not checked, so it must be a float32 C-order [rows, dim] matrix); `Error: --search-l needs a search list size from K
(<K>) to <max>`;
`Error: cannot read --self-ids: ...`; `Error: --self-ids names no id`; `Error: --self-ids needs
--query-file and --queries equal to its <n> ids`; `Error: query-file dim <d> != server dim <dim>`;
`Error: query-offset <o> >= file rows <rows>`; `Error: --self-ids names <n> ids, the query file supplies
<rows> rows`. While a controller phase is active, `recall` is the search load the controller measures
([CONTROLLER.md](CONTROLLER.md)).

### `insert`

```
insert N --file F [--start-id S] [--batch-size N] [--dispatch-window N]
       [--no-backlink] [--no-broadcast] [--search-only] [--debug]
```

Inserts N vectors, rows S.. of F (fewer if the file ends first). The new ids continue after the ids the
coordinator knows (load-time count + inserted so far), whatever S is. The reply is the workload result
with an `insert` section. Inserts finish asynchronously on the servers: run `drain` before a check that
needs every insert applied.

- `--no-backlink`, `--no-broadcast`, `--search-only`: diagnostic variants that skip parts of the insert
  (`observability.diagnostic_mode` marks them); the index is then incomplete. Each `insert` resets them.
- Errors: `Error: insert requires count > 0`; `Error: inserts need an insert file (insert ... --file PATH,
  mixed ... --insert-file PATH)`; `Error: cannot read the insert file: ...`; `Error: insert file
  dimension <d> != index dimension <dim>`; `Error: insert start id <s> is beyond the file's <rows>
  rows`; the controller and migration refusals above.

### `mixed`

```
mixed [--queries N] [--insert-ratio P] [--groundtruth F] [--insert-gt F] [--insert-file F]
      [--insert-start-id S] [--inserted-query-ratio PCT] [--no-backlink] [--no-broadcast]
      [--batch-size N] [--dispatch-window N] [--no-merge] [--debug]
```

One workload of N operations, P % of them inserts (default 50, 1..99) and the rest searches; a share of
the searches (`--inserted-query-ratio`, default 10 %) uses already inserted vectors as queries, scored
against `--insert-gt`. The searches cycle through the coordinator's loaded queries. Errors: `Error:
insert-ratio must be between 1 and 99`; `Error: inserted-query-ratio must be between 0 and 100`; the
insert-file errors of `insert`; `Error: mixed needs queries: ...` when the coordinator loaded no query
rows; the ground truth errors of `recall` (an unreadable or malformed `--groundtruth`, or one that does
not cover the searches); `Error: cannot read the insert ground truth: ...` and `Error: insert ground
truth <rows> x <cols> has fewer than K=<K> ids per row` for `--insert-gt`; the controller and migration
refusals above. (`mixed` runs only in the C++ unit tests: its refusals, and one `mixed --debug` command
against in-process fake servers that answer its searches and inserts.)

### `drain`

`drain` (no arguments) waits until every pending insert reply, backlink and broadcast is applied and
acknowledged on every server. Reply: `status`, `action`, `drain_seconds`, `pending_at_start`
(`insert_responses`, `backlink_batches`, `broadcasts`), `maintenance` (replies, backlink batches and
edges, broadcast acknowledgements, and `settled_at_barrier`), `drain_barrier_complete`.

## Controller phase

The options are listed in [CONTROLLER.md](CONTROLLER.md#options).

### `controller-start`

```
controller-start <journal_dir> [window_sec=S] [cooldown_sec=S] [min_window_queries=N]
                 [proposal_query_limit=N] [repair=algo:local[,key:value...]]
```

Drains, then starts a controller phase over every id the coordinator knows. Reply: `status`, `error`,
`journal_dir`, `nodes` (the phase's N), `n_base`, `inserted`, `drained`, `drain_seconds`,
`drain_settled_at_barrier`, `drain` (the drain's work), `membership_sha256` (of the coordinator's
membership as uint32 owners), `ownership_epoch`, and on success the controller state (as
`controller-status`).

Refusals (`"status":"error"`, `error`): `controller-start requires <journal_dir> ...`; `controller-start
takes no row-bytes file (...)` for a first argument ending in `.npy`; `duplicate option <key>`; `unknown
option <key>`; `window_sec needs a number`, `cooldown_sec needs a number`, `min_window_queries needs a
count`, `proposal_query_limit needs a count`; a `repair` token error (`repair token is empty`, `repair
token pair must be key:value`, `repair token duplicate key <k>`, `repair algo must be local`, `repair
token weight must be w (the default) or wnb`, `repair token balance_target must be lightest (the
default) or underloaded`, `repair token value <k>`, `repair token unknown key <k>`, `repair token domain
<k>`, `repair token requires algo:local`); `the repair controller supports 2 to 64 servers`; `a migration
is active; accept or roll it back first`; `controller already active`; `the drain before the controller
phase failed`; `invalid controller config` (for example `window_sec=0`); `cannot open controller
journal`; `history context/ownership mismatch or pending action` or `a first history requires ownership
epoch 0` ([CONTROLLER.md](CONTROLLER.md#phases-with-inserts)).

### `controller-status`

No arguments. Reply: `status` (always `"ok"`), `active`, `ownership_epoch`, `accepted_migrations`, and
while a controller is active its full state ([CONTROLLER.md](CONTROLLER.md#journal-and-files)); after a
stop, the stop's receipt.

### `controller-stop`

No arguments. Closes the servers' statistics windows and ends the phase. Reply: `status`,
`unresolved_action`, `was_active`, and the receipt of the statistics stop (`stats_off_confirmed`,
`stats_off_epoch`, `stats_off_ranks`, `stats_off_receipts`). Refused (`"status":"error"`) while a
committed burst awaits its verdict (`"unresolved_action":true`; run a search command until it resolves,
then stop again), when an argument is given, or when the statistics stop fails. Stopping when no
controller is active answers `"status":"ok"` with `"was_active":false`.

### `controller-abort`

No arguments. Ends a controller that stopped on error (`stopped_on_error`), without a coordinator
restart; the burst it left in place stays and the controller history is dropped (the next
`controller-start` needs ownership epoch 0: `reset` first after any accepted MOVE). Reply: `status`,
`error`, `was_active`, `terminal_reason`, `ownership_epoch`. Refusals: `controller-abort takes no
arguments`; `the controller is not terminal: controller-stop ends it once its burst is resolved`;
`ownership is ambiguous after <reason>: complete migration-accept (or migration-rollback if ACCEPT was
never sent) first`; `a migration is active; accept or roll it back first`; `the all-rank stats STOP
failed`.

## Node moves by hand

### `migration-apply`

```
migration-apply [--refresh] <plan>
```

Drains, then PREPAREs and COMMITs the MOVE in the plan file ([DATA_FORMAT.md](DATA_FORMAT.md#migration-plan-files)).
The move stays open until `migration-accept` or `migration-rollback`; until then `insert`, `mixed` and
`recreate-head` are refused. `--refresh` takes the moved nodes' current rows (and, under `--pq`, their
vectors) from their owners instead of the file's; under `--pq` a plan without vectors needs it. A file's
vectors must equal the originals byte for byte, and each source checks the vector digest against its own
vector. Reply: `status`, `refresh`, `ownership_epoch`, `move_count`, `payload` (bytes sent, per server,
and the deadlines used).

Refusals: `Error: migration-apply requires exactly one plan path (migration-apply [--refresh] <plan>)`;
during a controller phase (above); `[ERROR] A migration journal is already active; rollback or reset
first`; `[ERROR] Migration PREPARE requires empty pending maintenance state`; a plan that does not load
(`[ERROR] migration plan ...`: magic, version, server count, epochs not consecutive, invalid ids, owner
transitions or digests) or does not match the coordinator's membership and epoch; a PREPARE or COMMIT
that fails on a server (rolled back: `[ERROR] Distributed ... failed; attempting rollback`).

### `migration-accept`

No arguments. Drains, then makes the committed MOVE final on every server (sources free the moved rows
and vectors). Reply: `status`, `ownership_epoch`, `accepted_epoch`, `history_depth`. Errors: `Error:
migration-accept takes no arguments`; `[ERROR] No committed migration to accept`; an incomplete ACCEPT
is retried on every server. After a controller stopped on error with ambiguous ownership, a successful
`migration-accept` settles the ownership (`controller-abort` is then allowed).

### `migration-rollback`

No arguments. Drains, then restores the state before the open MOVE. Reply: `status`, `ownership_epoch`.
Errors: `Error: migration-rollback takes no arguments`; `"status":"error"` when no MOVE is open; `[ERROR]
ACCEPT was already sent for the active migration (...); a server that accepted it cannot roll back.
Retry migration-accept`.

### `membership-check`

```
membership-check [--write <path.npy>]
```

Drains, then asks every server to check its state against the coordinator's membership: the owner label
of every id, a local row exactly on each owned id, under `--pq` the original vector of every owned id
(compared through a per-server digest with the coordinator's copy, byte for byte) and a PQ code of every
id, and the free row-slot pool. `--write` also saves the coordinator's membership (uint32, one owner per
id). Reply: `status`, `error`, `ids`, `ownership_epoch`, `own_only_storage`, `original_vectors`,
`written`, `consistent` (true only when every server is), `coordinator_owned_by_rank`, and per server
`rank`, `consistent`, `owned`, `mapped`, `vectors`, `membership_mismatches`, `mapping_mismatches`,
`vector_mismatches`, `missing_codes`, `vectors_exact`, `vector_digest`, `expected_vector_digest`,
`first_mismatch`, `n_local`, `free_slots`, `pool_ok`. Errors (`"status":"error"`, `error`):
`membership-check takes no arguments but --write <path.npy>`; `a migration is active; accept or roll it
back first`; `drain failed`; `not every server answered the membership check`; `cannot write the
membership: ...`.

## State and maintenance

### `status`

No arguments. Reply: `status`, `n_base`, `total_inserted`, `n_samples` (head-index size), `maintenance`
(backlink edges and batches sent since start, pending insert replies, backlinks and broadcasts, orphan
replies).

### `stats`

No arguments. Asks every server for its counters (graph, search work and distance kinds, storage bytes,
queues). Reply: `status`, `action`, `servers` (one object per server, `partition_id` = rank;
[METRICS.md](METRICS.md)). Each server's `graph` object includes the backlink counters and
`insert_neighbors_undelivered`: insert candidates the server had not received yet (an id another server
inserted, whose broadcast or backlink was still queued there), dropped before the prune.

### `ping`

```
ping [count] [--batch N]
```

Measures `count` (default 100) round trips to every server, sequentially and, with `--batch N`, also
pipelined in batches of N. The result is also embedded in the next workload result. (Not exercised.)

### `recreate-head`

No arguments. Rebuilds the head index from the base and the inserted vectors with the current
membership. Reply: `status`, `action`, `n_samples`, `total_nodes`. Refused during a controller phase and
while a MOVE is open (above). (Exercised by the C++ unit tests only.)

### `reset`

No arguments. Every server reloads its load-time state from disk; the coordinator drops every inserted
vector and pending operation, reloads `membership.npy`, sets the ownership epoch to 0, restores the
load-time head index and drops the controller history. Reply: `status`, `action`, `n_base`,
`membership_size`, `total_inserted`. Refused during a controller phase; `status` is `"error"` when a
server did not confirm or the membership file cannot be reloaded.

### `shutdown`, `quit`, `exit`

`shutdown` drains, stops every server through a quiescent exchange and ends the coordinator. `quit` and
`exit` end the coordinator only. The reply is a text line (`Coordinator shutting down...` or `[ERROR]
Graceful shutdown incomplete`). (Not exercised as admin commands; the local runs remove their containers
instead. The servers' shutdown exchange has its own C++ unit test.)

### Unknown commands

`Unknown command: <word>`, followed by `Available commands:` and every command this document covers
(the coordinator prints the same list at start-up).

## What ran in this release

The demo ran `recall` (with `--search-l` in every search, `--query-file` and `--groundtruth`, and
`--no-groundtruth` for its query stream and for the searches of one hot cluster's queries), `insert`,
`drain`, `stats`, `membership-check` (with `--write`), `controller-start`, `controller-status` and
`controller-stop`. The local regression runs, which send every command through `tools/coord_client.py`,
also ran `status`, `migration-apply`, `migration-accept`, `reset` and `recall --self-ids`, and refused
`controller-stop` calls resolved by further searches (the demo has the same retry). `recreate-head`,
`controller-abort` and `migration-rollback` are exercised by the C++ unit tests
(`tests/test_coordinator_admin.cpp`, `tests/test_window_controller_lifecycle.cpp`), `mixed` only there
(its refusals, and one `mixed --debug` run against in-process fake servers that answer its searches and
inserts); the `baseline` alias only by a unit test of its refusal (a missing default ground truth);
`ping`, `shutdown`, `quit` and `exit` are not exercised.
