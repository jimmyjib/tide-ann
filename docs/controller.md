# Controller

The controller runs in the coordinator during a controller phase. It measures per-node search work, moves
nodes between servers in small batches and reverts batches that made things worse.

```
c controller-start /work/controller
c recall --queries 200000 --no-groundtruth
c controller-stop
```

It works only while a search command runs, and `insert` is rejected until `controller-stop`. `c` is the
admin client function from [admin_commands.md](admin_commands.md).

## One cycle

1. The coordinator opens a statistics window on every server while a search command runs. Each server
   counts the distance evaluations of every node it expands.
2. The window closes after `window_sec` seconds and `min_window_queries` completed queries.
3. Each server sends its counters, and the coordinator merges them on a worker thread.
4. If a batch is pending, this window decides whether it is kept or reverted.
5. The planner picks the next batch, cut to the byte budgets.
6. Dispatch pauses and the batch is applied as one MOVE.
7. The next window opens after `cooldown_sec` seconds.

## Weights

- `weight:w` (default): W(v) is the number of distance evaluations spent expanding node v on its owner
  during the window, divided by the window's completed queries. Exact, PQ and re-rank evaluations count.
- `weight:wnb`: W(v) plus the work of v's in-neighbors. The servers also send a 64-bit in-neighbor work
  value per node, which makes the statistics several times larger.

Server load is W[p] = c_d x (sum of the smoothed W(v) over the nodes p owns). c_d is the distance time per
distance evaluation in the phase's first planning window, pooled over all servers. skew = max(W) / mean(W).
When that window has no valid timing, c_d is 1.

## Planner

The planner keeps an exponential moving average of each node's weight (`ema`) and a per-node cooldown,
counted in planning windows. Each planning window runs these steps:

1. Calibrate. Update the averages and compute W[p] and the skew. Ids inserted since the previous
   controller phase start from their first measurement.
2. Refine. A node is a candidate when another server q owns more of its out-neighbors than its owner
   does, with q the server owning the most (lowest index on ties). Candidates are walked by gain, then
   weight, then id, and moved to q while the `refine` byte budget lasts. A move that would lift W[q]
   above (1 + `eps`) x mean(W) is skipped.
3. Balance, only when the skew exceeds `skew`. Nodes move away from the busiest server.
   A node v may go to server p only if p owns at least `tau` x deg(v) of v's out-neighbors. With
   `balance_target:lightest` (default), every move goes to the lightest server, and the walk stops when
   the lightest server has no allowed node on the busiest one. With `balance_target:underloaded`, a move
   may go to any server below the mean load that stays at most (1 + `eps`) x mean(W) and below the
   busiest server's load. Among those, v goes to the one owning most of its out-neighbors. The walk stops
   once the moved bytes reach the `balance` budget, so the last move can cross it by one row.
   Balance moves nothing when the busiest server's nodes have almost all their neighbors on that server.
4. A node the planner picks, including one the byte caps later drop, is not picked again until `cooldown`
   planning windows after this one. `cooldown` 0 and 1 add no wait.

### Planner states

The planner starts ACTIVE. After `converge` windows without a planned move and with the skew at or below
`skew`, it is CONVERGED and only calibrates until the skew rises again. If the last `converge` checked
windows all have a peak cost above (1 + `guard`) x the peak cost of the phase's first planning window, it
is ABORTED (`repair_aborted`) and only calibrates for the rest of the phase.

## Applying a batch

A batch is applied as one MOVE ([architecture.md](architecture.md)). Dispatch pauses until no search is
in flight. If that takes more than 120 s the batch is dropped. A failed PREPARE is rolled back. A failed
COMMIT stops the controller. The pause and the bytes moved are in the journal (`committed`: `pause_s`,
`burst_bytes`), and `controller-status` counts the applied batches (`bursts`).

## Keeping or reverting a batch

The window after a committed batch is compared with the window it was planned from. A server's cost per
query is its distances and requests per query, priced with the planning window's distance time per
distance and other search time per request. The batch is kept when all of these hold:

- the peak server cost grew by at most `slack`
- the summed server cost grew by at most `slack`
- the hops per query grew by at most `hop_slack`
- with `recall_guard:<pp>`, recall dropped by at most pp points. Both windows must be fully scored, so the
  search command needs ground truth.

Otherwise the inverse MOVE reverts it (`revert_fence_open`, `reverted`). A window that cannot be measured
counts toward `verify_invalid_limit`. At the limit the batch is reverted, so a pending batch cannot block
`controller-stop` forever.

## Budgets, caps and memory

A node's row bytes are 4 + 4 x out-degree, plus dim x 4 under `--pq` because a MOVE also sends the vector.
They are fixed per phase from its first merged statistics (`row_bytes_bound`). Total bytes are their sum.

- Per window, refine moves at most `refine` x total bytes and balance stops at `balance` x total bytes.
- Per phase, forward plus inverse bytes are at most 2 x `cum` x total bytes.
- Within any four controller periods (`window_sec` + `cooldown_sec`), forward plus inverse bytes are at
  most 2 x 4 x (`refine` + `balance`) x total bytes.
- Each batch reserves the bytes of its inverse MOVE in advance. When the phase cap is used up, the
  controller stops proposing for the rest of the phase.
- A kept or reverted batch also bars its nodes from the next 2 planning windows. A node of such a batch
  can move again from the third planning window after the one that planned it, or from `cooldown` windows
  after it if that is later, so a `cooldown` of 3 or less does not lengthen the wait. A reverted batch bars
  each (node, source -> target) direction for 2, 4 or 8 planning windows, doubling with repeated failures.
  A kept batch clears the bar of its directions.
- After `proposal_query_limit` queries of a search command, the controller stops proposing and only
  checks the pending batch. The count restarts with each search command.

The controller balances search work, not node counts or memory.

## Phases with inserts

`controller-start` drains every pending insert, backlink and broadcast, then covers every id the
coordinator knows. While the phase runs, inserts and the other commands listed under
[Replies](admin_commands.md#replies) are rejected. Inserted nodes are measured and moved like load-time
nodes. A batch still pending when its search command ends stays pending, and `controller-stop` is
rejected (`"unresolved_action": true`). Run another search command until it resolves, then stop again.
Later inserts reuse the slots that moves freed.

The ownership epoch, window counters, rejection memory and planner averages carry over to the next phase.
Every `controller-start` recomputes the caps, prices and guard reference, and the proposal count starts
again. The carried history is tied to the settings it started with: the server count, `window_sec`,
`cooldown_sec`, `min_window_queries`, `proposal_query_limit`, the journal directory, the vector bytes per
row and the `repair` values `refine`, `balance`, `cum`, `recall_guard` and `weight`. A later
`controller-start` with other values for these, or after a manual MOVE, is rejected, and the error names
what differs. To change them, run `reset`, which drops the history, reloads the load-time membership and
discards the inserts. The first `controller-start` of a history needs ownership epoch 0.
`controller-abort` also drops the history when it ends a controller that stopped on error.

## Stopping on error

Some failures stop the controller (`stopped_on_error` and `terminal_reason` in `controller-status`):
`commit_failed`, `accept_failed`, `prepare_rollback_failed`, `revert_prepare_failed`, `revert_fence_timeout`,
`revert_plan_failed`, `history_admission_failed`, `history_commit_failed`, `history_accept_failed`,
`history_reverse_failed` and `window_work_totals_failed`. `controller-abort` ends a stopped controller.
While an ACCEPT may have reached only some servers, ownership is ambiguous and the abort is rejected until
`migration-accept` (or `migration-rollback` if ACCEPT was never sent) settles it. The journal then records
`ownership_confirmed`. The next `controller-start` needs ownership epoch 0, so run `reset` first if any
move was kept.

## Journal

Every transition appends one JSON line (`"event"`, `"mono_ns"`, ...) to
`<journal_dir>/controller-journal.jsonl`. The directory must exist and be writable by the coordinator.
Each window writes its plan to `<journal_dir>/window-<epoch>/plan.bin`, and a reverted batch also writes
`revert-plan.bin`. Statistics collection times out after 25 s x max(1, K / 4).

The main events:

- `controller_started` and `controller_stopped`: the phase starts and ends
- `window_open`: a statistics window opens
- `harvest_requested` and `harvest_measured`: the servers' window statistics are requested and merged
  (`harvest_failed` when they are incomplete or late)
- `repair_verdict`, then `verify_accepted` or `verify_rejected`: the pending batch is kept or reverted
- `revert_fence_open` and `reverted`: dispatch pauses and the inverse MOVE is applied
- `repair_burst_planned`: the planner picked the next batch
- `fence_open`: dispatch pauses for the batch (`fence_waiting` while searches are still in flight,
  `fence_drain_timeout` after 120 s)
- `committed`: the batch is applied, with the dispatch pause (`pause_s`) and the bytes moved (`burst_bytes`)
- `repair_converged` and `repair_aborted`: the planner is CONVERGED or ABORTED
- `history_total_cap_exhausted`: the phase cap is used up, and proposals end for the phase
- `proposal_limit_reached`: the search command reached `proposal_query_limit`

The other `history_*` events record the byte-budget checks and how each batch is committed, kept,
reverted or cancelled.

## Options

Options of `controller-start`, each at most once ([admin_commands.md](admin_commands.md)):

| Option | Default | Meaning |
|---|---|---|
| `window_sec` | 10 | minimum window length in seconds, > 0 |
| `cooldown_sec` | 5 | seconds between windows, >= 0 |
| `min_window_queries` | 0 | minimum completed queries per window |
| `proposal_query_limit` | 0 (no limit) | queries of a search command before proposals stop |
| `repair` | the defaults below | comma-separated `key:value` list |

`repair` keys, each at most once:

| Key | Default | Domain | Meaning |
|---|---|---|---|
| `weight` | `w` | `w`, `wnb` | per-node weight |
| `refine` | 0.01 | 0..1 | refine byte budget per window, share of total bytes |
| `balance` | 0.004 | 0..1 | balance byte budget per window |
| `tau` | 0.2 | 0..1 | share of a node's out-neighbors the balance target must own |
| `eps` | 0.03 | >= 0 | load tolerance over the mean |
| `skew` | 1.05 | >= 1 | balance only above this max / mean load |
| `cooldown` | 2 | integer 0..10^6 | planning windows, counted from the one that picked a node, before the planner can pick it again (a moved node waits at least 3, see [Budgets](#budgets-caps-and-memory)) |
| `cum` | 0.45 | (0, 1] | forward byte cap per phase |
| `slack` | 0.03 | >= 0 | peak and total cost tolerance |
| `hop_slack` | 0.02 | >= 0 | hops-per-query tolerance, raise it or lengthen short windows |
| `recall_guard` | `off` | `off` or 0..100 | recall drop in points, set above window-to-window spread |
| `ema` | 0.5 | (0, 1] | weight smoothing |
| `converge` | 3 | integer >= 1 | quiet windows before CONVERGED |
| `guard` | 0.10 | >= 0 | abort when the checked peak cost stays this far above the first |
| `balance_target` | `lightest` | `lightest`, `underloaded` | balance destination rule |
| `verify_invalid_limit` | 3 | integer >= 1 | unmeasurable windows before a forced revert |
