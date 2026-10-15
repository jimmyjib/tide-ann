# The runtime repartitioning controller

The controller lives in the coordinator. During a controller phase it measures where the search work
lands, moves nodes between servers in small bursts, checks every burst against the next measurement and
undoes the ones that made things worse. This document describes one cycle, the planner, the MOVE fence,
the budgets and memory that limit it, how it works together with inserts, and every option.

The authoritative descriptions are the header comments of `src/tide/include/window_controller_runtime.h`
(the cycle), `src/tide/coordinator/window_repair_planner.h` (the planner rules) and
`src/tide/include/window_action_history.h` (budgets and memory).

## One cycle

```
 BEGIN window ─> serve ≥ window_sec and ≥ min_window_queries ─> HARVEST snapshots ─> merge
      ^                                                                                 |
      |                                        verdict on the previous burst (if one is pending)
      |                                                                                 |
   cooldown <─ FENCE: PREPARE, COMMIT, ACCEPT <─ admit under budgets and caps <─ plan the next burst
```

1. **Window.** The coordinator opens a statistics window on every server (`STATS_WINDOW_BEGIN`, with the
   window epoch and the phase's node count N) while a search command is running. Every server then counts,
   for each node it expands, the distance evaluations of that expansion (exact, PQ and re-rank).
2. **Close.** The window closes when both `window_sec` seconds and `min_window_queries` completed queries
   have passed.
3. **Harvest.** Each server returns one snapshot frame: for every node it owns, its work counter and its
   owner histogram (how many of its out-neighbours each server owns); under `weight:wnb` also the work of
   its in-neighbours. The coordinator decodes and merges the K frames on a worker thread and reads the
   window's per-server counters: distance evaluations D, requests R, distance time and total search time.
4. **Verdict.** If a burst is awaiting its verdict, this window is its verify window (see [Verdict](#verdict)).
5. **Plan.** The planner computes the next burst from the merged counters ([Planner](#planner)).
6. **Admit.** The burst is cut to what the per-phase caps allow ([Budgets, caps and memory](#budgets-caps-and-memory)).
7. **Fence.** The coordinator stops dispatching new searches, waits until none is in flight (at most
   120 s), and applies the burst as one MOVE: PREPARE, COMMIT, ACCEPT ([MOVE fence](#move-fence)).
   Dispatch then resumes.
8. **Cooldown.** After `cooldown_sec` seconds the next window begins. The window after a committed burst
   is its verify window.

The controller advances only inside a running search command (`recall`, `baseline`); between commands
it waits. A search command that ends while a burst is unresolved leaves that burst pending until the next
search command.

## Weights

The per-node weight the planner balances is chosen by the `repair` token:

- `weight:w` (the default): W(v) = the distance evaluations spent expanding node v on its owner during
  the window, divided by the window's completed queries. The harvest carries the work counters and the
  owner histograms only.
- `weight:wnb`: W^nb(v) = W(v) plus the work of v's in-neighbours. The harvest also carries a 64-bit
  in-neighbour work value per node, about 3.7 times the harvest bytes of `weight:w` at K = 4.

Each server's load is then priced: at the first window the planner runs on in the phase, the controller
freezes one pooled price for the whole cluster and gives it to every server: the cost of one distance
evaluation (the distance time summed over the servers / their summed D) and of one request (their summed
total time minus distance time / their summed R), in microseconds (1 and 0 when that window carries no
valid timing). A server's load W[p] is the price per distance evaluation times the sum of the smoothed
weights of the nodes it owns; the **skew** is max(W) / mean(W).

## Planner

The planner keeps, per node, an exponential moving average of the weight (`ema`) and a per-node cooldown
in planning windows. Each planning window runs:

1. **Calibrate.** Update the averages, compute W[p] for every server and the skew. Ids inserted since the
   previous controller phase start from their first measurement; known ids keep their average.
2. **Refine.** Candidates are nodes whose best other owner q (the server owning most of the node's
   out-neighbours, lowest index on ties) owns more of its out-neighbours than its current owner, that are
   off cooldown and allowed to move to q. They are walked by gain, then weight, then id, and moved to q
   while the refine byte budget lasts. A move that would lift W[q] above (1 + `eps`) x mean(W) is
   rejected.
3. **Balance** (only when the skew exceeds `skew`). Nodes move from the heaviest server; a node v may go
   to server p only if p owns at least `tau` x deg(v) of v's out-neighbours. With
   `balance_target:lightest` (the default) every move goes to the single lightest server and the walk
   stops when the lightest server has no admissible node on the heaviest; with
   `balance_target:underloaded` a move may go to any server below the mean load (to the one holding most
   of v's out-neighbours), and only if it keeps that server's load at most (1 + `eps`) x mean(W) and below
   the heaviest server's load. The walk stops once the moved bytes reach the balance byte budget; the
   budget is checked after each move, so the last move can cross it by one row.
4. A moved node is not moved again for `cooldown` planning windows.

**Phases of the planner.** It is ACTIVE at the start. After `converge` windows without a planned move
and with the skew at or below `skew`, it is CONVERGED and only calibrates until the skew rises above
`skew` again (reactivated). If the last `converge` verified windows' peak cost all exceed
(1 + `guard`) x the peak cost of the phase's first planning window (both priced with the pooled price),
it is ABORTED (`repair_aborted`) and only calibrates for the rest of the phase.

With K = 4 the planner reproduces a fixed reference exactly (`tests/testdata/repair_golden_k4.json`,
the planner's golden test).

## MOVE fence

A burst is applied as one stop-the-world MOVE over all servers:

1. **Fence.** Dispatch is frozen; the controller waits until no search is in flight. If that takes
   longer than 120 s the burst is dropped (or, for an inverse MOVE, the controller stops on error).
2. **Refresh.** The coordinator pulls each moved node's current adjacency row (and, under `--pq`, its
   original vector) from its current owner.
3. **PREPARE.** Every server receives the whole move list, with every moved node's row and row digest
   (under `--pq` also its vector digest); under `--pq` each moved node's original vector goes to its
   target server only. Each target installs the rows (and vectors) of the nodes moving to it in free or
   new row slots; each source checks the row digest, and under `--pq` the vector digest, of every node it
   gives away against its own copy. A failed PREPARE is rolled back and the burst dropped.
4. **COMMIT.** Every server publishes the new owners. A failed COMMIT is rolled back and the controller
   stops on error.
5. **ACCEPT.** The move becomes final; the sources free the moved rows and vectors into their slot pool.
6. **Resume.** Dispatch resumes; the coordinator's ownership epoch has increased by one.

The fence pause (from freezing dispatch to ACCEPT) and the bytes it moved are journaled (`committed`:
`pause_s`, `burst_bytes`, `fence_payload_bytes`). Exchange deadlines grow with the payload.

## Verdict

The verify window is judged against the trigger window (the window the burst was planned from). Both
windows are priced with the trigger window's own per-server prices (each server's distance time / D and
(total time - distance time) / R in that window), not with the phase's pooled price. For each server
the cost per completed query is (price per distance x distances per query) + (price per request x
requests per query). The burst is **accepted** when all of these hold:

- peak ratio = max over servers of the verify cost / max of the trigger cost <= 1 + `slack`;
- total ratio = sum of the verify costs / sum of the trigger costs <= 1 + `slack`;
- hop ratio = hops per query in the verify window / in the trigger window <= 1 + `hop_slack`, where hops
  per query is the servers' search requests per query (server visits, the entry server included);
- with `recall_guard:<pp>`: the verify window's recall >= the trigger window's recall - pp. Both
  windows must be fully scored (the search command needs ground truth); otherwise the clause fails.

A **rejected** burst is undone by the inverse MOVE through the same fence (`revert_fence_open`,
`reverted`). A verify window that cannot be measured (a server served nothing, or the harvest failed)
is counted; after `verify_invalid_limit` such windows in a row the burst is treated as rejected and
reverted, so a verdict cannot block `controller-stop` forever.

The recall guard's reference is not paired: the two windows score different queries of the load, so a
change in the query mix between them enters the clause. Choose `pp` above the window-to-window recall
spread. With short windows the hop rate can vary between windows at a fixed placement by a few percent,
more than the default `hop_slack`, so small bursts can be accepted or rejected by noise
([LIMITATIONS.md](LIMITATIONS.md)).

## Budgets, caps and memory

All byte quantities count, per node, its row bytes: 4 + 4 x out-degree, plus dim x 4 under `--pq`
(a MOVE ships the original vector too). They are bound once per phase, from the owner histograms of the
phase's first merged harvest (`row_bytes_bound` in the journal); the total bytes are their sum.

- **Per window:** refine moves at most `refine` x total bytes; balance stops once it reaches `balance` x
  total bytes, so it can exceed that budget by at most one row (its last move).
- **Total cap per phase:** forward plus inverse bytes at most 2 x `cum` x total bytes.
- **Rate cap:** forward plus inverse bytes within any four controller periods (period = `window_sec` +
  `cooldown_sec`) at most 2 x 4 x (`refine` + `balance`) x total bytes.
- **Revert reservation:** every admitted burst reserves the bytes of its inverse MOVE, so admission
  spends at most half of what either cap has left. When the total cap is exhausted the controller stops
  proposing for the rest of the phase.
- **Rejection memory** (counted in planning windows): every resolved burst puts its nodes on a cooldown
  for the next 2 planning windows; a rejected burst bars each (node, source -> target) direction for the
  next 2, 4 or 8 planning windows (doubling with repeated failures); an accepted burst clears the bar of
  its directions.
- **Proposal limit:** after `proposal_query_limit` completed queries of the current search command
  (0 = no limit) the controller plans nothing new and only judges the burst that is still pending; it
  then proposes nothing for the rest of the phase (until the next `controller-start`). Use it to leave a
  verification tail at the end of a search load. The count restarts with every search command, so a
  search command run only to resolve a pending burst can plan and commit new bursts of its own until its
  count reaches the limit, and one of those can need a further search command to resolve.

## Phases with inserts

Inserts and the controller alternate. A controller phase is one `controller-start` .. `controller-stop`.

- `controller-start <journal_dir> [options]` refuses while a controller or a migration is active, drains
  every pending insert, backlink and broadcast, and sets the phase's node count N to every id the
  coordinator knows (load-time and inserted). A server that has not applied every broadcast would refuse
  the window (its N is smaller), which is why the start drains first.
- During the phase `insert`, `mixed`, `recreate-head`, `reset` and `migration-apply` are refused.
- Inserted nodes are measured and moved like load-time nodes; a MOVE of an inserted node ships its
  vector from its current owner.
- `controller-stop` is refused while a committed burst awaits its verdict (`"unresolved_action": true`):
  run another search command until the burst resolves, then stop again. The stop closes the servers'
  statistics windows.
- Inserts after the phase take the row slots that accepted MOVEs freed before appending new ones.

Kept from one phase to the next: the ownership epoch, the window epoch and counters, the rejection
memory, and the planner's per-node averages, cooldowns and window counter (new ids are added). Reset at
every start: the caps and the bytes spent against them, the rate debits, the frozen prices, the guard's
reference, the proposal limit and a cap-exhausted stop.

The kept history is bound to the configuration (server count, window and cooldown seconds, query
limits, journal directory, vector bytes per row and every `repair` value) and to the ownership epoch the
controller last left. A later `controller-start` with another configuration, or after a manual MOVE
(`migration-apply` + `migration-accept`) between phases, is refused ("history context/ownership mismatch
or pending action"); the first `controller-start` of a history needs ownership epoch 0. `reset` and
`controller-abort` drop the history.

## Stopping on error

Some failures end the controller in a terminal state (`stopped_on_error`, `terminal_reason` in
`controller-status`): a failed COMMIT or ACCEPT (`commit_failed`, `accept_failed`), a failed PREPARE
whose rollback also failed (`prepare_rollback_failed`), a failed inverse MOVE (`revert_prepare_failed`,
`revert_fence_timeout`, `revert_plan_failed`), a history failure (`history_commit_failed`,
`history_accept_failed`, `history_reverse_failed`, `history_admission_failed`) or a broken window total
(`window_work_totals_failed`). `controller-abort`
ends a terminal controller; while ownership is ambiguous (an ACCEPT may have reached some servers only)
it is refused until `migration-accept` (or `migration-rollback` when ACCEPT was never sent) settles it.
The next `controller-start` after an abort needs ownership epoch 0 (`reset` first after any accepted
MOVE).

## Journal and files

Every transition appends one JSON line (`"event"`, `"mono_ns"`, ...) to
`<journal_dir>/controller-journal.jsonl`. The journal directory must exist and be writable by the
coordinator. Each window writes its plan files into `<journal_dir>/window-<epoch>/` (`plan.bin`; a
rejected burst also `revert-plan.bin`). A plan file that cannot be written is journaled and removed: a
burst whose `plan.bin` fails is dropped (`plan_write_failed`), a revert whose `revert-plan.bin` fails goes
on from the plan held in memory (`revert_plan_record_failed`). The events:

| Event | Meaning |
|---|---|
| `controller_started`, `controller_stopped`, `controller_aborted`, `stop_refused`, `abort_refused`, `stats_off` | phase boundaries; `stats_off` is the servers' statistics stop at `controller-stop` |
| `window_open`, `window_begin_failed`, `window_abandoned` | statistics windows; a window is abandoned when its search command ends before it closes (`search-command-changed`) |
| `harvest_requested`, `snapshot_frame_arrived`, `snapshot_received`, `snapshot_rejected`, `snapshot_owner_boundary_frozen` | a harvest: requested, each server's frame arrived, validated (or rejected), and the window's counters frozen once every frame is in |
| `structure_loaded`, `structure_unavailable`, `harvest_measured` | the merged frames (owner histograms, in-neighbour work) and the window's measurement |
| `harvest_failed` | a harvest that cannot be used: frames missing or rejected (`acks`, `rejected`), the decode or load deadline passed (`decode-or-load-deadline`), or the search command ended before the frames were loaded (`measurement-command-changed`; a measurement never spans two search commands) |
| `measurement_unavailable`, `verify_deferred_measurement` | a window that could not be measured; with a burst awaiting its verdict it counts toward `verify_invalid_limit` |
| `row_bytes_bound`, `row_bytes_mismatch` | the phase's row bytes, bound from its first merged harvest, and a later harvest that derives other ones |
| `planning_submitted`, `repair_calibrated`, `repair_calibration_only`, `repair_burst_planned`, `no_positive_action`, `planning_result_discarded`, `plan_failed`, `propose_failed`, `plan_write_failed`, `dump_dir_failed` | planning |
| `observation_only`, `observation_evidence_unavailable`, `recall_guard_unscored` | a window that judges or calibrates but plans nothing: proposals disabled, no merged structure, or an unscored window under the recall guard |
| `history_budget_selected`, `history_admitted`, `history_rate_wait`, `history_total_cap_exhausted`, `history_admission_failed` | admission under the budgets and caps (actions kept and skipped); the rate cap waits, an exhausted total cap stops proposing for the phase |
| `history_forward_committed`, `history_accepted`, `history_rejected_resolved`, `history_forward_cancelled`, `history_prepare_failed_resolved` | the bytes of a burst charged against the caps, and how the burst ended |
| `fence_open`, `fence_waiting`, `fence_drain_timeout`, `prepare_failed`, `committed`, `proposal_cancelled_stale`, `proposal_cancelled_limit`, `proposal_limit_reached` | the fence |
| `repair_verdict`, `verify_accepted`, `verify_rejected`, `revert_fence_open`, `reverted`, `revert_plan_record_failed` | verdicts and reverts |
| `repair_converged`, `repair_reactivated`, `repair_aborted` | planner phases |
| `commit_failed`, `accept_failed`, `prepare_rollback_failed`, `revert_prepare_failed`, `revert_fence_timeout`, `revert_plan_failed`, `history_commit_failed`, `history_accept_failed`, `history_reverse_failed`, `window_work_totals_failed` | terminal stops ([Stopping on error](#stopping-on-error)) |
| `ownership_confirmed` | ownership settled by `migration-accept` / `migration-rollback` after a terminal stop |

`controller-status` reports the live state: `state`, `cycles`, `commits`, `reverts`,
`actions_committed`, `moved_bytes`, `total_bytes`, `verify_pending`, `unresolved_action`,
`proposals_disabled`, `stopped_on_error`, `ownership_ambiguous`, `terminal_reason`, the window settings,
`repair_config` (the parsed token), `repair` (planner phase, windows, bursts, reverts, skew, prices) and
the history (caps and bytes spent).

## Options

`controller-start <journal_dir> [window_sec=S] [cooldown_sec=S] [min_window_queries=N]
[proposal_query_limit=N] [repair=algo:local[,key:value...]]`; each option at most once.

| Option | Default | Domain |
|---|---|---|
| `window_sec` | 10 | > 0 seconds |
| `cooldown_sec` | 5 | >= 0 seconds |
| `min_window_queries` | 0 | completed queries per window, integer |
| `proposal_query_limit` | 0 (no limit) | completed queries of the current search command, integer |
| `repair` | `algo:local` with the defaults below | a comma-separated `key:value` list that must contain `algo:local` |

`repair` keys (each at most once):

| Key | Default | Domain | Meaning |
|---|---|---|---|
| `algo` | required | `local` | the planner |
| `weight` | `w` | `w`, `wnb` | W(v) or W^nb(v) |
| `refine` | 0.01 | 0..1 | refine byte budget per window (fraction of the total bytes) |
| `balance` | 0.004 | 0..1 | balance byte budget per window |
| `tau` | 0.2 | 0..1 | balance: share of a node's out-neighbours the target must own |
| `eps` | 0.03 | >= 0 | refine, and balance with `balance_target:underloaded`: load tolerance over the mean |
| `skew` | 1.05 | >= 1 | balance only above this max/mean load |
| `cooldown` | 2 | integer 0..10^6 | planning windows a moved node waits |
| `cum` | 0.45 | (0, 1] | cumulative forward-byte cap per phase |
| `slack` | 0.03 | >= 0 | verdict: peak and total cost tolerance |
| `hop_slack` | 0.02 | >= 0 | verdict: hops-per-query tolerance |
| `recall_guard` | `off` | `off` or 0..100 | verdict: tolerated recall drop in percentage points |
| `ema` | 0.5 | (0, 1] | weight smoothing |
| `converge` | 3 | integer >= 1 | quiet windows before CONVERGED; windows of the guard |
| `guard` | 0.10 | >= 0 | abort when the verified peak cost stays above (1 + guard) x the first |
| `balance_target` | `lightest` | `lightest`, `underloaded` | balance destination rule |
| `verify_invalid_limit` | 3 | integer >= 1 | unmeasurable verify windows before a forced revert |

Refusals of `controller-start` (reply `"status":"error"` with `error`): a missing journal directory
argument; a `.npy` first argument; an unknown or duplicate option; a malformed number; a malformed,
unknown, duplicate or out-of-domain `repair` key; fewer than 2 or more than 64 servers; an active
migration; an active controller; a failed drain; a journal that cannot be opened; a history bound to
another configuration or epoch (above).

The settings used by the demo (`examples/demo.py`) in its controller phase are
`window_sec=0.25 cooldown_sec=0.25 min_window_queries=8192 proposal_query_limit=172032` (the 196,608 rows
of its query stream minus the last 24,576, which only resolve verdicts) and
`repair=algo:local,refine:0.01,balance:0.004,tau:0.2,eps:0.03,skew:1.02,cooldown:1,cum:1.0,slack:0.10,
hop_slack:0.05,ema:0.5,converge:3,guard:0.10,recall_guard:off,balance_target:underloaded`; every search
command uses search list size 200. The demo prints them. They suit its short phase on a small synthetic
dataset; they are not recommended settings.
