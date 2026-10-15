// Runtime repair controller: a state machine hosted by the serving pipeline loop.
//
// Cycle: BEGIN a stats window on every server mid-serving -> serve for
// window_seconds and at least min_window_queries -> request one snapshot per
// rank (snapshot wire v3: the work of every owned id, the owner histogram of
// every owned row over the K servers and the neighbour-aggregated work) ->
// decode each frame and merge them in memory on a worker thread -> if a burst is awaiting its verdict, judge it against the window it
// was planned from -> plan the next burst (window_repair_plan: calibrate,
// refine, balance) on the worker -> admit it under the byte caps -> FENCE:
// freeze dispatch, wait until no search is in flight, PREPARE (the refresh
// exchange pulls the current rows from the source servers), COMMIT, ACCEPT,
// resume -> cooldown. The next window is the verify window: a rejected burst
// is undone by the inverse MOVE through the same fence.
//
// Every transition appends one JSON line ("event", "mono_ns", ...) to
// <journal_dir>/controller-journal.jsonl.
#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "controller_hooks.h"
#include "window_controller.h"
#include "window_action_history.h"
#include "window_rank_time_cost.h"
#include "../coordinator/window_repair_planner.h"

namespace tide {
struct WindowNativeControl;

// Steady-clock nanoseconds of every journal line ("mono_ns").
inline uint64_t window_journal_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct WindowControllerConfig {
    // Bytes of each moved node's original vector that a MOVE ships besides its
    // row: dim x 4 under hybrid PQ own-only storage (controller-start sets it
    // from the cluster's storage mode), 0 under full replication. Added to every
    // node's row bytes when they bind, so the budgets and caps (fractions of the
    // total bytes) and every byte the journal reports count adjacency + vector.
    uint32_t vector_bytes_per_row = 0;
    std::string journal_dir;  // window snapshots, plans and the journal
    double window_seconds = 10.0;
    double cooldown_seconds = 5.0;
    uint64_t start_ownership_epoch = 0;  // the coordinator's epoch at start
    // K, the server count (the coordinator's n_servers); 2 <= K <= 64. No
    // default: controller-start sets it, and the start refuses 0.
    uint32_t ranks = 0;
    // Completed-query boundary of the serving command: beyond it the controller
    // only observes and judges (a verification tail), it plans nothing new.
    uint64_t proposal_query_limit = 0;
    // A window closes after BOTH window_seconds and this many completed queries.
    uint64_t min_window_queries = 0;
    WindowRepairConfig repair;
};

// Cumulative search counters read off the window ACKs (summed or per rank).
struct WindowCounterSums {
    uint64_t requests = 0;
    uint64_t distances = 0;
    uint64_t distance_us = 0;
    uint64_t total_us = 0;
    uint64_t adc_tables = 0;  // own-only ADC query tables built (their time is in total_us - distance_us)
};

// Per-rank distances and requests per completed query of one window, with the
// window's per-rank prices.
struct WindowRankCostSnapshot {
    std::vector<double> distances_per_query, requests_per_query;
    WindowRankTimeContext rank_time;
};

// What one harvested window measured; kept until its planning job resolves.
struct WindowMeasurement {
    uint64_t queries = 0, requests = 0, distances = 0, distance_us = 0, total_us = 0;
    uint64_t total_w = 0;
    std::vector<uint64_t> per_rank_w;
    double skew = 0.0;
    double hop_per_query = 0.0;
    // Live Recall@K of the window in percent; NaN unless every completed query
    // of the window was scored against ground truth.
    double recall_percent = std::numeric_limits<double>::quiet_NaN();
    WindowRankCostSnapshot rank;
    std::string fields;  // the journal fields of the measurement
};

enum class WindowControllerState : uint8_t {
    IDLE = 0,
    WINDOW_OPEN = 1,
    FENCE = 2,
    COOLDOWN = 3,
    BEGIN_WAIT = 4,
    HARVEST_WAIT = 5,
    RECOVERY_FAILED = 7,
    PLAN_WAIT = 8,
};

// The controller never blocks on its window fanouts: mid-serving, the shared
// response queue interleaves control ACKs with live search replies. The tick
// sends a fanout and records which ACKs are outstanding; the pipeline offers
// every reply to controller_offer, which routes ACKs and snapshots here.
enum class WindowExchangeKind : uint8_t { NONE = 0, BEGIN = 1, HARVEST = 2 };

struct WindowControllerRuntime {
    std::shared_ptr<WindowNativeControl> native_control;
    WindowControllerConfig config;
    // The window's node count N: the coordinator's membership size at
    // controller-start (load-time ids plus every inserted id). BEGIN and the
    // snapshot requests carry it; inserts are refused while a controller runs.
    uint32_t nodes = 0;
    // Per-node MOVE bytes of this phase: 4 + 4 x degree (+ vector_bytes_per_row),
    // the degree read off the owner histogram of the phase's first merged harvest
    // (window_controller_bind_row_bytes); empty, and total_bytes 0, until then.
    std::vector<uint64_t> row_bytes;
    std::shared_ptr<WindowActionHistory> action_history;
    uint64_t total_bytes = 0;
    uint64_t moved_bytes = 0;
    uint64_t window_epoch = 0;
    uint32_t cycles = 0;
    uint32_t commits = 0;
    uint32_t actions_committed = 0;
    uint32_t reverts = 0;
    bool stopped_on_error = false;
    // No query may resume after an unconfirmed ownership transaction.
    bool ownership_ambiguous = false;
    bool proposals_disabled = false;
    std::string terminal_reason;
    WindowControllerState state = WindowControllerState::IDLE;
    std::chrono::steady_clock::time_point window_started;
    std::chrono::steady_clock::time_point cooldown_until;
    std::chrono::steady_clock::time_point fence_started;
    std::chrono::steady_clock::time_point fence_last_heartbeat;
    uint64_t pending_bytes = 0;
    std::vector<WindowBurstAction> pending_actions;
    // The fence's skeleton plan, held in memory: the fence PREPAREs from these
    // bytes, never from a file (a revert needs no writable dump directory). pending_plan_path is the plan's record file, or empty
    // when it could not be written (an inverse MOVE only).
    std::vector<char> pending_plan;
    std::string pending_plan_path;
    WindowExchangeKind exchange = WindowExchangeKind::NONE;
    std::vector<uint8_t> exchange_acked;
    uint32_t exchange_ack_count = 0;
    bool exchange_rejected = false;
    std::chrono::steady_clock::time_point exchange_deadline;
    std::vector<uint64_t> harvest_per_rank_w;
    std::vector<std::string> snapshot_digests;
    // The validated frame of every rank of the current HARVEST, merged in memory.
    std::vector<std::shared_ptr<const WindowRankFrame>> snapshot_frames;
    uint64_t harvest_total_w = 0;
    // The servers' stats windows were closed by an all-rank STOP; a failed STOP
    // blocks new windows until it is retried.
    bool stats_off_confirmed = false;
    bool stats_stop_pending = false;
    WindowCounterSums exchange_counter_sums, window_begin_sums;
    std::vector<WindowCounterSums> exchange_rank_counters, window_begin_rank_counters;
    uint64_t window_begin_completed = 0;
    uint64_t window_begin_command = 0;
    double window_begin_recall_sum = 0.0;
    uint64_t window_begin_recall_count = 0;
    WindowMeasurement measurement;  // the last harvested window
    // The committed burst awaiting its verdict, and the inverse MOVE in flight.
    bool verify_pending = false;
    bool reverting = false;
    // Consecutive windows that could not judge the pending burst (unmeasurable
    // window, invalid observation, failed BEGIN or HARVEST); at
    // config.repair.verify_invalid_limit the burst is rejected and reverted.
    uint32_t verify_invalid_windows = 0;
    double staged_verify_hop_per_query = 0.0, verify_hop_per_query = 0.0;
    double staged_verify_recall_percent = std::numeric_limits<double>::quiet_NaN();
    double verify_recall_percent = std::numeric_limits<double>::quiet_NaN();
    WindowRankCostSnapshot staged_verify_rank_cost, verify_rank_cost;
    std::vector<WindowBurstAction> last_committed_actions;
    WindowRepairState repair_state;  // owner-installed; the planner works on frozen copies
    std::ofstream journal;
};

// Starts a controller phase over `node_count` ids (the coordinator's membership
// size): opens <journal_dir>/controller-journal.jsonl and binds the action
// history (a new phase). The row bytes bind at the first merged harvest.
// Returns nullptr with an error on invalid input.
WindowControllerRuntime* window_controller_start(
    const WindowControllerConfig& config, size_t node_count, std::string* error);
// Row bytes of a merged owner histogram (`owners` columns per node): 4 + 4 x the
// row sum (the node's out-degree over known owners) + vector_bytes.
std::vector<uint64_t> window_controller_histogram_row_bytes(const std::vector<uint32_t>& owner_hist,
                                                            uint32_t owners, uint32_t vector_bytes);
// Binds the phase's row bytes (`row_bytes`, one per node) at its first merged
// harvest: runtime.row_bytes, total_bytes and the history's phase caps, journaled
// as "row_bytes_bound". Later harvests of the phase must derive the same bytes
// (no insert runs during a phase and a MOVE keeps every row): a difference is
// refused (false, `error`), and so is a length other than runtime.nodes.
bool window_controller_bind_row_bytes(WindowControllerRuntime& runtime, std::vector<uint64_t> row_bytes,
                                      std::string* error);
// Refuses while an action is unresolved or the servers' stats windows are not
// confirmed closed after a window was opened. A terminal controller (stopped on
// error) always has an unresolved action, so it is ended by controller-abort.
bool window_controller_stop();
// controller-abort: ends a TERMINAL controller (stopped on error) whose
// ownership is not ambiguous: every server and the coordinator agree on one
// ownership epoch (the failed transaction rolled back, or ACCEPT completed, or
// an ambiguous terminal was resolved by a confirmed migration-accept or
// migration-rollback, window_controller_confirm_ownership). Whatever burst the
// terminal left in place stays where ownership left it (e.g. a rejected burst
// whose inverse MOVE failed) and no inverse MOVE is owed any more. The
// retained action history cannot account for that burst, so it is dropped: a
// later controller-start binds a fresh history, which needs ownership epoch 0
// (no MOVE ever accepted, or after reset). The window-epoch floor is kept.
// Refused (false, `refusal` says why) for a running controller (use
// controller-stop), while ownership is ambiguous, or while the servers' stats
// windows are not confirmed closed. True when no controller is active.
bool window_controller_abort_allowed(const WindowControllerRuntime& runtime, std::string* refusal);
bool window_controller_abort(std::string* refusal);
// The coordinator confirmed every server's ownership after an
// ownership-ambiguous terminal: a manual migration-accept or migration-rollback
// completed on every server and no migration is active. Dispatch resumes; the
// controller stays terminal (it opens no window and plans nothing) until
// controller-abort. No-op unless the active controller is ambiguous.
void window_controller_confirm_ownership(uint64_t ownership_epoch, const char* command);
// Drops the retained action history (caps, expected ownership epoch, rejection
// memory) so the next controller-start binds a fresh one at epoch 0; the
// window-epoch floor is kept. The coordinator's reset calls it. Refused (false)
// while a controller is active.
bool window_controller_reset_history();
WindowControllerRuntime* window_controller_active();
bool window_controller_has_unresolved_action(const WindowControllerRuntime& runtime);
bool window_controller_dispatch_frozen();
void window_controller_tick(const ControllerTickInputs& inputs);
// One JSON object body (no braces) of the controller's state, for controller-status.
std::string window_controller_status_json(const WindowControllerRuntime& runtime);
// Admits a planned burst under the byte caps, writes its skeleton plan and opens the fence.
void window_controller_apply_proposal(WindowControllerRuntime& runtime, std::vector<WindowBurstAction> actions,
                                      const std::string& propose_fields, std::chrono::steady_clock::time_point now);

// --- pure pieces (unit-tested) ---
double window_controller_skew(const std::vector<uint64_t>& per_rank_w);
// Serializes a skeleton MOVE plan (empty rows with self-consistent digests;
// the refresh exchange replaces them with the current rows).
bool window_controller_skeleton_plan(const std::vector<WindowBurstAction>& actions,
    uint64_t epoch_from, uint32_t n_servers, std::vector<char>* out, std::string* error);
// Swaps source and target of every action: the revert plan of a rejected burst.
std::vector<WindowBurstAction> window_controller_inverse_actions(
    const std::vector<WindowBurstAction>& actions);
void window_controller_arm_exchange(WindowControllerRuntime& runtime, WindowExchangeKind kind);
bool window_controller_is_ack(const void* data, size_t size);
// Consumes any stats-window ACK; only one matching the pending exchange and
// window epoch changes state. A rejected or invalid one marks the exchange rejected.
bool window_controller_offer_ack(WindowControllerRuntime& runtime, uint32_t server,
                                 const void* data, size_t size);
inline bool window_controller_exchange_complete(const WindowControllerRuntime& runtime) {
    return runtime.exchange != WindowExchangeKind::NONE && !runtime.exchange_rejected &&
           runtime.exchange_ack_count == runtime.config.ranks;
}
bool window_controller_observation_ready(const WindowControllerRuntime& runtime,
    const ControllerTickInputs& inputs, double elapsed_seconds);
// Counter domain of the window: same command, monotone counters, enough queries,
// every rank with nonzero distances and requests. Recall never enters it.
bool window_controller_measurement_valid(const WindowControllerRuntime& runtime,
    const ControllerTickInputs& inputs);
// The window's per-rank costs from the BEGIN and HARVEST ACK counters.
bool window_controller_rank_snapshot(const WindowControllerRuntime& runtime,
    uint64_t queries, WindowRankCostSnapshot* out);
// Window Recall@K in percent when every completed query was scored; NaN otherwise.
double window_controller_window_recall(const WindowControllerRuntime& runtime,
    const ControllerTickInputs& inputs);
// Per-rank costs (us per completed query) of `measured` priced with the frozen
// per-rank prices of `prices` (the trigger window), one per rank. False when
// unavailable or when the two windows cover different rank counts.
bool window_controller_rank_costs(const WindowRankCostSnapshot& prices,
    const WindowRankCostSnapshot& measured, std::vector<double>* costs);
// The BEGIN/HARVEST exchange deadline for K ranks, counted from the moment the
// exchange is armed (the HARVEST deadline covers capture, transfer, decode and
// the in-memory merge): 25 s at K <= 4, growing linearly with K (25 s x
// max(1, K / 4)); window_control_detail::kExchangeDeadlineSec has the basis.
double window_controller_exchange_deadline_sec(uint32_t ranks);
// Repair prices: one pooled us/distance and us/request frozen at the first
// planned window, and the window peak priced with them (the cumulative guard).
void window_controller_repair_freeze_prices(WindowControllerRuntime& runtime,
                                            const WindowRankCostSnapshot& snapshot);
double window_controller_repair_peak_us(const WindowControllerRuntime& runtime,
                                        const WindowRankCostSnapshot& snapshot);
inline uint64_t window_counter_delta(uint64_t end, uint64_t begin) {
    return end >= begin ? end - begin : 0;
}
inline bool window_controller_fence_drained(const ControllerTickInputs& inputs) {
    return inputs.logical_in_flight == 0 && inputs.transport_in_flight == 0 &&
           inputs.queued_search == 0 && inputs.queued_responses == 0;
}

}  // namespace tide
