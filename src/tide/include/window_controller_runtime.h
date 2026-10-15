// Controller state machine, run by the pipeline loop. Each transition appends one JSON line to
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
#include "window_repair_planner.h"

namespace tide {
struct WindowNativeControl;

// steady-clock ns of every journal line (mono_ns)
inline uint64_t window_journal_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct WindowControllerConfig {
    // Vector bytes a MOVE sends per node besides its row: dim x 4 in PQ mode, 0 with --full-vectors.
    // Counted in every row's bytes, so budgets, caps and the journal include it.
    uint32_t vector_bytes_per_row = 0;
    std::string journal_dir;  // window snapshots, plans and the journal
    double window_seconds = 10.0;
    double cooldown_seconds = 5.0;
    uint64_t start_ownership_epoch = 0;  // the coordinator's epoch at start
    // K, the server count (2..64), set by controller-start
    uint32_t ranks = 0;
    // after this many completed queries the controller only checks, it plans nothing new
    uint64_t proposal_query_limit = 0;
    // a window closes after both window_seconds and this many completed queries
    uint64_t min_window_queries = 0;
    WindowRepairConfig repair;
};

// cumulative search counters from the window ACKs
struct WindowCounterSums {
    uint64_t requests = 0;
    uint64_t distances = 0;
    uint64_t distance_us = 0;
    uint64_t total_us = 0;
    uint64_t adc_tables = 0;  // ADC tables built (PQ mode)
};

// per-rank distances and requests per completed query, with the window's prices
struct WindowRankCostSnapshot {
    std::vector<double> distances_per_query, requests_per_query;
    WindowRankTimeContext rank_time;
};

// one window's measurement, kept until its planning job resolves
struct WindowMeasurement {
    uint64_t queries = 0, requests = 0, distances = 0, distance_us = 0, total_us = 0;
    uint64_t total_w = 0;
    std::vector<uint64_t> per_rank_w;
    double skew = 0.0;
    double hop_per_query = 0.0;
    // Recall@K in percent, NaN unless every completed query was scored
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

// The tick sends a fanout and returns, and controller_offer routes the ACKs as they arrive.
enum class WindowExchangeKind : uint8_t { NONE = 0, BEGIN = 1, HARVEST = 2 };

struct WindowControllerRuntime {
    std::shared_ptr<WindowNativeControl> native_control;
    WindowControllerConfig config;
    // node count N: the membership size at controller-start
    uint32_t nodes = 0;
    // per-node MOVE bytes: 4 + 4 x degree (+ vector_bytes_per_row), bound at the first snapshot
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
    // no query resumes after an unconfirmed ownership change
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
    // The batch's plan, kept in memory, with pending_plan_path as its record file (empty if it could
    // not be written).
    std::vector<char> pending_plan;
    std::string pending_plan_path;
    WindowExchangeKind exchange = WindowExchangeKind::NONE;
    std::vector<uint8_t> exchange_acked;
    uint32_t exchange_ack_count = 0;
    bool exchange_rejected = false;
    std::chrono::steady_clock::time_point exchange_deadline;
    std::vector<uint64_t> harvest_per_rank_w;
    std::vector<std::string> snapshot_digests;
    // validated frames of the current snapshot, merged in memory
    std::vector<std::shared_ptr<const WindowRankFrame>> snapshot_frames;
    uint64_t harvest_total_w = 0;
    // closed by an all-rank STOP (a failed STOP blocks new windows until it is retried)
    bool stats_off_confirmed = false;
    bool stats_stop_pending = false;
    WindowCounterSums exchange_counter_sums, window_begin_sums;
    std::vector<WindowCounterSums> exchange_rank_counters, window_begin_rank_counters;
    uint64_t window_begin_completed = 0;
    uint64_t window_begin_command = 0;
    double window_begin_recall_sum = 0.0;
    uint64_t window_begin_recall_count = 0;
    WindowMeasurement measurement;  // the last measured window
    bool verify_pending = false;  // a committed batch waits for its check
    bool reverting = false;       // an inverse MOVE is in flight
    // windows in a row that could not check the pending batch (see verify_invalid_limit)
    uint32_t verify_invalid_windows = 0;
    double staged_verify_hop_per_query = 0.0, verify_hop_per_query = 0.0;
    double staged_verify_recall_percent = std::numeric_limits<double>::quiet_NaN();
    double verify_recall_percent = std::numeric_limits<double>::quiet_NaN();
    WindowRankCostSnapshot staged_verify_rank_cost, verify_rank_cost;
    std::vector<WindowBurstAction> last_committed_actions;
    WindowRepairState repair_state;  // the planner works on copies
    std::ofstream journal;
};

// Starts a controller phase over node_count ids and opens the journal. Returns nullptr on bad input.
WindowControllerRuntime* window_controller_start(
    const WindowControllerConfig& config, size_t node_count, std::string* error);
// 4 + 4 x row sum + vector_bytes per node
std::vector<uint64_t> window_controller_histogram_row_bytes(const std::vector<uint32_t>& owner_hist,
                                                            uint32_t owners, uint32_t vector_bytes);
// Binds the phase's row bytes at the first snapshot. Later snapshots must give the same bytes.
bool window_controller_bind_row_bytes(WindowControllerRuntime& runtime, std::vector<uint64_t> row_bytes,
                                      std::string* error);
// Fails while a batch is unresolved or the stats windows are not confirmed closed.
bool window_controller_stop();
// Ends a stopped controller once ownership is known. Drops the action history.
bool window_controller_abort_allowed(const WindowControllerRuntime& runtime, std::string* refusal);
bool window_controller_abort(std::string* refusal);
// Resumes dispatch after a manual migration-accept or migration-rollback confirmed ownership.
void window_controller_confirm_ownership(uint64_t ownership_epoch, const char* command);
// Drops the action history (reset). Fails while a controller is active.
bool window_controller_reset_history();
WindowControllerRuntime* window_controller_active();
bool window_controller_has_unresolved_action(const WindowControllerRuntime& runtime);
bool window_controller_dispatch_frozen();
void window_controller_tick(const ControllerTickInputs& inputs);
// controller-status JSON body (no braces)
std::string window_controller_status_json(const WindowControllerRuntime& runtime);
// Admits a planned batch under the byte caps, writes its plan and pauses dispatch.
void window_controller_apply_proposal(WindowControllerRuntime& runtime, std::vector<WindowBurstAction> actions,
                                      const std::string& propose_fields, std::chrono::steady_clock::time_point now);

double window_controller_skew(const std::vector<uint64_t>& per_rank_w);
// plan with empty rows (the refresh exchange fills in the current rows)
bool window_controller_skeleton_plan(const std::vector<WindowBurstAction>& actions,
    uint64_t epoch_from, uint32_t n_servers, std::vector<char>* out, std::string* error);
// swaps source and target of every action (the revert plan)
std::vector<WindowBurstAction> window_controller_inverse_actions(
    const std::vector<WindowBurstAction>& actions);
void window_controller_arm_exchange(WindowControllerRuntime& runtime, WindowExchangeKind kind);
bool window_controller_is_ack(const void* data, size_t size);
// only an ACK of the pending exchange and window epoch changes state
bool window_controller_offer_ack(WindowControllerRuntime& runtime, uint32_t server,
                                 const void* data, size_t size);
inline bool window_controller_exchange_complete(const WindowControllerRuntime& runtime) {
    return runtime.exchange != WindowExchangeKind::NONE && !runtime.exchange_rejected &&
           runtime.exchange_ack_count == runtime.config.ranks;
}
bool window_controller_observation_ready(const WindowControllerRuntime& runtime,
    const ControllerTickInputs& inputs, double elapsed_seconds);
// same command, monotone counters, enough queries, nonzero work on every rank
bool window_controller_measurement_valid(const WindowControllerRuntime& runtime,
    const ControllerTickInputs& inputs);
bool window_controller_rank_snapshot(const WindowControllerRuntime& runtime,
    uint64_t queries, WindowRankCostSnapshot* out);
// Recall@K in percent, NaN unless every completed query was scored
double window_controller_window_recall(const WindowControllerRuntime& runtime,
    const ControllerTickInputs& inputs);
// Per-rank costs (us per query) of `measured` at the prices of `prices` (the trigger window).
bool window_controller_rank_costs(const WindowRankCostSnapshot& prices,
    const WindowRankCostSnapshot& measured, std::vector<double>* costs);
// Snapshot exchange deadline: 25 s x max(1, K / 4).
double window_controller_exchange_deadline_sec(uint32_t ranks);
// one pooled price fixed at the first planned window, and the peak at that price
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
