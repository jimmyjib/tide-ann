// Byte accounting of every MOVE the repair controller makes. One history is
// kept per coordinator process and survives controller-stop/start as long as
// the controller configuration (its context hash: the configuration only, not
// the row bytes, which change with the inserts between controller phases) is
// unchanged; a coordinator reset or a controller-abort drops it
// (window_controller_reset_history, window_controller_abort).
//
// A controller PHASE is one controller-start .. controller-stop (inserts
// and the controller alternate). Each phase binds the row bytes of its first
// merged harvest (window_history_bind_phase_bytes) and the caps below are
// fractions of that phase's total bytes, reset at every start:
//  - total cap: forward plus inverse bytes within the phase
//    (window_repair_total_cap: 2 * cum * total row bytes);
//  - rate cap: forward plus inverse bytes within a rolling window of
//    kHistoryRateWindows (four) controller periods, period = window + cooldown
//    seconds (window_repair_move_rate_cap);
//  - revert reservation: every admitted forward burst reserves the same number
//    of bytes for the inverse MOVE that a rejected verdict needs, so admission
//    spends at most half of what either cap has left;
//  - rejection memory, counted in PLANNING WINDOWS:
//    the plans the controller completed under this history
//    (window_history_count_planning_window), never in seconds. A measured
//    window cycle can last several configured periods, so a wall-clock memory
//    could lapse before the next planning window. Every resolved burst puts
//    its nodes on a node cooldown for the next two planning windows; a
//    rejected burst (and a burst cancelled before COMMIT) backs off each
//    (node, source -> target) direction for the next 2^failures planning
//    windows, failures capped at 3 (2, 4, 8); an accepted burst clears the
//    backoff of its directions. Planning excludes
//    cooled-down nodes and backed-off directions (window_history_eligibility).
//    The planner's own window-counted cooldown is not advanced by a verify
//    window, so without this memory the next planning window would re-propose
//    a rejected burst; with it, a rejected burst cannot be re-proposed at the
//    next planning window, whatever the window and cycle durations.
// Kept across phases: the ownership epoch, the window epoch and the runtime
// counters, the rejection memory, and the planner's per-node state (its EMA of
// the per-query weight, its window-indexed cooldowns and window counter; ids
// inserted since are padded at the next plan, window_repair_calibrate). Reset
// per phase: the caps and the bytes spent against them, the rate debits, the
// frozen prices and the cumulative guard (a new runtime), the proposal limit
// and a cap-exhausted stop (proposals_disabled).
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "window_controller.h"

namespace tide {
struct WindowControllerRuntime;
using WindowHistoryClock = std::chrono::steady_clock;

// Rejection memory in planning windows: an entry applies while the history's
// planning_windows count is below `until_plan`.
constexpr uint64_t kHistoryNodeCooldownPlans = 2;
constexpr uint32_t kHistoryBackoffMaxFailures = 3;  // 2, 4, 8 planning windows

struct WindowDirectionBackoff {
    uint32_t failures = 0;
    uint64_t until_plan = 0;
};

struct WindowActionHistory {
    std::string context_sha256;
    // total_spent and the caps are the current phase's; lifetime_spent sums every phase.
    uint64_t expected_epoch = 0, total_spent = 0, total_cap = 0, move_rate_cap = 0, lifetime_spent = 0;
    uint32_t phases = 0;  // controller phases bound to this history (1 for the first)
    bool phase_bytes_bound = false;  // the current phase's row bytes (and caps) are bound
    double period_s = 0;  // seconds; the rate cap's unit only
    std::deque<std::pair<WindowHistoryClock::time_point, uint64_t>> move_debits;
    // Planning windows completed under this history: the rejection memory's clock.
    uint64_t planning_windows = 0;
    // Rejection memory: node -> planning-window count at which its cooldown
    // ends; window_history_direction_key (node, source, target) -> backoff.
    std::map<uint32_t, uint64_t> move_cooldowns;
    std::map<uint64_t, WindowDirectionBackoff> direction_backoffs;
    // The one admitted burst between admission and its verdict.
    std::vector<WindowBurstAction> pending;
    uint64_t pending_bytes = 0, reverse_reserved = 0;
    bool committed = false;
    // Runtime counters restored by the next controller-start of the same context.
    uint64_t saved_window_epoch = 0;
    uint32_t saved_cycles = 0, saved_commits = 0, saved_actions = 0, saved_reverts = 0;
    WindowHistoryClock::time_point saved_cooldown;
    // The planner's per-node state carried into the next phase.
    std::vector<double> saved_pred;
    std::vector<uint32_t> saved_cooldown_until;
    uint32_t saved_repair_windows = 0;
};

// Per node, bit p of the uint64 set: the planner may move the node to owner p
// (K <= 64). Empty when no cooldown or backoff applies (the planner is then
// unrestricted).
struct WindowHistoryEligibility {
    std::vector<uint64_t> move_targets;
    uint64_t cooldown_nodes = 0, backoff_directions = 0;
    uint64_t planning_window = 0;  // 1-based index of the planning window it serves
};

enum class WindowHistoryAdmission { Admitted, RateWait, TotalCap, Invalid };
struct WindowHistoryBudget { uint64_t total_left = 0, rate_left = 0; };

// Binds the runtime to the retained history (creating it at ownership epoch 0)
// and starts a new phase: the caps and the phase's spent bytes and rate debits
// reset (the caps are set when the phase's row bytes bind), the saved runtime
// counters and planner state are restored. Refuses another context, another
// ownership epoch, or a history with an unresolved burst.
bool window_history_bind(WindowControllerRuntime& runtime,
    std::shared_ptr<WindowActionHistory>& retained, uint64_t epoch, std::string& error);
// The phase's row bytes are bound (runtime.row_bytes / total_bytes, from the
// first merged harvest): sets the phase's total and rate caps from total_bytes.
// False when the history is missing or the phase's bytes are bound already.
bool window_history_bind_phase_bytes(WindowControllerRuntime& runtime);
void window_history_save(WindowControllerRuntime& runtime);
// Key of one MOVE direction: node << 32 | source << 16 | target. Injective for
// every owner below 65536, so two directions never share a key at any K (two
// bits per owner would collide from K = 5 on).
uint64_t window_history_direction_key(uint32_t node, uint32_t source, uint32_t target);
// Moves the rejection memory allows from `membership` at the next planning
// window (planning_windows + 1); expired node cooldowns are pruned. Time plays
// no part: only completed planning windows age the memory.
WindowHistoryEligibility window_history_eligibility(WindowControllerRuntime& runtime,
    const std::vector<uint32_t>& membership);
// One planning window completed (the controller installed its plan, with or
// without actions): the rejection memory ages by one planning window.
void window_history_count_planning_window(WindowControllerRuntime& runtime);
WindowHistoryBudget window_history_budget(const WindowControllerRuntime& runtime,
    WindowHistoryClock::time_point now);
// Truncates `actions` to the longest prefix both budgets admit and records it as pending.
WindowHistoryAdmission window_history_admit(WindowControllerRuntime& runtime,
    std::vector<WindowBurstAction>& actions, uint64_t epoch,
    WindowHistoryClock::time_point now, std::string& error);
// admit + journal; false when nothing was admitted (the runtime may be disabled).
bool window_history_limit_actions(WindowControllerRuntime& runtime,
    std::vector<WindowBurstAction>& actions, uint64_t epoch,
    WindowHistoryClock::time_point now, const std::string& fields, uint64_t& kept);
bool window_history_committed(WindowControllerRuntime& runtime, uint64_t epoch,
                              WindowHistoryClock::time_point now);
// accepted: the committed burst stays; otherwise the inverse burst (now in
// runtime.pending_actions) has committed and consumes the reservation. Either
// way the burst's nodes enter the node cooldown (two planning windows); a
// rejection backs off its directions, an acceptance clears their backoff.
bool window_history_resolved(WindowControllerRuntime& runtime, bool accepted,
                             uint64_t epoch, WindowHistoryClock::time_point now);
// Releases an admitted burst that never committed and backs off its directions.
void window_history_cancel(WindowControllerRuntime& runtime, WindowHistoryClock::time_point now,
                           bool prepare_failed = false);
std::string window_history_json(const WindowControllerRuntime& runtime,
                                WindowHistoryClock::time_point now);
}  // namespace tide
