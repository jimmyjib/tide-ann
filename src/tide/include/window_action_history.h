// Byte caps and rejection memory for controller moves. One history per coordinator process, dropped by
// reset and controller-abort.
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

// Rejection memory counts planning windows: an entry applies while planning_windows < until_plan.
constexpr uint64_t kHistoryNodeCooldownPlans = 2;
constexpr uint32_t kHistoryBackoffMaxFailures = 3;  // 2, 4, 8 planning windows

struct WindowDirectionBackoff {
    uint32_t failures = 0;
    uint64_t until_plan = 0;
};

struct WindowActionHistory {
    std::string context_sha256;
    // the bound settings by name, to say which one a later controller-start changes
    std::vector<std::pair<std::string, std::string>> settings;
    // total_spent and the caps are the current phase's, and lifetime_spent sums every phase.
    uint64_t expected_epoch = 0, total_spent = 0, total_cap = 0, move_rate_cap = 0, lifetime_spent = 0;
    uint32_t phases = 0;  // controller phases bound to this history (1 for the first)
    bool phase_bytes_bound = false;  // the current phase's row bytes (and caps) are bound
    double period_s = 0;  // seconds, the rate cap's unit only
    std::deque<std::pair<WindowHistoryClock::time_point, uint64_t>> move_debits;
    // planning windows completed under this history
    uint64_t planning_windows = 0;
    // node -> end of its cooldown
    std::map<uint32_t, uint64_t> move_cooldowns;
    // direction key -> backoff
    std::map<uint64_t, WindowDirectionBackoff> direction_backoffs;
    // the allowed batch until it is checked
    std::vector<WindowBurstAction> pending;
    uint64_t pending_bytes = 0, reverse_reserved = 0;
    bool committed = false;
    // restored by the next controller-start with the same configuration
    uint64_t saved_window_epoch = 0;
    uint32_t saved_cycles = 0, saved_commits = 0, saved_actions = 0, saved_reverts = 0;
    WindowHistoryClock::time_point saved_cooldown;
    // planner state carried into the next phase
    std::vector<double> saved_pred;
    std::vector<uint32_t> saved_cooldown_until;
    uint32_t saved_repair_windows = 0;
};

// Per node, bit p set: the node may move to owner p. Empty when nothing is restricted.
struct WindowHistoryEligibility {
    std::vector<uint64_t> move_targets;
    uint64_t cooldown_nodes = 0, backoff_directions = 0;
    uint64_t planning_window = 0;  // 1-based index of the planning window it serves
};

enum class WindowHistoryAdmission { Admitted, RateWait, TotalCap, Invalid };
struct WindowHistoryBudget { uint64_t total_left = 0, rate_left = 0; };

// Starts a phase on the retained history (created at ownership epoch 0). Fails on another configuration,
// another ownership epoch or an unresolved batch.
bool window_history_bind(WindowControllerRuntime& runtime,
    std::shared_ptr<WindowActionHistory>& retained, uint64_t epoch, std::string& error);
// Sets the phase's caps from its row bytes. False when already bound.
bool window_history_bind_phase_bytes(WindowControllerRuntime& runtime);
void window_history_save(WindowControllerRuntime& runtime);
// node << 32 | source << 16 | target
uint64_t window_history_direction_key(uint32_t node, uint32_t source, uint32_t target);
// Moves allowed at the next planning window. Expired cooldowns are pruned.
WindowHistoryEligibility window_history_eligibility(WindowControllerRuntime& runtime,
    const std::vector<uint32_t>& membership);
// Ages the rejection memory by one planning window.
void window_history_count_planning_window(WindowControllerRuntime& runtime);
WindowHistoryBudget window_history_budget(const WindowControllerRuntime& runtime,
    WindowHistoryClock::time_point now);
// Truncates `actions` to the longest prefix both budgets allow and records it as pending.
WindowHistoryAdmission window_history_admit(WindowControllerRuntime& runtime,
    std::vector<WindowBurstAction>& actions, uint64_t epoch,
    WindowHistoryClock::time_point now, std::string& error);
// window_history_admit plus the journal line. False when nothing was allowed.
bool window_history_limit_actions(WindowControllerRuntime& runtime,
    std::vector<WindowBurstAction>& actions, uint64_t epoch,
    WindowHistoryClock::time_point now, const std::string& fields, uint64_t& kept);
bool window_history_committed(WindowControllerRuntime& runtime, uint64_t epoch,
                              WindowHistoryClock::time_point now);
// Kept batch: its directions' backoff is cleared. Reverted batch: its directions back off. The nodes
// cool down for two planning windows either way.
bool window_history_resolved(WindowControllerRuntime& runtime, bool accepted,
                             uint64_t epoch, WindowHistoryClock::time_point now);
// Releases an allowed batch that never committed and backs off its directions.
void window_history_cancel(WindowControllerRuntime& runtime, WindowHistoryClock::time_point now,
                           bool prepare_failed = false);
std::string window_history_json(const WindowControllerRuntime& runtime,
                                WindowHistoryClock::time_point now);
}  // namespace tide
