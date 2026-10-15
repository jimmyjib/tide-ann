// Repair planner: calibrate per-node weights, refine by neighbor ownership, balance load under a byte
// budget. Pure functions, 2 <= K <= 64.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "window_controller.h"

namespace tide {

// Rolling move-rate window of the action history, in controller periods.
inline constexpr uint32_t kHistoryRateWindows = 4;

// Balance destination (token balance_target): the lightest owner (default) or any owner below the mean.
enum class WindowRepairBalanceTarget : uint8_t { LIGHTEST = 1, UNDERLOADED = 2 };
const char* window_repair_balance_target_name(WindowRepairBalanceTarget target);

// Per-node weight (token weight): W(v) = work[v] / Q, or WNB = (work[v] + in-neighbor work) / Q.
enum class WindowRepairWeight : uint8_t { W = 1, WNB = 2 };
const char* window_repair_weight_name(WindowRepairWeight weight);  // "w" / "wnb"

struct WindowRepairConfig {
    bool enabled = false;
    WindowRepairWeight weight = WindowRepairWeight::W;
    double refine_budget = 0.01;     // fraction of total row bytes per window
    double balance_budget = 0.004;   // fraction of total row bytes per window
    double tau = 0.2;                // allow[v][p] = hist[v][p] >= tau * deg(v), never for deg 0
    double eps = 0.03;               // refine: W[target] + w_v <= (1+eps) * mean(W)
    double skew_trigger = 1.05;      // balance only when max(W)/mean(W) > skew_trigger
    uint32_t cooldown_windows = 2;
    double cumulative_cap = 0.45;    // forward bytes over the whole repair, fraction of total
    double slack = 0.03;             // check: peak and total rank-time ratio <= 1 + slack
    double hop_slack = 0.02;         // check: hop_per_query ratio <= 1 + hop_slack
    // Optional: reject when verify-window recall drops more than recall_guard_pp below the trigger
    // window's (different queries, so set it above window-to-window noise).
    bool recall_guard = false;
    double recall_guard_pp = 0.0;
    double ema = 0.5;                // weight smoothing across windows
    uint32_t converge_windows = 3;   // quiet windows before repair_converged
    double guard = 0.10;             // abort: last converge_windows checked peaks all > (1+guard) * first
    WindowRepairBalanceTarget balance_target = WindowRepairBalanceTarget::LIGHTEST;
    // Consecutive windows that cannot check a pending batch before it is reverted.
    uint32_t verify_invalid_limit = 3;
};

enum class WindowRepairPhase : uint8_t { IDLE = 0, ACTIVE = 1, CONVERGED = 2, ABORTED = 3 };
const char* window_repair_phase_name(WindowRepairPhase phase);

struct WindowRepairState {
    WindowRepairPhase phase = WindowRepairPhase::IDLE;
    std::vector<double> pred;              // per node: EMA of per-query work
    std::vector<uint32_t> cooldown_until;  // window index
    // Per rank c_d (us per distance) and c_r (us per request), fixed at the first window, or 1.0 / 0.0
    // when unavailable.
    std::vector<double> price, price_r;
    bool price_frozen = false;
    uint32_t windows = 0, quiet_windows = 0, bursts = 0, reverts = 0;
    uint64_t cumulative_bytes = 0;         // forward committed repair bytes
    double first_peak_us = std::numeric_limits<double>::quiet_NaN();
    double skew = std::numeric_limits<double>::quiet_NaN();  // last calibrated skew
    std::vector<double> verified_peaks;    // last converge_windows checked peaks
};

struct WindowRepairCalibration {
    bool valid = false;
    uint64_t queries = 0, active_nodes = 0;
    std::vector<double> W;               // [K]
    double skew = 0.0;
    std::vector<double> measured_share;  // [K]
};

struct WindowRepairRefineResult {
    uint64_t candidates = 0, moved = 0, bytes = 0, rejected_balance = 0;
};
struct WindowRepairBalanceResult {
    bool triggered = false;
    uint64_t moved = 0, bytes = 0;
    double skew_before = 0.0;
    WindowRepairBalanceTarget target = WindowRepairBalanceTarget::LIGHTEST;  // the rule applied
};
struct WindowRepairBurstStats {
    uint64_t refine_candidates = 0, refine_moved = 0, refine_bytes = 0, rejected_balance = 0;
    bool balance_triggered = false;
    uint64_t balance_moved = 0, balance_bytes = 0;
    double skew_before = 0.0, skew_after = 0.0;
    uint64_t refine_budget_bytes = 0, balance_budget_bytes = 0;
    WindowRepairBalanceTarget balance_target = WindowRepairBalanceTarget::LIGHTEST;  // the rule applied
};

// `planned` is false when the state allows calibration only (CONVERGED below the trigger, or ABORTED).
struct WindowRepairPlan {
    bool valid = false;
    std::string error;
    bool planned = false;
    WindowRepairCalibration calibration;
    WindowRepairBurstStats stats;
    std::vector<double> pred;
    std::vector<uint32_t> cooldown_until;
    std::vector<double> W;  // [K] predicted per-owner loads after the actions
    std::vector<WindowBurstAction> actions;
};

struct WindowRepairVerdict {
    bool accepted = false;
    double peak_ratio = std::numeric_limits<double>::quiet_NaN();
    double total_ratio = std::numeric_limits<double>::quiet_NaN();
    double hop_ratio = std::numeric_limits<double>::quiet_NaN();
};

struct WindowRepairTransition {
    bool converged = false, reactivated = false;
};

// Snapshot sections requested and consumed: owner histogram, plus nbwork under WNB.
uint32_t window_repair_structure_sections(const WindowRepairConfig& config);

// max(W) / mean(W) over the W.size() owners, or 1.0 for an empty or zero W.
double window_repair_skew(const std::vector<double>& W);

// pred(v) = ema * q(v) + (1 - ema) * pred(v), or q(v) for ids pred does not cover yet.
// W[p] = price[p] * sum of pred over the nodes p owns. False (pred untouched) on a shape mismatch.
bool window_repair_calibrate(const WindowRepairConfig& config, uint32_t owners,
                             const std::vector<uint64_t>& work,
                             const std::vector<uint64_t>& nb_work,
                             uint64_t queries,
                             const std::vector<uint32_t>& membership,
                             const std::vector<double>& price,
                             std::vector<double>& pred,
                             std::vector<double>& W,
                             WindowRepairCalibration* out, std::string* error);

// allowed_targets (optional): per node, bit p set means it may move to owner p. Null allows all.
// Refine: move v to the owner holding most of its neighbors if W[q] stays within (1 + eps) * mean(W).
WindowRepairRefineResult window_repair_refine(const WindowRepairConfig& config, uint32_t owners,
                                              const std::vector<uint32_t>& owner_hist,
                                              const std::vector<double>& pred,
                                              const std::vector<double>& price,
                                              const std::vector<uint64_t>& row_bytes,
                                              uint64_t budget_bytes, uint32_t window,
                                              std::vector<uint32_t>& membership,
                                              std::vector<double>& W,
                                              std::vector<uint32_t>& cooldown_until,
                                              std::vector<uint8_t>& moved_now,
                                              std::vector<WindowBurstAction>& actions,
                                              const std::vector<uint64_t>* allowed_targets = nullptr);

// Balance, when max(W)/mean(W) > skew_trigger: move nodes off the busiest owner to allowed targets
// (hist[v][p] >= tau * deg(v)) until budget_bytes, either to the lightest owner or (underloaded) to
// the below-mean owner holding most of v's neighbors.
WindowRepairBalanceResult window_repair_balance(const WindowRepairConfig& config, uint32_t owners,
                                                const std::vector<uint32_t>& owner_hist,
                                                const std::vector<double>& pred,
                                                const std::vector<double>& price,
                                                const std::vector<uint64_t>& row_bytes,
                                                uint64_t budget_bytes, uint32_t window,
                                                std::vector<uint32_t>& membership,
                                                std::vector<double>& W,
                                                std::vector<uint32_t>& cooldown_until,
                                                std::vector<uint8_t>& moved_now,
                                                std::vector<WindowBurstAction>& actions,
                                                const std::vector<uint64_t>* allowed_targets = nullptr);

// Calibrate, then refine and balance on copies. pred and cooldown_until may be shorter than the
// membership (inserted ids), never longer. nb_work is empty under weight W.
WindowRepairPlan window_repair_plan(const WindowRepairConfig& config, uint32_t owners, WindowRepairPhase phase,
                                    uint32_t window, uint64_t queries, uint64_t total_bytes,
                                    const std::vector<uint32_t>& membership,
                                    const std::vector<uint64_t>& row_bytes,
                                    const std::vector<uint64_t>& work,
                                    const std::vector<uint64_t>& nb_work,
                                    const std::vector<uint32_t>& owner_hist,
                                    const std::vector<double>& price,
                                    std::vector<double> pred,
                                    std::vector<uint32_t> cooldown_until,
                                    const std::vector<uint64_t>* allowed_targets = nullptr);

// Keep-or-revert check on per-rank costs (us per query): keep iff the hop, peak and total ratios are
// within their slacks and, with recall_guard, recall >= reference - recall_guard_pp. Bad inputs reject.
WindowRepairVerdict window_repair_verdict(const WindowRepairConfig& config, uint32_t owners,
                                          const std::vector<double>& baseline_cost_us,
                                          const std::vector<double>& measured_cost_us,
                                          double baseline_hop, double measured_hop,
                                          double recall, double reference);

// State update after one planned window. `issued` is whether any action was planned.
WindowRepairTransition window_repair_account_window(const WindowRepairConfig& config,
                                                    WindowRepairState& state,
                                                    bool issued, double skew);
// True when the last converge_windows checked peaks all exceed (1 + guard) * first_peak_us.
bool window_repair_guard_tripped(const WindowRepairConfig& config, WindowRepairState& state,
                                 double verified_peak_us);

// total_cap = 2 * cum * total_bytes (forward plus inverse bytes).
// move_rate_cap = 2 * kHistoryRateWindows * (refine + balance) * total_bytes.
uint64_t window_repair_total_cap(const WindowRepairConfig& config, uint64_t total_bytes);
uint64_t window_repair_move_rate_cap(const WindowRepairConfig& config, uint64_t total_bytes);
std::string window_repair_identity_tag(const WindowRepairConfig& config);

// `repair=` token body: comma-separated key:value pairs, listed in docs/controller.md. `algo:local` is
// accepted and changes nothing, other algo values are rejected.
bool window_repair_parse_token(const std::string& body, WindowRepairConfig* out, std::string* error);

std::string window_repair_config_json(const WindowRepairConfig& config);     // ,"repair_config":{...}
std::string window_repair_state_json(const WindowRepairState& state);        // ,"repair":{...}
std::string window_repair_calibration_json(const WindowRepairCalibration& c, const std::vector<double>& price);
std::string window_repair_stats_json(const WindowRepairBurstStats& stats);

}  // namespace tide
