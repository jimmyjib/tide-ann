// Coordinator "local loop" repair planner: calibrate, refine and balance
// (greedy balance under a byte budget) on the per-node counters the servers
// ship in the window snapshot. The coordinator
// never holds the adjacency: every decision uses only w (window work),
// nbwork (neighbour-aggregated work, weight WNB only), the owner histogram rows, membership,
// row bytes and the frozen per-rank prices. Every function here is pure or
// operates on the explicit POD state it receives.
//
// K servers: every function takes the owner count `owners` (K, the coordinator's
// n_servers) explicitly; per-owner loads are vectors of size K and the owner
// histogram has K columns per node (row v at owner_hist[v * K]). 2 <= K <=
// kWindowMaxOwners (64, window_snapshot_wire.h: the per-node target mask
// allowed_targets is one uint64); K = 1 and K > 64 are refused by every function
// here (window_owners_valid; refine needs a best OTHER owner). At K = 4 a
// golden file pins the planner's output byte for byte (golden test).
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "window_controller.h"

namespace tide {

// The action history's rolling move-rate window, in controller periods
// (window + cooldown seconds). Unrelated to the owner count.
inline constexpr uint32_t kHistoryRateWindows = 4;

// Destination rule of the balance step (token balance_target:...).
//  LIGHTEST:    every move goes from the heaviest owner to the single lightest
//               owner; the walk stops as soon as the lightest owner has no
//               admissible node on the heaviest one. The default at every K.
//  UNDERLOADED: every move goes from the heaviest owner to an owner below the
//               mean load; the walk continues while any underloaded owner
//               qualifies (rule 3b at window_repair_balance). Opt-in only.
enum class WindowRepairBalanceTarget : uint8_t { LIGHTEST = 1, UNDERLOADED = 2 };
const char* window_repair_balance_target_name(WindowRepairBalanceTarget target);

// Per-node weight the planner calibrates (token weight:w, the default, or weight:wnb).
//  W:   W(v), pred(v) = work[v] / Q: the distance evaluations spent expanding v on
//       its owner during the window (exact + ADC + re-rank). The harvest carries
//       the owner histogram only (kWindowStructureOwnerHist); nb_work is empty.
//  WNB: W^nb(v), pred(v) = (work[v] + nb_work[v]) / Q with nb_work[v] the work of v's
//       in-neighbours; the planner golden output (testdata/repair_golden_k4.json) pins
//       it. The harvest also carries the u64 nbwork section of every node id from every
//       rank, about 3.7x the W harvest bytes at K = 4.
enum class WindowRepairWeight : uint8_t { W = 1, WNB = 2 };
const char* window_repair_weight_name(WindowRepairWeight weight);  // "w" / "wnb"

struct WindowRepairConfig {
    bool enabled = false;
    WindowRepairWeight weight = WindowRepairWeight::W;
    double refine_budget = 0.01;     // fraction of total row bytes per window
    double balance_budget = 0.004;   // fraction of total row bytes per window
    double tau = 0.2;                // allow[v][p] = hist[v][p] >= tau * deg(v); deg 0 never
    double eps = 0.03;               // refine: W[target] + w_v <= (1+eps) * mean(W)
    double skew_trigger = 1.05;      // balance only when max(W)/mean(W) > skew_trigger
    uint32_t cooldown_windows = 2;
    double cumulative_cap = 0.45;    // forward bytes over the whole repair, fraction of total
    double slack = 0.03;             // verify: peak and total rank-time ratio <= 1 + slack
    double hop_slack = 0.02;         // verify: hop_per_query ratio <= 1 + hop_slack
    // Optional quality clause (token recall_guard:off, the default, or recall_guard:<pp>):
    // the verify window's recall >= the trigger window's recall - recall_guard_pp. Both
    // recalls are live Recall@K against ground truth the search command supplies; without
    // the guard the verdict rests on the peak, total and hop ratios alone. Fails closed:
    // a window's recall exists only when every completed query of it was scored
    // (window_controller_window_recall), so a guarded controller under a search command
    // without ground truth plans nothing (recall_guard_unscored) and rejects a pending
    // burst whose verify window is unscored.
    // The reference is NOT paired: the trigger window (the one the burst was planned
    // from) and the verify window score DIFFERENT queries of the stream, so a change in
    // the query mix between the two windows enters the clause as if the burst had caused
    // it; a paired reference cannot be rebuilt from the rolling recall sums the
    // controller receives. Choose recall_guard_pp above the window-to-window recall
    // spread of the search load.
    bool recall_guard = false;
    double recall_guard_pp = 0.0;
    double ema = 0.5;                // weight smoothing across windows
    uint32_t converge_windows = 3;   // quiet windows before repair_converged
    double guard = 0.10;             // abort: last converge_windows verified peaks all > (1+guard) * first
    WindowRepairBalanceTarget balance_target = WindowRepairBalanceTarget::LIGHTEST;
    // A committed burst whose verify windows cannot be measured (e.g. a rank
    // served nothing, so its per-rank prices are undefined, or the harvest
    // failed) is resolved as REJECTED and reverted after this many consecutive
    // such windows, so a pending verdict never blocks controller-stop forever.
    uint32_t verify_invalid_limit = 3;
};

enum class WindowRepairPhase : uint8_t { IDLE = 0, ACTIVE = 1, CONVERGED = 2, ABORTED = 3 };
const char* window_repair_phase_name(WindowRepairPhase phase);

struct WindowRepairState {
    WindowRepairPhase phase = WindowRepairPhase::IDLE;
    std::vector<double> pred;              // per node: EMA of per-query neighbour-aggregated work
    std::vector<uint32_t> cooldown_until;  // window index
    // Per rank c_d (us per distance) and c_r (us per request), frozen at the
    // first repair window; 1.0 / 0.0 when unavailable. Prices scale W and
    // the cumulative guard's rank time; the verify gate prices with the
    // staged trigger window exactly like the existing gate.
    std::vector<double> price, price_r;
    bool price_frozen = false;
    uint32_t windows = 0, quiet_windows = 0, bursts = 0, reverts = 0;
    uint64_t cumulative_bytes = 0;         // forward committed repair bytes
    double first_peak_us = std::numeric_limits<double>::quiet_NaN();
    double skew = std::numeric_limits<double>::quiet_NaN();  // last calibrated skew
    std::vector<double> verified_peaks;    // last converge_windows frozen-priced verified peaks
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

// One window's pure planning output. `planned` is false when the phase
// admits calibration only (CONVERGED below the trigger, or ABORTED).
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

// The snapshot structure sections the controller requests and the planner
// consumes: kWindowStructureOwnerHist | (weight WNB ? kWindowStructureNbWork : 0).
// The single source of the section mask (harvest request, frame decode, merge,
// planning-input check).
uint32_t window_repair_structure_sections(const WindowRepairConfig& config);

// max(W) / mean(W) over the W.size() owners; 1.0 for an empty or zero W.
double window_repair_skew(const std::vector<double>& W);

// q(v) = work[v] / queries under weight W (nb_work must be empty), or
// (work[v] + nb_work[v]) / queries under WNB (nb_work has one entry per node);
// pred(v) = ema * q + (1 - ema) * pred(v) for every v the previous pred covers
// (v < pred.size()), q for the others: the whole table on the first
// calibration (pred empty), the ids inserted since the previous controller
// phase otherwise (padded, not a reset). W[p] = price[p] * sum of pred over
// the nodes p owns. Returns false (and leaves pred untouched) on a shape/domain
// mismatch, including a pred longer than the membership.
bool window_repair_calibrate(const WindowRepairConfig& config, uint32_t owners,
                             const std::vector<uint64_t>& work,
                             const std::vector<uint64_t>& nb_work,
                             uint64_t queries,
                             const std::vector<uint32_t>& membership,
                             const std::vector<double>& price,
                             std::vector<double>& pred,
                             std::vector<double>& W,
                             WindowRepairCalibration* out, std::string* error);

// `allowed_targets` (optional, every rule below): per node, bit p of the uint64
// set means the node may move to owner p (K <= 64). The controller passes the action history's
// rejection memory (window_history_eligibility); null leaves every target
// open, which is the behaviour the golden output pins.
//
// Rule 2. Candidates: gain = h[v][q] - h[v][own] > 0 with q the best other
// owner (lowest index on ties), off cooldown, not moved this window, q
// allowed; walked in (gain desc, pred desc, id asc) order. Stops at the byte
// budget, rejects (counted) a move that would lift W[q] above (1 + eps) *
// mean(W) computed once before the walk. Gains are not re-derived after moves.
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

// Rule 3, the balance step. Common to both destination rules: priced weights
// (price[owner] * pred), an edge-admissible allow mask (allow[v][p]: hist[v][p]
// >= tau * deg(v) with deg(v) the row sum, deg 0 never, the target bit of
// allowed_targets set), nodes on cooldown or moved earlier this window never
// admissible; candidates are the nodes with pred > 0, listed per owner (the
// membership at entry) in (pred desc, id asc) order; each node moves at most
// once; a move debits price[source] * pred and credits price[target] * pred;
// the walk stops once the moved bytes reach budget_bytes (checked after each
// move). Runs only when max(W)/mean(W) > skew_trigger and budget_bytes > 0.
// The rule is config.balance_target (LIGHTEST unless the token opts in to UNDERLOADED):
//
// Rule 3a, LIGHTEST (the greedy balance step): each step takes the
// heaviest owner P (first max) and the lightest owner L (first min), gap =
// W[P] - W[L]; the candidates of P whose weight is >= gap are dropped for good;
// among the remaining ones admissible toward L, the first whose weight is <=
// gap / 2 moves, else the first (largest improving) one; none: stop. At large K
// the walk therefore stops as soon as the single lightest owner has no
// admissible node on the heaviest.
//
// Rule 3b, UNDERLOADED: each step takes the heaviest owner P (first max), mean
// = mean(W) and limit = (1 + eps) * mean. A destination q of node v is any owner
// q != P with allow[v][q], W[q] < mean, W[q] + price[q] * pred(v) <= limit and
// W[q] + price[q] * pred(v) < W[P]; among them v goes to the one holding most
// of v's out-neighbours (largest hist[v][q]), ties to the lower W[q], then the
// lower index. The first candidate of P (in list order) that has a
// destination moves; none: stop. Stops as well when W[P] <= mean.
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

// Whole-window planning: calibrate, then (phase permitting) refine + balance
// on private copies. `pred`/`cooldown_until` are the state before the window;
// either may be shorter than the membership (ids inserted since the previous
// phase: pred padded by calibration, cooldowns padded with 0), never longer;
// `allowed_targets` (optional, size nodes) restricts the targets of both rules.
// `nb_work` is empty under weight W and has one entry per node under WNB.
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

// Non-regression gate on frozen-priced per-rank costs (us per query):
// accept iff hop ratio <= 1 + hop_slack, peak ratio <= 1 + slack, total ratio
// <= 1 + slack and, with recall_guard on, recall and reference (the trigger
// window's recall) are finite with recall >= reference - recall_guard_pp. A zero
// or non-finite baseline, or cost vectors that are not both of size `owners`,
// fail closed.
WindowRepairVerdict window_repair_verdict(const WindowRepairConfig& config, uint32_t owners,
                                          const std::vector<double>& baseline_cost_us,
                                          const std::vector<double>& measured_cost_us,
                                          double baseline_hop, double measured_hop,
                                          double recall, double reference);

// Phase bookkeeping after one planned window. `issued` is whether the planner
// produced any action; `skew` the window's calibrated skew.
WindowRepairTransition window_repair_account_window(const WindowRepairConfig& config,
                                                    WindowRepairState& state,
                                                    bool issued, double skew);
// Cumulative guard: records one verified window's frozen-priced peak and
// returns true when the last converge_windows of them all exceed
// (1 + guard) * first_peak_us. The caller moves the phase to ABORTED.
bool window_repair_guard_tripped(const WindowRepairConfig& config, WindowRepairState& state,
                                 double verified_peak_us);

// History caps in repair mode: total_cap = 2 * cum * total_bytes in gross (forward +
// inverse) bytes, cum <= 1, so at most twice the table; the /2 inverse reservation at
// admission leaves cum forward bytes;
// move_rate_cap = 2 * kHistoryRateWindows * (refine + balance) * total_bytes.
uint64_t window_repair_total_cap(const WindowRepairConfig& config, uint64_t total_bytes);
uint64_t window_repair_move_rate_cap(const WindowRepairConfig& config, uint64_t total_bytes);
std::string window_repair_identity_tag(const WindowRepairConfig& config);

// `repair=` token body: algo:local[,key:value...] with keys weight (w, the
// default, or wnb; parsed before any numeric value), refine, balance, tau, eps,
// skew, cooldown, cum, slack, hop_slack, recall_guard, ema, converge, guard,
// balance_target (lightest, the default, or underloaded) and
// verify_invalid_limit (integer >= 1). Duplicate/unknown keys, a missing or
// non-local algo, or an out-of-domain value fail with a message.
bool window_repair_parse_token(const std::string& body, WindowRepairConfig* out, std::string* error);

std::string window_repair_config_json(const WindowRepairConfig& config);     // ,"repair_config":{...}
std::string window_repair_state_json(const WindowRepairState& state);        // ,"repair":{...}
std::string window_repair_calibration_json(const WindowRepairCalibration& c, const std::vector<double>& price);
std::string window_repair_stats_json(const WindowRepairBurstStats& stats);

}  // namespace tide
