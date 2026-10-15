#include "window_repair_planner.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace tide {
namespace {
std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return out.str();
}
std::string vector_json(const std::vector<double>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += number(values[i]);
    }
    return out + "]";
}
// Summed in owner order (W[0] + W[1] + ...), one fixed order, so the loads
// (and the K = 4 golden output) stay bit-identical.
double mean_of(const std::vector<double>& W) {
    double sum = 0.0;
    for (const double value : W) sum += value;
    return sum / double(W.size());
}
bool prices_valid(const std::vector<double>& price, uint32_t owners) {
    if (!window_owners_valid(owners) || price.size() != owners) return false;
    for (const double p : price) if (!std::isfinite(p) || p <= 0.0) return false;
    return true;
}
// Best other owner for a histogram row: lowest index wins ties.
uint32_t best_other(const uint32_t* row, uint32_t own, uint32_t owners) {
    uint32_t q = owners, best = 0;
    for (uint32_t p = 0; p < owners; ++p) {
        if (p == own) continue;
        if (q == owners || row[p] > best) { q = p; best = row[p]; }
    }
    return q;
}
}  // namespace

const char* window_repair_balance_target_name(WindowRepairBalanceTarget target) {
    switch (target) {
        case WindowRepairBalanceTarget::LIGHTEST: return "lightest";
        case WindowRepairBalanceTarget::UNDERLOADED: return "underloaded";
    }
    return "invalid";
}

const char* window_repair_weight_name(WindowRepairWeight weight) {
    switch (weight) {
        case WindowRepairWeight::W: return "w";
        case WindowRepairWeight::WNB: return "wnb";
    }
    return "invalid";
}

uint32_t window_repair_structure_sections(const WindowRepairConfig& config) {
    return kWindowStructureOwnerHist | (config.weight == WindowRepairWeight::WNB ? kWindowStructureNbWork : 0u);
}

const char* window_repair_phase_name(WindowRepairPhase phase) {
    switch (phase) {
        case WindowRepairPhase::IDLE: return "IDLE";
        case WindowRepairPhase::ACTIVE: return "ACTIVE";
        case WindowRepairPhase::CONVERGED: return "CONVERGED";
        case WindowRepairPhase::ABORTED: return "ABORTED";
    }
    return "INVALID";
}

double window_repair_skew(const std::vector<double>& W) {
    if (W.empty()) return 1.0;
    const double mean = mean_of(W);
    if (!(mean > 0.0)) return 1.0;  // no measured work: report an even load
    return *std::max_element(W.begin(), W.end()) / mean;
}

bool window_repair_calibrate(const WindowRepairConfig& config, uint32_t owners,
                             const std::vector<uint64_t>& work,
                             const std::vector<uint64_t>& nb_work,
                             uint64_t queries,
                             const std::vector<uint32_t>& membership,
                             const std::vector<double>& price,
                             std::vector<double>& pred,
                             std::vector<double>& W,
                             WindowRepairCalibration* out, std::string* error) {
    auto fail = [&](const char* why) {
        if (error) *error = why;
        if (out) *out = WindowRepairCalibration{};
        return false;
    };
    const size_t nodes = membership.size();
    if (!window_owners_valid(owners)) return fail("repair calibration owner count out of range (2..64)");
    if (config.weight != WindowRepairWeight::W && config.weight != WindowRepairWeight::WNB)
        return fail("repair calibration weight domain");
    const bool wnb = config.weight == WindowRepairWeight::WNB;
    // Weight W takes no nbwork (the harvest does not carry the section); WNB needs one entry per node.
    if (!nodes || work.size() != nodes || nb_work.size() != (wnb ? nodes : 0))
        return fail("repair calibration shape mismatch");
    if (!queries) return fail("repair calibration requires completed queries");
    if (!prices_valid(price, owners)) return fail("repair calibration requires one positive price per owner");
    if (!(config.ema > 0.0 && config.ema <= 1.0)) return fail("repair calibration ema domain");
    for (const uint32_t owner : membership) if (owner >= owners) return fail("repair calibration membership domain");
    // pred carries the EMA of the ids it covers; ids beyond it (inserted since
    // the previous controller phase) start from their first measurement
    // instead of resetting every node's EMA. Ids never shrink within a history.
    if (pred.size() > nodes) return fail("repair calibration state covers more nodes than the membership");
    const size_t known = pred.size();
    std::vector<double> next(nodes);
    WindowRepairCalibration calibration;
    calibration.queries = queries;
    std::vector<double> measured(owners, 0.0);
    double measured_total = 0.0;
    const double q = double(queries);
    for (size_t v = 0; v < nodes; ++v) {
        // The W^nb expression stays exactly as written: the golden output pins it.
        const double per_query = wnb ? (double(work[v]) + double(nb_work[v])) / q : double(work[v]) / q;
        next[v] = v >= known ? per_query : config.ema * per_query + (1.0 - config.ema) * pred[v];
        if (work[v]) ++calibration.active_nodes;
        measured[membership[v]] += double(work[v]);
        measured_total += double(work[v]);
    }
    std::vector<double> sums(owners, 0.0);
    for (size_t v = 0; v < nodes; ++v) sums[membership[v]] += next[v];
    W.assign(owners, 0.0);
    calibration.W.assign(owners, 0.0);
    calibration.measured_share.assign(owners, 0.0);
    for (uint32_t p = 0; p < owners; ++p) {
        W[p] = price[p] * sums[p];
        calibration.W[p] = W[p];
        calibration.measured_share[p] = measured_total > 0.0 ? measured[p] / measured_total : 0.0;
    }
    calibration.skew = window_repair_skew(W);
    calibration.valid = true;
    pred = std::move(next);
    if (out) *out = calibration;
    if (error) error->clear();
    return true;
}

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
                                              const std::vector<uint64_t>* allowed_targets) {
    WindowRepairRefineResult result;
    const size_t nodes = membership.size();
    if (!nodes || !prices_valid(price, owners) || owner_hist.size() != nodes * owners || pred.size() != nodes ||
        row_bytes.size() != nodes || cooldown_until.size() != nodes || moved_now.size() != nodes ||
        W.size() != owners || (allowed_targets && allowed_targets->size() != nodes)) return result;
    struct Candidate { uint32_t node, target; int64_t gain; };
    std::vector<Candidate> candidates;
    for (size_t v = 0; v < nodes; ++v) {
        const uint32_t own = membership[v];
        if (own >= owners) continue;
        const uint32_t* row = owner_hist.data() + v * owners;
        const uint32_t q = best_other(row, own, owners);
        const int64_t gain = int64_t(row[q]) - int64_t(row[own]);
        if (gain <= 0 || cooldown_until[v] > window || moved_now[v]) continue;
        if (allowed_targets && !(((*allowed_targets)[v] >> q) & 1u)) continue;
        candidates.push_back({static_cast<uint32_t>(v), q, gain});
    }
    result.candidates = candidates.size();
    std::stable_sort(candidates.begin(), candidates.end(), [&](const Candidate& a, const Candidate& b) {
        if (a.gain != b.gain) return a.gain > b.gain;
        if (pred[a.node] != pred[b.node]) return pred[a.node] > pred[b.node];
        return a.node < b.node;
    });
    const double limit = (1.0 + config.eps) * mean_of(W);
    uint64_t used = 0;
    for (const Candidate& c : candidates) {
        const uint32_t v = c.node;
        if (used + row_bytes[v] > budget_bytes) break;
        const uint32_t own = membership[v], q = c.target;
        const double weight_target = price[q] * pred[v];
        if (W[q] + weight_target > limit) { ++result.rejected_balance; continue; }
        W[own] -= price[own] * pred[v];
        W[q] += weight_target;
        membership[v] = q;
        cooldown_until[v] = window + config.cooldown_windows;
        moved_now[v] = 1;
        used += row_bytes[v];
        ++result.moved;
        WindowBurstAction action;
        action.node = v; action.source_owner = own; action.target_owner = q;
        action.adjacency_bytes = row_bytes[v];
        actions.push_back(action);
    }
    result.bytes = used;
    return result;
}

namespace {
// Rule 3b, UNDERLOADED (window_repair_planner.h). `allow` and `by_owner` are the
// ones window_repair_balance built; W becomes the loads after the walk.
void balance_underloaded(const WindowRepairConfig& config, uint32_t owners,
                         const std::vector<uint32_t>& owner_hist, const std::vector<double>& pred,
                         const std::vector<double>& price, const std::vector<uint64_t>& row_bytes,
                         uint64_t budget_bytes, uint32_t window, std::vector<uint32_t>& membership,
                         std::vector<double>& W, std::vector<uint32_t>& cooldown_until,
                         std::vector<uint8_t>& moved_now, std::vector<WindowBurstAction>& actions,
                         const std::vector<uint8_t>& allow, const std::vector<std::vector<uint32_t>>& by_owner,
                         WindowRepairBalanceResult& result) {
    const size_t nodes = membership.size();
    // A candidate with no admissible owner besides its own can never move.
    std::vector<uint8_t> movable(nodes, 0);
    for (uint32_t p = 0; p < owners; ++p)
        for (const uint32_t v : by_owner[p])
            for (uint32_t q = 0; q < owners && !movable[v]; ++q)
                movable[v] = q != p && allow[size_t(v) * owners + q];
    std::vector<size_t> ptr(owners, 0);
    std::vector<uint8_t> used(nodes, 0);
    std::vector<double> sums = W;
    uint64_t moved_bytes = 0;
    for (;;) {
        const uint32_t peak = uint32_t(std::max_element(sums.begin(), sums.end()) - sums.begin());
        const double mean = mean_of(sums);
        if (!(sums[peak] > mean)) break;
        const double limit = (1.0 + config.eps) * mean;
        const auto& cand = by_owner[peak];
        // Leading entries that already moved or can never move are skipped for good.
        while (ptr[peak] < cand.size() && (used[cand[ptr[peak]]] || !movable[cand[ptr[peak]]])) ++ptr[peak];
        uint32_t v = 0, target = owners;
        for (size_t i = ptr[peak]; i < cand.size() && target == owners; ++i) {
            const uint32_t node = cand[i];
            if (used[node] || !movable[node]) continue;
            const uint32_t* row = owner_hist.data() + size_t(node) * owners;
            for (uint32_t q = 0; q < owners; ++q) {
                if (q == peak || !allow[size_t(node) * owners + q] || !(sums[q] < mean)) continue;
                const double after = sums[q] + price[q] * pred[node];
                if (after > limit || !(after < sums[peak])) continue;
                if (target == owners || row[q] > row[target] || (row[q] == row[target] && sums[q] < sums[target]))
                    target = q;
            }
            if (target != owners) v = node;
        }
        if (target == owners) break;  // no underloaded owner qualifies for any candidate
        used[v] = 1;
        sums[peak] -= price[peak] * pred[v];
        sums[target] += price[target] * pred[v];
        membership[v] = target;
        cooldown_until[v] = window + config.cooldown_windows;
        moved_now[v] = 1;
        moved_bytes += row_bytes[v];
        ++result.moved;
        WindowBurstAction action;
        action.node = v; action.source_owner = peak; action.target_owner = target;
        action.adjacency_bytes = row_bytes[v];
        actions.push_back(action);
        if (moved_bytes >= budget_bytes) break;
    }
    W = sums;
    result.bytes = moved_bytes;
}
}  // namespace

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
                                                const std::vector<uint64_t>* allowed_targets) {
    WindowRepairBalanceResult result;
    const size_t nodes = membership.size();
    if (!nodes || !prices_valid(price, owners) || owner_hist.size() != nodes * owners || pred.size() != nodes ||
        row_bytes.size() != nodes || cooldown_until.size() != nodes || moved_now.size() != nodes ||
        W.size() != owners || (allowed_targets && allowed_targets->size() != nodes)) return result;
    result.target = config.balance_target;
    result.skew_before = window_repair_skew(W);
    if (result.skew_before <= config.skew_trigger || budget_bytes == 0) return result;
    result.triggered = true;
    // allow[v][p]: edge-admissible target; cooldown/moved rows are never admissible.
    std::vector<uint8_t> allow(nodes * owners, 0);
    for (size_t v = 0; v < nodes; ++v) {
        if (cooldown_until[v] > window || moved_now[v]) continue;
        const uint32_t* row = owner_hist.data() + v * owners;
        double deg = 0.0;  // summed in owner order, as row[0] + row[1] + ...
        for (uint32_t p = 0; p < owners; ++p) deg += double(row[p]);
        if (!(deg > 0.0)) continue;
        for (uint32_t p = 0; p < owners; ++p)
            allow[v * owners + p] = double(row[p]) >= config.tau * deg &&
                (!allowed_targets || (((*allowed_targets)[v] >> p) & 1u));
    }
    // Per-owner candidate lists in (pred desc, id asc) order, pred > 0 only,
    // built once from the membership at entry (later moves do not rebuild them).
    std::vector<uint32_t> order;
    order.reserve(nodes);
    for (size_t v = 0; v < nodes; ++v) if (pred[v] > 0.0 && membership[v] < owners) order.push_back(uint32_t(v));
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (pred[a] != pred[b]) return pred[a] > pred[b];
        return a < b;
    });
    std::vector<std::vector<uint32_t>> by_owner(owners);
    for (const uint32_t v : order) by_owner[membership[v]].push_back(v);
    if (result.target == WindowRepairBalanceTarget::UNDERLOADED) {
        balance_underloaded(config, owners, owner_hist, pred, price, row_bytes, budget_bytes, window, membership, W,
                            cooldown_until, moved_now, actions, allow, by_owner, result);
        return result;
    }
    // Rule 3a, LIGHTEST. The allow[.][q] subsequence of every owner list, in the same order, keeping
    // each entry's position in by_owner. Walking it visits exactly the entries a
    // walk of by_owner[peak] from ptr[peak] would accept, without re-reading the
    // inadmissible ones on every move (at 1M nodes almost every peak-owner node is
    // inadmissible toward the lightest owner, which made the plain walk
    // O(moves x candidates)).
    // admissible[p * K + q] / next[p * K + q]: source owner p, target owner q.
    struct Admissible { uint32_t node, pos; };
    std::vector<std::vector<Admissible>> admissible(size_t(owners) * owners);
    for (uint32_t p = 0; p < owners; ++p)
        for (size_t i = 0; i < by_owner[p].size(); ++i) {
            const uint32_t v = by_owner[p][i];
            for (uint32_t q = 0; q < owners; ++q)
                if (q != p && allow[size_t(v) * owners + q]) admissible[size_t(p) * owners + q].push_back({v, uint32_t(i)});
        }
    std::vector<size_t> next(size_t(owners) * owners, 0);
    std::vector<size_t> ptr(owners, 0);
    std::vector<uint8_t> used(nodes, 0);
    std::vector<double> sums = W;
    uint64_t moved_bytes = 0;
    for (;;) {
        const uint32_t peak = uint32_t(std::max_element(sums.begin(), sums.end()) - sums.begin());
        const uint32_t low = uint32_t(std::min_element(sums.begin(), sums.end()) - sums.begin());
        const double gap = sums[peak] - sums[low];
        const auto& cand = by_owner[peak];
        auto weight = [&](uint32_t v) { return price[peak] * pred[v]; };
        // The gap never grows, so nodes at or above it are dropped for good.
        while (ptr[peak] < cand.size() && (used[cand[ptr[peak]]] || weight(cand[ptr[peak]]) >= gap)) ++ptr[peak];
        if (ptr[peak] >= cand.size() || gap <= 0.0) break;
        size_t pick = cand.size(), first_improving = cand.size();
        // Entries before ptr[peak] or already moved never qualify again.
        const auto& list = admissible[size_t(peak) * owners + low];
        size_t& first = next[size_t(peak) * owners + low];
        while (first < list.size() && (list[first].pos < ptr[peak] || used[list[first].node])) ++first;
        if (first < list.size()) {
            first_improving = list[first].pos;                              // heaviest admissible node below the gap
            // Weights do not increase along the list: the first unused entry at
            // or after the first weight <= gap / 2 is the walk's pick.
            size_t lo = first, hi = list.size();
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (weight(list[mid].node) <= 0.5 * gap) hi = mid; else lo = mid + 1;
            }
            while (lo < list.size() && used[list[lo].node]) ++lo;
            if (lo < list.size()) pick = list[lo].pos;
        }
        if (pick == cand.size()) pick = first_improving;                    // largest improving move
        if (pick == cand.size()) break;                                     // no admissible node toward the lightest owner
        const uint32_t v = cand[pick];
        used[v] = 1;
        if (pick == ptr[peak]) ++ptr[peak];
        const double w_peak = weight(v);
        sums[peak] -= w_peak;
        sums[low] += price[low] * pred[v];
        membership[v] = low;
        cooldown_until[v] = window + config.cooldown_windows;
        moved_now[v] = 1;
        moved_bytes += row_bytes[v];
        ++result.moved;
        WindowBurstAction action;
        action.node = v; action.source_owner = peak; action.target_owner = low;
        action.adjacency_bytes = row_bytes[v];
        actions.push_back(action);
        if (moved_bytes >= budget_bytes) break;  // the greedy's one byte checkpoint, after each move
    }
    W = sums;
    result.bytes = moved_bytes;
    return result;
}

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
                                    const std::vector<uint64_t>* allowed_targets) {
    WindowRepairPlan plan;
    const size_t nodes = membership.size();
    if (!window_owners_valid(owners)) {
        plan.error = "repair planning owner count out of range (2..64)"; return plan;
    }
    plan.stats.balance_target = config.balance_target;
    if (!nodes || row_bytes.size() != nodes || owner_hist.size() != nodes * owners ||
        (allowed_targets && allowed_targets->size() != nodes)) {
        plan.error = "repair planning shape mismatch"; return plan;
    }
    // Cooldowns are window-indexed: ids inserted since the previous phase are
    // padded off cooldown, every carried cooldown stays.
    if (cooldown_until.size() > nodes) {
        plan.error = "repair planning cooldowns cover more nodes than the membership"; return plan;
    }
    if (!window_repair_calibrate(config, owners, work, nb_work, queries, membership, price, pred, plan.W,
                                 &plan.calibration, &plan.error)) return plan;
    cooldown_until.resize(nodes, 0);
    plan.stats.skew_before = plan.calibration.skew;
    plan.stats.skew_after = plan.calibration.skew;
    const bool active = phase == WindowRepairPhase::IDLE || phase == WindowRepairPhase::ACTIVE ||
        (phase == WindowRepairPhase::CONVERGED && plan.calibration.skew > config.skew_trigger);
    if (active) {
        plan.planned = true;
        std::vector<uint32_t> local = membership;
        std::vector<uint8_t> moved_now(nodes, 0);
        plan.stats.refine_budget_bytes = uint64_t(config.refine_budget * double(total_bytes));
        plan.stats.balance_budget_bytes = uint64_t(config.balance_budget * double(total_bytes));
        const auto refine = window_repair_refine(config, owners, owner_hist, pred, price, row_bytes,
            plan.stats.refine_budget_bytes, window, local, plan.W, cooldown_until, moved_now, plan.actions,
            allowed_targets);
        plan.stats.refine_candidates = refine.candidates; plan.stats.refine_moved = refine.moved;
        plan.stats.refine_bytes = refine.bytes; plan.stats.rejected_balance = refine.rejected_balance;
        const auto balance = window_repair_balance(config, owners, owner_hist, pred, price, row_bytes,
            plan.stats.balance_budget_bytes, window, local, plan.W, cooldown_until, moved_now, plan.actions,
            allowed_targets);
        plan.stats.balance_triggered = balance.triggered; plan.stats.balance_moved = balance.moved;
        plan.stats.balance_bytes = balance.bytes;
        plan.stats.skew_after = window_repair_skew(plan.W);
    }
    plan.pred = std::move(pred);
    plan.cooldown_until = std::move(cooldown_until);
    plan.valid = true;
    return plan;
}

WindowRepairVerdict window_repair_verdict(const WindowRepairConfig& config, uint32_t owners,
                                          const std::vector<double>& baseline_cost_us,
                                          const std::vector<double>& measured_cost_us,
                                          double baseline_hop, double measured_hop,
                                          double recall, double reference) {
    WindowRepairVerdict verdict;
    if (!window_owners_valid(owners) || baseline_cost_us.size() != owners ||
        measured_cost_us.size() != owners) return verdict;
    double base_peak = 0.0, base_total = 0.0, peak = 0.0, total = 0.0;
    for (size_t r = 0; r < owners; ++r) {
        const double b = baseline_cost_us[r], m = measured_cost_us[r];
        if (!std::isfinite(b) || b < 0.0 || !std::isfinite(m) || m < 0.0) return verdict;
        base_peak = std::max(base_peak, b); base_total += b;
        peak = std::max(peak, m); total += m;
    }
    if (!(base_peak > 0.0) || !(base_total > 0.0) || !(baseline_hop > 0.0) || !std::isfinite(baseline_hop) ||
        !std::isfinite(measured_hop) || measured_hop < 0.0) return verdict;
    verdict.peak_ratio = peak / base_peak;
    verdict.total_ratio = total / base_total;
    verdict.hop_ratio = measured_hop / baseline_hop;
    // The recall clause exists only with a configured guard; it then fails closed
    // when either window has no scored recall.
    const bool recall_ok = !config.recall_guard ||
        (std::isfinite(recall) && std::isfinite(reference) && recall >= reference - config.recall_guard_pp);
    verdict.accepted = recall_ok &&
        verdict.hop_ratio <= 1.0 + config.hop_slack &&
        verdict.peak_ratio <= 1.0 + config.slack &&
        verdict.total_ratio <= 1.0 + config.slack;
    return verdict;
}

WindowRepairTransition window_repair_account_window(const WindowRepairConfig& config,
                                                    WindowRepairState& state,
                                                    bool issued, double skew) {
    WindowRepairTransition transition;
    ++state.windows;
    state.skew = skew;
    if (state.phase == WindowRepairPhase::ABORTED) return transition;
    if (state.phase == WindowRepairPhase::IDLE) { state.phase = WindowRepairPhase::ACTIVE; state.quiet_windows = 0; }
    if (state.phase == WindowRepairPhase::CONVERGED) {
        if (!(skew > config.skew_trigger)) return transition;
        state.phase = WindowRepairPhase::ACTIVE; state.quiet_windows = 0; transition.reactivated = true;
    }
    if (!issued && skew <= config.skew_trigger) ++state.quiet_windows; else state.quiet_windows = 0;
    if (config.converge_windows && state.quiet_windows >= config.converge_windows) {
        state.phase = WindowRepairPhase::CONVERGED; transition.converged = true;
    }
    return transition;
}

bool window_repair_guard_tripped(const WindowRepairConfig& config, WindowRepairState& state,
                                 double verified_peak_us) {
    if (!std::isfinite(verified_peak_us)) return false;
    state.verified_peaks.push_back(verified_peak_us);
    const size_t keep = std::max<size_t>(1, config.converge_windows);
    while (state.verified_peaks.size() > keep) state.verified_peaks.erase(state.verified_peaks.begin());
    if (!config.converge_windows || state.verified_peaks.size() < config.converge_windows ||
        !std::isfinite(state.first_peak_us) || !(state.first_peak_us > 0.0)) return false;
    const double bound = (1.0 + config.guard) * state.first_peak_us;
    for (const double peak : state.verified_peaks) if (!(peak > bound)) return false;
    return true;
}

uint64_t window_repair_total_cap(const WindowRepairConfig& config, uint64_t total_bytes) {
    // Gross (forward + inverse) bytes. The parser keeps cum <= 1, so this is at most twice the
    // table; the /2 inverse reservation at admission always leaves cum forward bytes.
    return uint64_t(2.0 * config.cumulative_cap * double(total_bytes));
}
uint64_t window_repair_move_rate_cap(const WindowRepairConfig& config, uint64_t total_bytes) {
    return uint64_t(2.0 * double(kHistoryRateWindows) * (config.refine_budget + config.balance_budget) *
                    double(total_bytes));
}
std::string window_repair_identity_tag(const WindowRepairConfig& config) {
    std::ostringstream out;
    out << std::setprecision(17) << ":repair:local:refine=" << config.refine_budget
        << ":balance=" << config.balance_budget << ":cum=" << config.cumulative_cap << ":recall_guard=";
    if (config.recall_guard) out << config.recall_guard_pp; else out << "off";
    out << ":weight=" << window_repair_weight_name(config.weight);
    return out.str();
}

bool window_repair_parse_token(const std::string& body, WindowRepairConfig* out, std::string* error) {
    auto fail = [&](const std::string& why) { if (error) *error = why; return false; };
    if (!out) return fail("repair token output missing");
    WindowRepairConfig config;
    bool algo = false;
    std::vector<std::string> seen;
    size_t start = 0;
    if (body.empty()) return fail("repair token is empty");
    while (start <= body.size()) {
        const size_t comma = body.find(',', start);
        const std::string pair = body.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        start = comma == std::string::npos ? body.size() + 1 : comma + 1;
        const size_t colon = pair.find(':');
        if (pair.empty() || colon == std::string::npos || colon == 0 || colon + 1 >= pair.size())
            return fail("repair token pair must be key:value");
        const std::string key = pair.substr(0, colon), value = pair.substr(colon + 1);
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) return fail("repair token duplicate key " + key);
        seen.push_back(key);
        if (key == "algo") {
            if (value != "local") return fail("repair algo must be local");
            algo = true; continue;
        }
        // The weight is a name, parsed before the numeric values (weight:x never reaches strtod).
        if (key == "weight") {
            if (value == "w") config.weight = WindowRepairWeight::W;
            else if (value == "wnb") config.weight = WindowRepairWeight::WNB;
            else return fail("repair token weight must be w (the default) or wnb");
            continue;
        }
        if (key == "recall_guard" && value == "off") {
            config.recall_guard = false; config.recall_guard_pp = 0.0; continue;
        }
        if (key == "balance_target") {
            // LIGHTEST is the default at every K.
            if (value == "lightest") config.balance_target = WindowRepairBalanceTarget::LIGHTEST;
            else if (value == "underloaded") config.balance_target = WindowRepairBalanceTarget::UNDERLOADED;
            else return fail("repair token balance_target must be lightest (the default) or underloaded");
            continue;
        }
        char* end = nullptr;
        const double parsed = std::strtod(value.c_str(), &end);
        if (end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) return fail("repair token value " + key);
        auto integer = [&](uint32_t& target, double low, double high) {
            if (parsed < low || parsed > high || parsed != std::floor(parsed)) return false;
            target = static_cast<uint32_t>(parsed); return true;
        };
        bool ok = true;
        if (key == "refine") { config.refine_budget = parsed; ok = parsed >= 0.0 && parsed <= 1.0; }
        else if (key == "balance") { config.balance_budget = parsed; ok = parsed >= 0.0 && parsed <= 1.0; }
        else if (key == "tau") { config.tau = parsed; ok = parsed >= 0.0 && parsed <= 1.0; }
        else if (key == "eps") { config.eps = parsed; ok = parsed >= 0.0; }
        else if (key == "skew") { config.skew_trigger = parsed; ok = parsed >= 1.0; }
        else if (key == "cooldown") ok = integer(config.cooldown_windows, 0, 1e6);
        else if (key == "cum") { config.cumulative_cap = parsed; ok = parsed > 0.0 && parsed <= 1.0; }
        else if (key == "slack") { config.slack = parsed; ok = parsed >= 0.0; }
        else if (key == "hop_slack") { config.hop_slack = parsed; ok = parsed >= 0.0; }
        else if (key == "recall_guard") {
            config.recall_guard = true; config.recall_guard_pp = parsed; ok = parsed >= 0.0 && parsed <= 100.0;
        }
        else if (key == "ema") { config.ema = parsed; ok = parsed > 0.0 && parsed <= 1.0; }
        else if (key == "converge") ok = integer(config.converge_windows, 1, 1e6);
        else if (key == "guard") { config.guard = parsed; ok = parsed >= 0.0; }
        else if (key == "verify_invalid_limit") ok = integer(config.verify_invalid_limit, 1, 1e6);
        else return fail("repair token unknown key " + key);
        if (!ok) return fail("repair token domain " + key);
    }
    if (!algo) return fail("repair token requires algo:local");
    config.enabled = true;
    *out = config;
    if (error) error->clear();
    return true;
}

std::string window_repair_config_json(const WindowRepairConfig& c) {
    std::ostringstream out;
    out << ",\"repair_config\":{\"algo\":\"local\",\"refine\":" << number(c.refine_budget)
        << ",\"balance\":" << number(c.balance_budget) << ",\"tau\":" << number(c.tau)
        << ",\"eps\":" << number(c.eps) << ",\"skew\":" << number(c.skew_trigger)
        << ",\"cooldown\":" << c.cooldown_windows << ",\"cum\":" << number(c.cumulative_cap)
        << ",\"slack\":" << number(c.slack) << ",\"hop_slack\":" << number(c.hop_slack)
        << ",\"recall_guard\":" << (c.recall_guard ? number(c.recall_guard_pp) : std::string("\"off\""))
        << ",\"ema\":" << number(c.ema)
        << ",\"converge\":" << c.converge_windows << ",\"guard\":" << number(c.guard)
        << ",\"balance_target\":\"" << window_repair_balance_target_name(c.balance_target) << '"'
        << ",\"verify_invalid_limit\":" << c.verify_invalid_limit
        << ",\"weight\":\"" << window_repair_weight_name(c.weight) << "\"}";
    return out.str();
}

std::string window_repair_state_json(const WindowRepairState& s) {
    std::ostringstream out;
    out << ",\"repair\":{\"phase\":\"" << window_repair_phase_name(s.phase) << "\",\"windows\":" << s.windows
        << ",\"quiet_windows\":" << s.quiet_windows << ",\"bursts\":" << s.bursts << ",\"reverts\":" << s.reverts
        << ",\"cumulative_bytes\":" << s.cumulative_bytes << ",\"skew\":" << number(s.skew)
        << ",\"first_peak_us\":" << number(s.first_peak_us) << ",\"price\":" << vector_json(s.price) << '}';
    return out.str();
}

std::string window_repair_calibration_json(const WindowRepairCalibration& c, const std::vector<double>& price) {
    std::ostringstream out;
    out << "\"queries\":" << c.queries << ",\"active_nodes\":" << c.active_nodes << ",\"W\":" << vector_json(c.W)
        << ",\"skew\":" << number(c.skew) << ",\"measured_share\":" << vector_json(c.measured_share)
        << ",\"price\":" << vector_json(price);
    return out.str();
}

std::string window_repair_stats_json(const WindowRepairBurstStats& s) {
    std::ostringstream out;
    out << ",\"refine_candidates\":" << s.refine_candidates << ",\"refine_moved\":" << s.refine_moved
        << ",\"refine_bytes\":" << s.refine_bytes << ",\"rejected_balance\":" << s.rejected_balance
        << ",\"balance_triggered\":" << (s.balance_triggered ? "true" : "false")
        << ",\"balance_moved\":" << s.balance_moved << ",\"balance_bytes\":" << s.balance_bytes
        << ",\"skew_before\":" << number(s.skew_before) << ",\"skew_after\":" << number(s.skew_after)
        << ",\"refine_budget_bytes\":" << s.refine_budget_bytes << ",\"balance_budget_bytes\":" << s.balance_budget_bytes
        << ",\"balance_target\":\"" << window_repair_balance_target_name(s.balance_target) << '"';
    return out.str();
}
}  // namespace tide
