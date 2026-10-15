// Reference planner (test only), written separately from window_repair_planner.cpp. See test_window_repair_reference.h.
#ifndef REFERENCE_TRACE
#error "define REFERENCE_TRACE before including test_window_repair_reference.inc"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "test_window_repair_reference.h"

#if REFERENCE_TRACE
#define REF_SCALE (trace ? trace->scale : 1.0)
#define REF_NOTE(n, what, a, b, unit) do { if (trace) trace->note((n), (what), (a), (b), (unit)); } while (0)
#define REF_CMP(n, what, a, op, b, unit) ref_compare(trace, (n), (what), (a), (b), (unit), [](double x, double y) { return x op y; })
#define REF_NOTE_RUNNER_UP(n, what, values, chosen, largest) note_runner_up(trace, (n), (what), (values), (chosen), (largest))
#else
#define REF_SCALE 1.0
#define REF_NOTE(n, what, a, b, unit) ((void)0)
#define REF_CMP(n, what, a, op, b, unit) ((a) op (b))
#define REF_NOTE_RUNNER_UP(n, what, values, chosen, largest) ((void)0)
#endif

namespace repair_reference {
namespace {

#if REFERENCE_TRACE
template <class Op>
bool ref_compare(ReferenceTrace* trace, size_t n, const char* what, double a, double b, double unit, Op op) {
    if (trace) trace->note(n, what, a, b, unit);
    return op(a, b);
}
// The chosen first max (or min) against the best of the others.
void note_runner_up(ReferenceTrace* trace, size_t n, const char* what, const std::vector<double>& values,
                    uint32_t chosen, bool largest) {
    if (!trace || values.size() < 2) return;
    double other = largest ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    for (size_t p = 0; p < values.size(); ++p)
        if (p != chosen) other = largest ? std::max(other, values[p]) : std::min(other, values[p]);
    trace->note(n, what, values[chosen], other, trace->scale);
}
#endif

double reference_mean(const std::vector<double>& W) {
    double sum = 0; for (const double w : W) sum += w;
    return sum / double(W.size());
}
double reference_skew(const std::vector<double>& W) {
    const double mean = reference_mean(W);
    return mean > 0.0 ? *std::max_element(W.begin(), W.end()) / mean : 1.0;
}
uint32_t reference_first_max(const std::vector<double>& W) {
    uint32_t at = 0;
    for (uint32_t p = 1; p < W.size(); ++p) if (W[p] > W[at]) at = p;
    return at;
}
// allow[v][p] at entry: hist[v][p] >= tau * deg(v) (deg the integer row sum, > 0),
// the mask bit set, the node neither on cooldown nor moved earlier this window.
std::vector<uint8_t> reference_allow(const WindowRepairConfig& config, uint32_t K, const std::vector<uint32_t>& hist,
                                     const std::vector<uint32_t>& cooldown, const std::vector<uint8_t>& moved,
                                     uint32_t window, const std::vector<uint64_t>* allowed) {
    const size_t n = cooldown.size();
    std::vector<uint8_t> allow(n * K, 0);
    for (size_t v = 0; v < n; ++v) {
        if (cooldown[v] > window || moved[v]) continue;
        uint64_t degree = 0;
        for (uint32_t p = 0; p < K; ++p) degree += hist[v * K + p];
        if (!degree) continue;
        for (uint32_t p = 0; p < K; ++p) {
            const bool masked = allowed && !(((*allowed)[v] >> p) & 1u);
            allow[v * K + p] = !masked && double(hist[v * K + p]) >= config.tau * double(degree);
        }
    }
    return allow;
}
WindowRepairBalanceResult reference_balance_lightest(const WindowRepairConfig& config, uint32_t K,
        const std::vector<uint32_t>& owner_hist, const std::vector<double>& pred,
        const std::vector<double>& price, const std::vector<uint64_t>& row_bytes,
        uint64_t budget_bytes, uint32_t window, std::vector<uint32_t>& membership,
        std::vector<double>& W, std::vector<uint32_t>& cooldown_until,
        std::vector<uint8_t>& moved_now, std::vector<WindowBurstAction>& actions,
        const std::vector<uint64_t>* allowed, ReferenceTrace* trace) {
    (void)trace;
    WindowRepairBalanceResult result;
    result.target = WindowRepairBalanceTarget::LIGHTEST;
    const size_t nodes = membership.size();
    result.skew_before = reference_skew(W);
    if (REF_CMP(actions.size(), "balance trigger (skew)", result.skew_before, <=, config.skew_trigger, 1.0) ||
        budget_bytes == 0) return result;
    result.triggered = true;
    const std::vector<uint8_t> allow = reference_allow(config, K, owner_hist, cooldown_until, moved_now, window, allowed);
    std::vector<uint32_t> order;
    for (size_t v = 0; v < nodes; ++v) if (pred[v] > 0.0 && membership[v] < K) order.push_back(uint32_t(v));
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (pred[a] != pred[b]) return pred[a] > pred[b];
        return a < b;
    });
    std::vector<std::vector<uint32_t>> by_owner(K);
    for (const uint32_t v : order) by_owner[membership[v]].push_back(v);
    std::vector<size_t> ptr(K, 0);
    std::vector<uint8_t> used(nodes, 0);
    std::vector<double> sums = W;
    uint64_t moved_bytes = 0;
    for (;;) {
        const uint32_t peak = uint32_t(std::max_element(sums.begin(), sums.end()) - sums.begin());
        const uint32_t low = uint32_t(std::min_element(sums.begin(), sums.end()) - sums.begin());
        REF_NOTE_RUNNER_UP(actions.size(), "lightest: heaviest owner", sums, peak, true);
        REF_NOTE_RUNNER_UP(actions.size(), "lightest: lightest owner", sums, low, false);
        const double gap = sums[peak] - sums[low];
        const auto& cand = by_owner[peak];
        auto weight = [&](uint32_t v) { return price[peak] * pred[v]; };
        while (ptr[peak] < cand.size() &&
               (used[cand[ptr[peak]]] ||
                REF_CMP(actions.size(), "lightest: weight >= gap", weight(cand[ptr[peak]]), >=, gap, REF_SCALE)))
            ++ptr[peak];
        if (ptr[peak] >= cand.size() || REF_CMP(actions.size(), "lightest: gap <= 0", gap, <=, 0.0, REF_SCALE)) break;
        size_t pick = cand.size(), first_improving = cand.size();
        for (size_t index = ptr[peak]; index < cand.size(); ++index) {
            const uint32_t node = cand[index];
            if (used[node] || !allow[node * K + low]) continue;
            if (first_improving == cand.size()) first_improving = index;
            if (REF_CMP(actions.size(), "lightest: weight <= gap / 2", weight(node), <=, 0.5 * gap, REF_SCALE)) {
                pick = index; break;
            }
        }
        if (pick == cand.size()) pick = first_improving;
        if (pick == cand.size()) break;
        const uint32_t v = cand[pick];
        used[v] = 1;
        if (pick == ptr[peak]) ++ptr[peak];
        sums[peak] -= weight(v);
        sums[low] += price[low] * pred[v];
        membership[v] = low;
        cooldown_until[v] = window + config.cooldown_windows;
        moved_now[v] = 1;
        moved_bytes += row_bytes[v];
        ++result.moved;
        actions.push_back({v, peak, low, row_bytes[v]});
        if (moved_bytes >= budget_bytes) break;
    }
    W = sums;
    result.bytes = moved_bytes;
    return result;
}
// Rule 3b from its definition: every step scans every node for the first
// candidate of the heaviest owner, in (pred desc, id asc) order, that has a
// destination, and picks that destination by (hist desc, load asc, index asc).
WindowRepairBalanceResult reference_balance_underloaded(const WindowRepairConfig& config, uint32_t K,
        const std::vector<uint32_t>& hist, const std::vector<double>& pred,
        const std::vector<double>& price, const std::vector<uint64_t>& row_bytes,
        uint64_t budget_bytes, uint32_t window, std::vector<uint32_t>& membership,
        std::vector<double>& W, std::vector<uint32_t>& cooldown_until,
        std::vector<uint8_t>& moved_now, std::vector<WindowBurstAction>& actions,
        const std::vector<uint64_t>* allowed, ReferenceTrace* trace) {
    (void)trace;
    WindowRepairBalanceResult result;
    result.target = WindowRepairBalanceTarget::UNDERLOADED;
    const size_t n = membership.size();
    result.skew_before = reference_skew(W);
    if (REF_CMP(actions.size(), "balance trigger (skew)", result.skew_before, <=, config.skew_trigger, 1.0) ||
        budget_bytes == 0) return result;
    result.triggered = true;
    const std::vector<uint8_t> allow = reference_allow(config, K, hist, cooldown_until, moved_now, window, allowed);
    const std::vector<uint32_t> entry_owner = membership;
    std::vector<uint8_t> used(n, 0);
    uint64_t moved_bytes = 0;
    for (;;) {
        const uint32_t peak = reference_first_max(W);
        REF_NOTE_RUNNER_UP(actions.size(), "underloaded: heaviest owner", W, peak, true);
        const double mean = reference_mean(W);
        if (!REF_CMP(actions.size(), "underloaded: heaviest > mean", W[peak], >, mean, REF_SCALE)) break;
        const double limit = (1.0 + config.eps) * mean;
        auto destination = [&](uint32_t v) {
            uint32_t best = K;
            for (uint32_t q = 0; q < K; ++q) {
                if (q == peak || !allow[size_t(v) * K + q] ||
                    !REF_CMP(actions.size(), "underloaded: load < mean", W[q], <, mean, REF_SCALE)) continue;
                const double after = W[q] + price[q] * pred[v];
                if (REF_CMP(actions.size(), "underloaded: load after > limit", after, >, limit, REF_SCALE) ||
                    !REF_CMP(actions.size(), "underloaded: load after < heaviest", after, <, W[peak], REF_SCALE))
                    continue;
                const uint32_t h = hist[size_t(v) * K + q];
                if (best == K) { best = q; continue; }
                const uint32_t hb = hist[size_t(v) * K + best];
                if (h > hb ||
                    (h == hb && REF_CMP(actions.size(), "underloaded: lighter destination", W[q], <, W[best], REF_SCALE)))
                    best = q;
            }
            return best;
        };
        uint32_t chosen = UINT32_MAX, target = K;
        for (uint32_t v = 0; v < n; ++v) {
            if (entry_owner[v] != peak || used[v] || !(pred[v] > 0.0)) continue;
            if (chosen != UINT32_MAX && (pred[v] < pred[chosen] || (pred[v] == pred[chosen] && v > chosen))) continue;
            const uint32_t q = destination(v);
            if (q == K) continue;
            chosen = v; target = q;
        }
        if (chosen == UINT32_MAX) break;
        used[chosen] = 1;
        W[peak] -= price[peak] * pred[chosen];
        W[target] += price[target] * pred[chosen];
        membership[chosen] = target;
        cooldown_until[chosen] = window + config.cooldown_windows;
        moved_now[chosen] = 1;
        moved_bytes += row_bytes[chosen];
        ++result.moved;
        actions.push_back({chosen, peak, target, row_bytes[chosen]});
        if (moved_bytes >= budget_bytes) break;
    }
    result.bytes = moved_bytes;
    return result;
}
ReferencePlan reference_plan(const WindowRepairConfig& config, uint32_t K, uint32_t window, uint64_t queries,
        uint64_t total_bytes, const std::vector<uint32_t>& membership, const std::vector<uint64_t>& row_bytes,
        const std::vector<uint64_t>& work, const std::vector<uint64_t>& nb, const std::vector<uint32_t>& hist,
        const std::vector<double>& price, const std::vector<double>& previous, std::vector<uint32_t> cooldown,
        const std::vector<uint64_t>* allowed, ReferenceTrace* trace) {
    ReferencePlan out;
    const size_t n = membership.size();
    // Calibration: pred from this window's per-query work (EMA after the first),
    // W[p] = price[p] x the sum of pred over p's nodes in ascending id order.
    // Weight W: the window's own work per query (nb is empty). WNB adds the in-neighbors' work.
    // EMA over the ids `previous` covers. Later ids (inserted since) start fresh.
    const size_t known = previous.size() <= n ? previous.size() : 0;
    const bool wnb = config.weight == tide::WindowRepairWeight::WNB;
    out.pred.resize(n);
    for (size_t v = 0; v < n; ++v) {
        const double q = wnb ? (double(work[v]) + double(nb[v])) / double(queries) : double(work[v]) / double(queries);
        out.pred[v] = v >= known ? q : config.ema * q + (1.0 - config.ema) * previous[v];
    }
    out.W.assign(K, 0.0);
    for (uint32_t p = 0; p < K; ++p) {
        double owned = 0.0;
        for (size_t v = 0; v < n; ++v) if (membership[v] == p) owned += out.pred[v];
        out.W[p] = price[p] * owned;
    }
#if REFERENCE_TRACE
    if (trace) {
        double scale = 0.0;
        for (const double w : out.W) scale += std::fabs(w);
        trace->scale = scale > 0.0 ? scale : 1.0;
    }
#else
    (void)trace;
#endif
    out.skew_before = reference_skew(out.W);
    // Refine: a node is a candidate when the best other owner q (most of its
    // neighbors, lowest index on ties) holds more of its neighbors than its own
    // owner, it is off cooldown, and q is allowed. Candidates are taken one at a
    // time, next in (gain desc, pred desc, id asc) order. The walk ends at the
    // first that would exceed the byte budget. A move that would lift W[q] above
    // (1 + eps) x the entry mean is counted as rejected and skipped.
    std::vector<uint32_t> local = membership;
    std::vector<uint8_t> moved(n, 0);
    std::vector<uint32_t> target(n, K);
    std::vector<int64_t> gain(n, 0);
    for (size_t v = 0; v < n; ++v) {
        const uint32_t own = membership[v];
        uint32_t top = 0;
        bool any = false;
        for (uint32_t p = 0; p < K; ++p)
            if (p != own && (!any || hist[v * K + p] > top)) { top = hist[v * K + p]; any = true; }
        uint32_t q = 0;
        while (q == own || hist[v * K + q] != top) ++q;
        const int64_t g = int64_t(top) - int64_t(hist[v * K + own]);
        const bool masked = allowed && !(((*allowed)[v] >> q) & 1u);
        if (g > 0 && !(cooldown[v] > window) && !masked) { target[v] = q; gain[v] = g; ++out.refine_candidates; }
    }
    const uint64_t refine_budget = uint64_t(config.refine_budget * double(total_bytes));
    const double limit = (1.0 + config.eps) * reference_mean(out.W);
    std::vector<uint8_t> visited(n, 0);
    for (;;) {
        size_t next = n;
        for (size_t v = 0; v < n; ++v) {
            if (target[v] == K || visited[v]) continue;
            if (next == n || gain[v] > gain[next] ||
                (gain[v] == gain[next] && out.pred[v] > out.pred[next])) next = v;  // ascending v keeps the lower id
        }
        if (next == n) break;
        visited[next] = 1;
        if (out.refine_bytes + row_bytes[next] > refine_budget) break;
        const uint32_t own = membership[next], q = target[next];
        if (REF_CMP(out.actions.size(), "refine: load after > limit", out.W[q] + price[q] * out.pred[next], >, limit,
                    REF_SCALE)) { ++out.rejected; continue; }
        out.W[own] -= price[own] * out.pred[next];
        out.W[q] += price[q] * out.pred[next];
        local[next] = q; cooldown[next] = window + config.cooldown_windows; moved[next] = 1;
        out.refine_bytes += row_bytes[next]; ++out.refine_moved;
        out.actions.push_back({uint32_t(next), own, q, row_bytes[next]});
    }
    const uint64_t balance_budget = uint64_t(config.balance_budget * double(total_bytes));
    const bool underloaded = config.balance_target == WindowRepairBalanceTarget::UNDERLOADED;
    const auto balance = underloaded
        ? reference_balance_underloaded(config, K, hist, out.pred, price, row_bytes, balance_budget, window, local,
                                        out.W, cooldown, moved, out.actions, allowed, trace)
        : reference_balance_lightest(config, K, hist, out.pred, price, row_bytes, balance_budget, window, local,
                                     out.W, cooldown, moved, out.actions, allowed, trace);
    out.balance_triggered = balance.triggered; out.balance_moved = balance.moved; out.balance_bytes = balance.bytes;
    out.skew_after = reference_skew(out.W);
    out.cooldown = cooldown;
    return out;
}

}  // namespace
}  // namespace repair_reference
