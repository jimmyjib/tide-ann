// Reference planner (test only), compiled without FP contraction in test_window_repair_reference_exact.cpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "window_repair_planner.h"

namespace repair_reference {

using tide::WindowBurstAction;
using tide::WindowRepairBalanceResult;
using tide::WindowRepairBalanceTarget;
using tide::WindowRepairConfig;
using tide::WindowRepairPlan;

struct ReferencePlan {
    std::vector<double> W, pred;
    std::vector<uint32_t> cooldown;
    std::vector<WindowBurstAction> actions;
    uint64_t refine_candidates = 0, refine_moved = 0, refine_bytes = 0, rejected = 0;
    bool balance_triggered = false;
    uint64_t balance_moved = 0, balance_bytes = 0;
    double skew_before = 0, skew_after = 0;
};

// A decisive floating-point comparison of the exact reference whose two sides lie
// within `tolerance * scale` of each other: rounded another way (a fused
// multiply-add in the planner's build) it may go the other way.
//   tolerance: 1e-12. Every load is a sum of priced weights, and each add or subtract
//   rounds by at most 2^-53 of the load scale, so after the at most ~1,000 updates
//   of one trial the two builds differ by < 1.2e-13 x scale. Distinct exact values
//   of these inputs (integers / q <= 64 times prices 0.75 + 0.05 k) differ by far
//   more than 1e-12 x scale (scale <= ~2e4).
//   scale: the sum of the calibrated loads for load comparisons, and 1 for the skew
//   trigger (a ratio >= 1).
struct ReferenceNote {
    size_t actions = 0;  // actions the reference had emitted when it compared
    const char* what = "";
    double lhs = 0, rhs = 0;
};
struct ReferenceTrace {
    double scale = 1.0;
    double tolerance = 1e-12;
    std::vector<ReferenceNote> near_ties;
    void note(size_t actions, const char* what, double lhs, double rhs, double unit) {
        if (std::fabs(lhs - rhs) <= tolerance * unit && near_ties.size() < 256)
            near_ties.push_back({actions, what, lhs, rhs});
    }
};

// Exported by the exact copy (contraction off). `trace` may be null.
ReferencePlan exact_plan(const WindowRepairConfig& config, uint32_t K, uint32_t window, uint64_t queries,
                         uint64_t total_bytes, const std::vector<uint32_t>& membership,
                         const std::vector<uint64_t>& row_bytes, const std::vector<uint64_t>& work,
                         const std::vector<uint64_t>& nb, const std::vector<uint32_t>& hist,
                         const std::vector<double>& price, const std::vector<double>& previous,
                         std::vector<uint32_t> cooldown, const std::vector<uint64_t>* allowed,
                         ReferenceTrace* trace);
WindowRepairBalanceResult exact_balance(const WindowRepairConfig& config, uint32_t K,
                                        const std::vector<uint32_t>& owner_hist, const std::vector<double>& pred,
                                        const std::vector<double>& price, const std::vector<uint64_t>& row_bytes,
                                        uint64_t budget_bytes, uint32_t window, std::vector<uint32_t>& membership,
                                        std::vector<double>& W, std::vector<uint32_t>& cooldown_until,
                                        std::vector<uint8_t>& moved_now, std::vector<WindowBurstAction>& actions,
                                        const std::vector<uint64_t>* allowed, ReferenceTrace* trace);

// Comparison of the planner with the exact reference.
enum class Agreement { EXACT, ROUNDING_BOUNDARY, MISMATCH };

struct Comparison {
    Agreement agreement = Agreement::MISMATCH;
    bool bit_identical = false;   // every number equal bit for bit
    double max_load_diff = 0.0;   // largest |W_planner - W_reference| / scale (identical decisions)
    std::string difference;       // the first differing decision (first_difference)
    std::string boundary;         // the near tie that explains it (ROUNDING_BOUNDARY)
};

inline bool same_action(const WindowBurstAction& a, const WindowBurstAction& b) {
    return a.node == b.node && a.source_owner == b.source_owner && a.target_owner == b.target_owner &&
           a.adjacency_bytes == b.adjacency_bytes;
}

// The first decision in which two action lists differ: its index (the
// reference's action count when it took that decision) and a description.
inline size_t first_action_difference(const std::vector<WindowBurstAction>& planner,
                                      const std::vector<WindowBurstAction>& reference, std::string* text) {
    char line[256];
    for (size_t i = 0; i < planner.size() && i < reference.size(); ++i) {
        if (same_action(planner[i], reference[i])) continue;
        const auto &a = planner[i], &b = reference[i];
        std::snprintf(line, sizeof line, "action %zu: planner %u %u->%u, reference %u %u->%u", i, a.node,
                      a.source_owner, a.target_owner, b.node, b.source_owner, b.target_owner);
        if (text) *text = line;
        return i;
    }
    if (planner.size() != reference.size()) {
        std::snprintf(line, sizeof line, "action count: planner %zu, reference %zu (lists agree before)",
                      planner.size(), reference.size());
        if (text) *text = line;
        return std::min(planner.size(), reference.size());
    }
    if (text) text->clear();
    return SIZE_MAX;
}

inline std::string format_near_ties(const ReferenceTrace& trace, bool every, size_t at) {
    std::string out;
    char line[256];
    for (const auto& note : trace.near_ties) {
        if (!every && note.actions != at) continue;
        std::snprintf(line, sizeof line, "%s%s at action %zu: %a vs %a (diff %.3g x scale %.6g)",
                      out.empty() ? "" : "; ", note.what, note.actions, note.lhs, note.rhs,
                      std::fabs(note.lhs - note.rhs) / trace.scale, trace.scale);
        out += line;
    }
    return out;
}
// The near ties the exact reference noted at action count `at`: only a tie at
// the same action count as the differing decision explains it.
inline std::string describe_near_ties(const ReferenceTrace& trace, size_t at) {
    return format_near_ties(trace, false, at);
}
// Every near tie of a trial (a report, never an explanation).
inline std::string describe_all_near_ties(const ReferenceTrace& trace) { return format_near_ties(trace, true, 0); }

// The balance step alone (no refine): `actions`, counts, loads and cooldowns.
struct BalanceSide {
    const WindowRepairBalanceResult* result;
    const std::vector<uint32_t>* membership;
    const std::vector<uint32_t>* cooldown;
    const std::vector<uint8_t>* moved;
    const std::vector<double>* W;
    const std::vector<WindowBurstAction>* actions;
};

inline bool loads_close(const std::vector<double>& a, const std::vector<double>& b, double tolerance_abs,
                        double* max_diff, bool* bit_identical) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (a[i] != b[i]) *bit_identical = false;
        if (!(d <= tolerance_abs)) return false;
        if (d > *max_diff) *max_diff = d;
    }
    return true;
}

// Decisions (actions, counts, cooldowns) must agree exactly, and numbers within the
// trace's tolerance. A differing decision passes only as ROUNDING_BOUNDARY: the
// exact reference took that same decision (at the same action count) on a
// comparison whose sides were within the tolerance. When the action lists
// agree but a count or a cooldown differs, the differing decision emitted no
// action on either side (any decision that does changes the lists), so it was
// taken after the last action: only a near tie at the final action count
// (the lists' length) explains it.
inline Comparison compare_plan(const WindowRepairPlan& plan, const ReferencePlan& ref, const ReferenceTrace& trace,
                               WindowRepairBalanceTarget target) {
    Comparison out;
    char line[320];
    if (!plan.valid || !plan.planned || plan.stats.balance_target != target) {
        out.difference = "plan invalid, not planned or with another balance target: " + plan.error;
        return out;
    }
    if (plan.stats.refine_candidates != ref.refine_candidates) {  // integer gains: never a rounding question
        std::snprintf(line, sizeof line, "refine candidates %llu vs %llu",
                      (unsigned long long)plan.stats.refine_candidates, (unsigned long long)ref.refine_candidates);
        out.difference = line;
        return out;
    }
    const size_t at = first_action_difference(plan.actions, ref.actions, &out.difference);
    const bool counts_same = plan.stats.refine_moved == ref.refine_moved && plan.stats.rejected_balance == ref.rejected &&
        plan.stats.refine_bytes == ref.refine_bytes && plan.stats.balance_triggered == ref.balance_triggered &&
        plan.stats.balance_moved == ref.balance_moved && plan.stats.balance_bytes == ref.balance_bytes &&
        plan.cooldown_until == ref.cooldown;
    if (at == SIZE_MAX && counts_same) {
        // Identical decisions: the numbers may differ only by rounding.
        out.bit_identical = plan.pred == ref.pred && plan.W == ref.W && plan.stats.skew_before == ref.skew_before &&
                            plan.stats.skew_after == ref.skew_after;
        bool bits = true;
        double pred_diff = 0.0, load_diff = 0.0;
        double pred_scale = 0.0;
        for (const double p : ref.pred) pred_scale = std::max(pred_scale, std::fabs(p));
        const bool close =
            loads_close(plan.pred, ref.pred, trace.tolerance * std::max(1.0, pred_scale), &pred_diff, &bits) &&
            loads_close(plan.W, ref.W, trace.tolerance * trace.scale, &load_diff, &bits) &&
            std::fabs(plan.stats.skew_before - ref.skew_before) <= trace.tolerance &&
            std::fabs(plan.stats.skew_after - ref.skew_after) <= trace.tolerance;
        out.max_load_diff = load_diff / trace.scale;
        if (close) { out.agreement = Agreement::EXACT; return out; }
        std::snprintf(line, sizeof line, "same decisions but numbers beyond the tolerance (max load diff %.3g x scale)",
                      load_diff / trace.scale);
        out.difference = line;
        return out;
    }
    if (at == SIZE_MAX) {
        std::snprintf(line, sizeof line,
                      "same actions, other counts: refine moved/rejected %llu/%llu vs %llu/%llu, balance triggered "
                      "%d vs %d",
                      (unsigned long long)plan.stats.refine_moved, (unsigned long long)plan.stats.rejected_balance,
                      (unsigned long long)ref.refine_moved, (unsigned long long)ref.rejected,
                      int(plan.stats.balance_triggered), int(ref.balance_triggered));
        out.difference = line;
    }
    out.boundary = describe_near_ties(trace, at == SIZE_MAX ? ref.actions.size() : at);
    if (!out.boundary.empty()) out.agreement = Agreement::ROUNDING_BOUNDARY;
    return out;
}

inline Comparison compare_balance(const BalanceSide& planner, const BalanceSide& ref, const ReferenceTrace& trace) {
    Comparison out;
    char line[320];
    const size_t at = first_action_difference(*planner.actions, *ref.actions, &out.difference);
    const bool counts_same = planner.result->triggered == ref.result->triggered &&
        planner.result->moved == ref.result->moved && planner.result->bytes == ref.result->bytes &&
        planner.result->target == ref.result->target && *planner.membership == *ref.membership &&
        *planner.cooldown == *ref.cooldown && *planner.moved == *ref.moved;
    if (at == SIZE_MAX && counts_same) {
        bool bits = true;
        double load_diff = 0.0;
        out.bit_identical = *planner.W == *ref.W && planner.result->skew_before == ref.result->skew_before;
        const bool close = loads_close(*planner.W, *ref.W, trace.tolerance * trace.scale, &load_diff, &bits) &&
            std::fabs(planner.result->skew_before - ref.result->skew_before) <= trace.tolerance;
        out.max_load_diff = load_diff / trace.scale;
        if (close) { out.agreement = Agreement::EXACT; return out; }
        std::snprintf(line, sizeof line, "same decisions but loads beyond the tolerance (%.3g x scale)",
                      load_diff / trace.scale);
        out.difference = line;
        return out;
    }
    if (at == SIZE_MAX) out.difference = "same actions, other counts or state";
    out.boundary = describe_near_ties(trace, at == SIZE_MAX ? ref.actions->size() : at);
    if (!out.boundary.empty()) out.agreement = Agreement::ROUNDING_BOUNDARY;
    return out;
}

}  // namespace repair_reference
