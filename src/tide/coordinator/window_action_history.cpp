#include "window_action_history.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <set>
#include <sstream>

#include "sha256.h"
#include "window_controller_runtime.h"

namespace tide {
namespace {
using Clock = WindowHistoryClock;
Clock::time_point after(Clock::time_point now, double seconds) {
    return now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}
int64_t clock_ns(Clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}
// Forward plus inverse bytes debited within the rolling window of kHistoryRateWindows periods.
uint64_t rolling(const WindowActionHistory& h, Clock::time_point now) {
    uint64_t used = 0;
    for (const auto& debit : h.move_debits)
        if (now < after(debit.first, kHistoryRateWindows * h.period_s)) used += debit.second;
    return used;
}
void prune_debits(WindowActionHistory& h, Clock::time_point now) {
    while (!h.move_debits.empty() &&
           now >= after(h.move_debits.front().first, kHistoryRateWindows * h.period_s))
        h.move_debits.pop_front();
}
// Node cooldowns end by planning windows, never by time.
void prune_cooldowns(WindowActionHistory& h) {
    for (auto i = h.move_cooldowns.begin(); i != h.move_cooldowns.end();)
        if (h.planning_windows >= i->second) i = h.move_cooldowns.erase(i); else ++i;
}
// a rejected or cancelled batch: back off for 2, 4, then 8 planning windows
void backoff(WindowActionHistory& h) {
    for (const auto& a : h.pending) {
        auto& b = h.direction_backoffs[window_history_direction_key(a.node, a.source_owner, a.target_owner)];
        b.failures = std::min<uint32_t>(b.failures + 1, kHistoryBackoffMaxFailures);
        b.until_plan = h.planning_windows + (uint64_t{1} << b.failures);
    }
}
uint64_t active_cooldowns(const WindowActionHistory& h) {
    uint64_t active = 0;
    for (const auto& entry : h.move_cooldowns) active += h.planning_windows < entry.second;
    return active;
}
uint64_t active_backoffs(const WindowActionHistory& h) {
    uint64_t active = 0;
    for (const auto& entry : h.direction_backoffs) active += h.planning_windows < entry.second.until_plan;
    return active;
}
bool same_actions(const std::vector<WindowBurstAction>& a, const std::vector<WindowBurstAction>& b, bool inverse) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].node != b[i].node || a[i].adjacency_bytes != b[i].adjacency_bytes ||
            a[i].source_owner != (inverse ? b[i].target_owner : b[i].source_owner) ||
            a[i].target_owner != (inverse ? b[i].source_owner : b[i].target_owner)) return false;
    return true;
}
void event(WindowControllerRuntime& r, const char* name, Clock::time_point now, const std::string& fields = "") {
    const auto& h = *r.action_history;
    r.journal << "{\"event\":\"" << name << "\",\"window_epoch\":" << r.window_epoch
              << ",\"mono_ns\":" << window_journal_now_ns() << window_history_json(r, now)
              << ",\"history_actions\":" << h.pending.size();
    if (!fields.empty()) r.journal << ',' << fields;
    r.journal << "}\n";
    r.journal.flush();
}
void clear_pending(WindowActionHistory& h) {
    h.pending.clear(); h.pending_bytes = h.reverse_reserved = 0; h.committed = false;
}
// The settings the context hash covers, by the names controller-start and the docs use.
std::vector<std::pair<std::string, std::string>> bound_settings(const WindowControllerConfig& c) {
    auto number = [](double value) {  // the shortest text that reads back as the value
        std::string text;
        for (int digits = 6; digits <= 17; ++digits) {
            std::ostringstream out; out << std::setprecision(digits) << value; text = out.str();
            if (std::strtod(text.c_str(), nullptr) == value) break;
        }
        return text;
    };
    const auto& r = c.repair;
    return {{"the server count", std::to_string(c.ranks)},
            {"window_sec", number(c.window_seconds)},
            {"cooldown_sec", number(c.cooldown_seconds)},
            {"proposal_query_limit", std::to_string(c.proposal_query_limit)},
            {"min_window_queries", std::to_string(c.min_window_queries)},
            {"the journal directory", c.journal_dir},
            {"the vector bytes per row", std::to_string(c.vector_bytes_per_row)},
            {"repair refine", number(r.refine_budget)},
            {"repair balance", number(r.balance_budget)},
            {"repair cum", number(r.cumulative_cap)},
            {"repair recall_guard", r.recall_guard ? number(r.recall_guard_pp) : "off"},
            {"repair weight", window_repair_weight_name(r.weight)}};
}
// "window_sec is 2 now and 1 in the history, ..." for the settings that differ.
std::string changed_settings(const std::vector<std::pair<std::string, std::string>>& bound,
                             const std::vector<std::pair<std::string, std::string>>& now) {
    std::string out;
    for (size_t i = 0; i < now.size(); ++i) {
        const std::string was = i < bound.size() ? bound[i].second : "?";
        if (was == now[i].second) continue;
        out += (out.empty() ? "" : ", ") + now[i].first + " is " + now[i].second + " now and " + was +
               " in the history";
    }
    return out;
}
}  // namespace

bool window_history_bind(WindowControllerRuntime& r, std::shared_ptr<WindowActionHistory>& retained,
                         uint64_t epoch, std::string& error) {
    const auto& c = r.config;
    const double period = c.window_seconds + c.cooldown_seconds;
    if (!std::isfinite(period) || period <= 0 ||
        period > std::chrono::duration<double>(Clock::duration::max()).count() / 16) {
        error = "history period must be finite and positive"; return false;
    }
    // row bytes are bound per phase (window_history_bind_phase_bytes)
    std::ostringstream identity;
    identity << std::setprecision(17) << c.ranks << ':' << c.window_seconds << ':' << c.cooldown_seconds
             << ':' << c.proposal_query_limit << ':' << c.min_window_queries << ':' << c.journal_dir
             << ':' << c.vector_bytes_per_row;
    identity << window_repair_identity_tag(c.repair);
    const auto text = identity.str();
    const auto context = sha256_hex(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    const char* fresh = "run reset to start a new controller history (reset also reloads the load-time membership "
                        "and drops the inserts)";
    if (retained) {
        if (retained->context_sha256 != context) {
            const auto changed = changed_settings(retained->settings, bound_settings(c));
            error = "the settings differ from the controller history of the earlier controller-start (" +
                    (changed.empty() ? std::string("another configuration") : changed) +
                    "). Use the same settings, or " + fresh;
            return false;
        }
        if (retained->expected_epoch != epoch) {
            error = "ownership epoch " + std::to_string(epoch) + " differs from the controller history's epoch " +
                    std::to_string(retained->expected_epoch) + " (a MOVE ran outside the controller). To go on, " +
                    fresh;
            return false;
        }
        if (!retained->pending.empty()) {
            error = std::string("the controller history still has a pending batch. To go on, ") + fresh;
            return false;
        }
    } else {
        if (epoch != 0) { error = "a first history requires ownership epoch 0"; return false; }
        retained = std::make_shared<WindowActionHistory>();
        retained->context_sha256 = context;
        retained->settings = bound_settings(c);
        retained->period_s = period;
    }
    // a new phase binds new caps
    auto& h = *retained;
    ++h.phases;
    h.total_spent = 0; h.total_cap = 0; h.move_rate_cap = 0; h.phase_bytes_bound = false;
    h.move_debits.clear();
    r.action_history = retained;
    r.moved_bytes = 0; r.window_epoch = h.saved_window_epoch; r.cycles = h.saved_cycles;
    r.commits = h.saved_commits; r.actions_committed = h.saved_actions; r.reverts = h.saved_reverts;
    r.cooldown_until = std::max(r.cooldown_until, h.saved_cooldown);
    r.repair_state.pred = h.saved_pred;
    r.repair_state.cooldown_until = h.saved_cooldown_until;
    r.repair_state.windows = h.saved_repair_windows;
    return true;
}

bool window_history_bind_phase_bytes(WindowControllerRuntime& r) {
    if (!r.action_history || r.action_history->phase_bytes_bound) return false;
    auto& h = *r.action_history;
    h.total_cap = window_repair_total_cap(r.config.repair, r.total_bytes);
    h.move_rate_cap = window_repair_move_rate_cap(r.config.repair, r.total_bytes);
    h.phase_bytes_bound = true;
    return true;
}

void window_history_save(WindowControllerRuntime& r) {
    if (!r.action_history) return;
    auto& h = *r.action_history;
    h.saved_window_epoch = r.window_epoch; h.saved_cycles = r.cycles; h.saved_commits = r.commits;
    h.saved_actions = r.actions_committed; h.saved_reverts = r.reverts;
    h.saved_cooldown = r.cooldown_until;
    h.saved_pred = r.repair_state.pred;
    h.saved_cooldown_until = r.repair_state.cooldown_until;
    h.saved_repair_windows = r.repair_state.windows;
}

uint64_t window_history_direction_key(uint32_t node, uint32_t source, uint32_t target) {
    return (uint64_t{node} << 32) | (uint64_t{source & 0xFFFFu} << 16) | (target & 0xFFFFu);
}

WindowHistoryEligibility window_history_eligibility(WindowControllerRuntime& r,
    const std::vector<uint32_t>& membership) {
    WindowHistoryEligibility view;
    if (!r.action_history) return view;
    auto& h = *r.action_history; prune_cooldowns(h);
    view.planning_window = h.planning_windows + 1;
    const uint32_t owners = r.config.ranks;
    const uint64_t all = owners >= 64 ? ~uint64_t{0} : (uint64_t{1} << owners) - 1;
    auto narrow = [&] {
        if (view.move_targets.empty()) {
            view.move_targets.resize(membership.size());
            for (size_t v = 0; v < membership.size(); ++v)
                view.move_targets[v] = membership[v] < owners && membership[v] < 64
                    ? all ^ (uint64_t{1} << membership[v]) : 0;
        }
    };
    for (const auto& [node, until] : h.move_cooldowns) if (node < membership.size() && h.planning_windows < until) {
        narrow(); view.move_targets[node] = 0; ++view.cooldown_nodes;
    }
    for (const auto& [k, b] : h.direction_backoffs) {
        const uint64_t node = k >> 32;
        const uint32_t source = uint32_t(k >> 16) & 0xFFFFu, target = uint32_t(k) & 0xFFFFu;
        if (node >= membership.size() || source != membership[node] || target >= 64 ||
            h.planning_windows >= b.until_plan) continue;
        narrow(); view.move_targets[node] &= ~(uint64_t{1} << target); ++view.backoff_directions;
    }
    return view;
}

void window_history_count_planning_window(WindowControllerRuntime& r) {
    if (!r.action_history) return;
    ++r.action_history->planning_windows;
    prune_cooldowns(*r.action_history);
}

WindowHistoryBudget window_history_budget(const WindowControllerRuntime& r, Clock::time_point now) {
    if (!r.action_history || r.action_history->total_spent > r.action_history->total_cap) return {};
    const auto& h = *r.action_history;
    const uint64_t used = rolling(h, now);
    // half of each remaining cap (the other half is reserved for the inverse MOVE)
    return {(h.total_cap - h.total_spent) / 2, used <= h.move_rate_cap ? (h.move_rate_cap - used) / 2 : 0};
}

WindowHistoryAdmission window_history_admit(WindowControllerRuntime& r, std::vector<WindowBurstAction>& actions,
    uint64_t epoch, Clock::time_point now, std::string& error) {
    if (!r.action_history || r.action_history->expected_epoch != epoch ||
        window_controller_has_unresolved_action(r) || actions.empty()) {
        error = "history missing, epoch mismatch or unresolved admission"; return WindowHistoryAdmission::Invalid;
    }
    auto& h = *r.action_history; prune_debits(h, now);
    if (h.total_spent > h.total_cap) { error = "cumulative accounting exceeds the safety cap"; return WindowHistoryAdmission::Invalid; }
    const auto budget = window_history_budget(r, now);
    const uint64_t limit = std::min(budget.total_left, budget.rate_left);
    uint64_t kept = 0; size_t count = 0; std::set<uint32_t> nodes;
    for (const auto& a : actions) {
        if (a.node >= r.row_bytes.size() || a.source_owner >= r.config.ranks || a.target_owner >= r.config.ranks ||
            a.source_owner == a.target_owner || a.adjacency_bytes != r.row_bytes[a.node] ||
            !nodes.insert(a.node).second) {
            error = "invalid action: node, owner or row size out of range, or node repeated";
            return WindowHistoryAdmission::Invalid;
        }
        if (a.adjacency_bytes > limit - kept) break;
        kept += a.adjacency_bytes; ++count;
    }
    if (!count) return budget.total_left < actions.front().adjacency_bytes ? WindowHistoryAdmission::TotalCap
                                                                            : WindowHistoryAdmission::RateWait;
    actions.resize(count); h.pending = actions; h.pending_bytes = kept; h.reverse_reserved = kept;
    event(r, "history_admitted", now); return WindowHistoryAdmission::Admitted;
}

bool window_history_committed(WindowControllerRuntime& r, uint64_t epoch, Clock::time_point now) {
    if (!r.action_history) return false;
    auto& h = *r.action_history;
    if (h.pending.empty() || h.committed || h.expected_epoch == UINT64_MAX || epoch != h.expected_epoch + 1 ||
        h.pending_bytes != r.pending_bytes || !same_actions(h.pending, r.pending_actions, false) ||
        h.pending_bytes > h.total_cap - h.total_spent) return false;
    h.expected_epoch = epoch; h.total_spent += h.pending_bytes; h.lifetime_spent += h.pending_bytes;
    h.committed = true;
    h.move_debits.emplace_back(now, h.pending_bytes);
    event(r, "history_forward_committed", now); return true;
}

bool window_history_limit_actions(WindowControllerRuntime& r, std::vector<WindowBurstAction>& actions,
    uint64_t epoch, Clock::time_point now, const std::string& fields, uint64_t& kept) {
    const auto proposed = actions.size(); std::string error;
    const auto verdict = window_history_admit(r, actions, epoch, now, error);
    const auto admitted = verdict == WindowHistoryAdmission::Admitted ? actions.size() : 0;
    const auto budget_fields = fields + ",\"history_proposed_actions\":" + std::to_string(proposed) +
        ",\"history_kept_actions\":" + std::to_string(admitted) +
        ",\"history_skipped_budget_actions\":" + std::to_string(proposed - admitted);
    if (verdict == WindowHistoryAdmission::Admitted) {
        kept = r.action_history->pending_bytes;
        event(r, "history_budget_selected", now, budget_fields); return true;
    }
    if (verdict == WindowHistoryAdmission::Invalid) {
        r.stopped_on_error = r.proposals_disabled = true; r.state = WindowControllerState::RECOVERY_FAILED;
        r.terminal_reason = "history_admission_failed";
    } else if (verdict == WindowHistoryAdmission::TotalCap) {
        r.proposals_disabled = true; r.terminal_reason = "history_total_cap_exhausted";
    }
    if (r.action_history) event(r, verdict == WindowHistoryAdmission::RateWait ? "history_rate_wait" :
        verdict == WindowHistoryAdmission::TotalCap ? "history_total_cap_exhausted" : "history_admission_failed", now,
        budget_fields + (error.empty() ? "" : ",\"error\":\"" + error + "\""));
    return false;
}

bool window_history_resolved(WindowControllerRuntime& r, bool accepted, uint64_t epoch, Clock::time_point now) {
    if (!r.action_history) return false;
    auto& h = *r.action_history;
    if (h.pending.empty() || !h.committed) return false;
    uint64_t reverse_consumed = 0;
    if (!accepted) {
        if (epoch != h.expected_epoch + 1 || h.reverse_reserved != h.pending_bytes ||
            !same_actions(h.pending, r.pending_actions, true) || h.reverse_reserved > h.total_cap - h.total_spent)
            return false;
        reverse_consumed = h.reverse_reserved;
        h.total_spent += reverse_consumed; h.lifetime_spent += reverse_consumed;
        h.move_debits.emplace_back(now, reverse_consumed); h.expected_epoch = epoch;
    } else if (epoch != h.expected_epoch) return false;
    if (r.moved_bytes != h.total_spent) return false;
    h.reverse_reserved = 0;
    if (accepted) for (const auto& a : h.pending)
        h.direction_backoffs.erase(window_history_direction_key(a.node, a.source_owner, a.target_owner));
    else backoff(h);
    for (const auto& a : h.pending) h.move_cooldowns[a.node] = h.planning_windows + kHistoryNodeCooldownPlans;
    event(r, accepted ? "history_accepted" : "history_rejected_resolved", now,
          "\"history_reverse_consumed_bytes\":" + std::to_string(reverse_consumed));
    clear_pending(h); return true;
}

void window_history_cancel(WindowControllerRuntime& r, Clock::time_point now, bool prepare_failed) {
    if (!r.action_history || r.action_history->pending.empty() || r.action_history->committed) return;
    backoff(*r.action_history);
    event(r, prepare_failed ? "history_prepare_failed_resolved" : "history_forward_cancelled", now);
    clear_pending(*r.action_history);
}

std::string window_history_json(const WindowControllerRuntime& r, Clock::time_point now) {
    if (!r.action_history) return "";
    const auto& h = *r.action_history; std::ostringstream out;
    out << std::setprecision(17)
        << ",\"history_expected_epoch\":" << h.expected_epoch
        << ",\"history_phase\":" << h.phases << ",\"history_phase_bytes_bound\":"
        << (h.phase_bytes_bound ? "true" : "false") << ",\"history_lifetime_bytes\":" << h.lifetime_spent
        << ",\"history_total_bytes\":" << h.total_spent << ",\"history_total_cap_bytes\":" << h.total_cap
        << ",\"history_pending_forward_bytes\":" << h.pending_bytes
        << ",\"history_reverse_reserved_bytes\":" << h.reverse_reserved
        << ",\"history_forward_committed\":" << (h.committed ? "true" : "false")
        << ",\"history_now_ns\":" << clock_ns(now)
        << ",\"history_move_rate_used_bytes\":" << rolling(h, now) << ",\"history_move_rate_cap_bytes\":" << h.move_rate_cap
        << ",\"history_rate_window_s\":" << kHistoryRateWindows * h.period_s
        << ",\"history_planning_windows\":" << h.planning_windows
        << ",\"history_node_cooldown_plans\":" << kHistoryNodeCooldownPlans
        << ",\"history_backoff_base_plans\":" << (uint64_t{1} << 1)
        << ",\"history_backoff_max_plans\":" << (uint64_t{1} << kHistoryBackoffMaxFailures)
        // entries still active at the next planning window
        << ",\"history_cooldown_entries\":" << active_cooldowns(h)
        << ",\"history_backoff_entries\":" << active_backoffs(h);
    return out.str();
}
}  // namespace tide
