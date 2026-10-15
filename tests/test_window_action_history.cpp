// Byte caps and the inverse-MOVE reservation of the action history; invoked by
// the window_controller test.
#include "window_controller_runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <sys/wait.h>

namespace {
using namespace tide;
using Clock = WindowHistoryClock;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", message); std::abort(); }
}
WindowRepairConfig repair() {
    WindowRepairConfig c;
    c.enabled = true; c.refine_budget = 0.01; c.balance_budget = 0.004; c.cumulative_cap = 0.45;
    return c;
}
// 1,000 rows of 100 bytes: total 100,000; total cap 2 x 0.45 x total = 90,000 gross bytes;
// rate cap 8 x (0.01 + 0.004) x total = 11,200 bytes per four 4-second periods.
struct Fixture {
    WindowControllerRuntime r;
    std::shared_ptr<WindowActionHistory> retained;
    Clock::time_point now = Clock::now();
    std::string error;
    explicit Fixture(uint32_t ranks = 4) {
        r.config.ranks = ranks;
        r.config.window_seconds = 3; r.config.cooldown_seconds = 1; r.config.journal_dir = "unit-history";
        r.config.repair = repair();
        r.row_bytes.assign(1000, 100); r.total_bytes = 100000;
        check(window_history_bind(r, retained, 0, error), "fresh epoch-0 history");
        check(retained->total_cap == 0 && retained->move_rate_cap == 0 && !retained->phase_bytes_bound,
              "a phase's caps wait for its row bytes");
        check(window_history_bind_phase_bytes(r) && !window_history_bind_phase_bytes(r) &&
              retained->total_cap == 90000 && retained->move_rate_cap == 11200, "the phase's caps bind once");
    }
    WindowHistoryAdmission admit(std::vector<WindowBurstAction> a) {
        const auto status = window_history_admit(r, a, retained->expected_epoch, now, error);
        if (status == WindowHistoryAdmission::Admitted) { r.pending_actions = a; r.pending_bytes = retained->pending_bytes; }
        return status;
    }
    void commit() {
        check(window_history_committed(r, retained->expected_epoch + 1, now), "exact forward commit");
        r.moved_bytes += r.pending_bytes;
        r.last_committed_actions = r.pending_actions; r.pending_actions.clear(); r.pending_bytes = 0;
        r.verify_pending = true;
    }
    void accept() {
        r.verify_pending = false; r.last_committed_actions.clear();
        check(window_history_resolved(r, true, retained->expected_epoch, now), "resolved accept");
    }
    void reject() {
        r.pending_actions = window_controller_inverse_actions(r.last_committed_actions);
        uint64_t bytes = 0;
        for (const auto& a : r.pending_actions) bytes += a.adjacency_bytes;
        r.moved_bytes += bytes; r.verify_pending = false; r.last_committed_actions.clear();
        check(window_history_resolved(r, false, retained->expected_epoch + 1, now), "resolved inverse MOVE");
        r.pending_actions.clear();
    }
};
std::vector<WindowBurstAction> rows(uint32_t first, uint32_t count) {
    std::vector<WindowBurstAction> out;
    for (uint32_t node = first; node < first + count; ++node) out.push_back({node, 0, 1, 100});
    return out;
}

void repair_caps() {
    Fixture f;
    check(f.retained->total_cap == 90000 && f.retained->move_rate_cap == 11200 && f.retained->period_s == 4,
          "total cap 2 x cum x total; rate cap 8 x (refine + balance) x total");
    const auto budget = window_history_budget(f.r, f.now);
    check(budget.total_left == 45000 && budget.rate_left == 5600, "half of each cap stays reserved for the inverse");
    std::printf("repair_caps OK\n");
}

void revert_reservation() {
    Fixture f;
    check(f.admit(rows(0, 5)) == WindowHistoryAdmission::Admitted && f.retained->pending_bytes == 500 &&
          f.retained->reverse_reserved == 500, "an admitted burst reserves its own bytes for the inverse");
    f.commit();
    check(f.retained->total_spent == 500 && f.retained->committed && f.retained->reverse_reserved == 500,
          "commit charges the forward bytes, keeps the reservation");
    window_history_cancel(f.r, f.now + std::chrono::seconds(100));
    check(f.retained->pending.size() == 5 && f.retained->reverse_reserved == 500,
          "a committed burst cannot be cancelled: its inverse remains owed");
    f.reject();
    check(f.retained->total_spent == 1000 && f.r.moved_bytes == 1000 && f.retained->pending.empty() &&
          !f.retained->reverse_reserved && f.retained->expected_epoch == 2,
          "forward and inverse both charged exactly once");
    check(f.admit(rows(10, 1)) == WindowHistoryAdmission::Admitted, "next burst admitted");
    f.commit();
    f.accept();
    check(f.retained->total_spent == 1100 && !f.retained->reverse_reserved && f.retained->expected_epoch == 3,
          "an accepted burst releases its reservation unspent");
    std::printf("revert_reservation OK\n");
}

void rate_and_total_caps() {
    Fixture f;
    // rate_left 5,600: 56 rows fit, the 57th waits.
    auto burst = rows(0, 60);
    check(f.admit(burst) == WindowHistoryAdmission::Admitted && f.r.pending_actions.size() == 56 &&
          f.retained->pending_bytes == 5600, "the rate budget truncates the burst to a prefix");
    f.commit(); f.reject();  // 11,200 debited in the rolling window
    check(f.admit(rows(100, 1)) == WindowHistoryAdmission::RateWait && f.retained->pending.empty(),
          "a new MOVE waits for the rolling refill");
    f.now += std::chrono::seconds(16);
    check(f.admit(rows(100, 1)) == WindowHistoryAdmission::Admitted, "the rolling window refills after 4 periods");
    window_history_cancel(f.r, f.now); f.r.pending_actions.clear(); f.r.pending_bytes = 0;
    check(f.retained->pending.empty() && f.retained->total_spent == 11200, "cancel releases an uncommitted burst");
    f.retained->total_cap = f.retained->total_spent + 100;
    check(f.admit(rows(200, 1)) == WindowHistoryAdmission::TotalCap,
          "a forward MOVE cannot spend the last bytes its inverse would need");
    std::string error;
    auto wrong = rows(300, 1); wrong[0].adjacency_bytes = 99;
    f.retained->total_cap = 90000;
    check(window_history_admit(f.r, wrong, f.retained->expected_epoch, f.now, error) == WindowHistoryAdmission::Invalid,
          "an action whose bytes differ from its row is invalid");
    auto duplicate = rows(300, 1); duplicate.push_back(duplicate[0]);
    check(window_history_admit(f.r, duplicate, f.retained->expected_epoch, f.now, error) == WindowHistoryAdmission::Invalid,
          "a node moves once per burst");
    check(window_history_admit(f.r, duplicate, f.retained->expected_epoch + 1, f.now, error) == WindowHistoryAdmission::Invalid,
          "admission at another ownership epoch is invalid");
    std::printf("rate_and_total_caps OK\n");
}

// The rejection memory, counted in planning windows (never seconds): a
// resolved burst cools its nodes down for the next two planning windows; a
// rejection (or a cancel before COMMIT) backs off each direction for the next
// 2, 4, then 8 planning windows; an acceptance clears the backoff. Wall-clock
// time alone never ends any of it.
void rejection_memory() {
    Fixture f;
    const std::vector<uint32_t> owners(1000, 0);
    auto planned = [&](unsigned windows) {
        for (unsigned i = 0; i < windows; ++i) window_history_count_planning_window(f.r);
    };
    check(window_history_eligibility(f.r, owners).move_targets.empty(), "no memory: unrestricted");
    planned(1);  // the planning window that proposed the burst
    check(f.admit(rows(0, 2)) == WindowHistoryAdmission::Admitted, "burst of nodes 0 and 1, rank 0 -> 1");
    f.commit(); f.reject();
    f.now += std::chrono::hours(24);  // time passes: the memory does not age by it
    auto view = window_history_eligibility(f.r, owners);
    check(view.planning_window == 2 && view.move_targets.size() == 1000 && view.move_targets[0] == 0 &&
          view.move_targets[1] == 0 && view.move_targets[2] == 14 && view.cooldown_nodes == 2 &&
          view.backoff_directions == 2,
          "at the next planning window, however late, both nodes are frozen; other nodes keep every other target");
    check(f.retained->direction_backoffs.at(window_history_direction_key(0, 0, 1)).failures == 1, "first failure");
    planned(1);
    view = window_history_eligibility(f.r, owners);
    check(view.planning_window == 3 && view.cooldown_nodes == 2 && view.backoff_directions == 2,
          "the second planning window after the rejection is still covered");
    planned(1);
    view = window_history_eligibility(f.r, owners);
    check(view.move_targets.empty() && f.retained->move_cooldowns.empty(),
          "after two planning windows the node cooldown is pruned and the first backoff has ended");
    // The same burst rejected again: the backoff doubles to four planning windows.
    check(f.admit(rows(0, 2)) == WindowHistoryAdmission::Admitted, "the same burst again");
    f.commit(); f.reject();
    planned(2);
    view = window_history_eligibility(f.r, owners);
    check(view.cooldown_nodes == 0 && view.backoff_directions == 2 && view.move_targets[0] == 12 &&
          view.move_targets[1] == 12, "after the cooldown only the rejected direction 0 -> 1 stays closed");
    std::vector<uint32_t> moved = owners; moved[0] = 2;
    check(window_history_eligibility(f.r, moved).move_targets[0] == 11,
          "a backoff applies only while the node is still at the rejected source");
    planned(1);
    check(window_history_eligibility(f.r, owners).backoff_directions == 2, "three planning windows: still closed");
    planned(1);
    check(window_history_eligibility(f.r, owners).move_targets.empty(),
          "the second backoff ends after four planning windows");
    // An accepted burst clears its directions' backoff but still cools its nodes down.
    f.now += std::chrono::seconds(100);  // clear of the rolling rate window
    check(f.admit(rows(0, 1)) == WindowHistoryAdmission::Admitted, "node 0 again");
    f.commit(); f.accept();
    check(!f.retained->direction_backoffs.count(window_history_direction_key(0, 0, 1)) &&
          f.retained->direction_backoffs.count(window_history_direction_key(1, 0, 1)) == 1,
          "acceptance clears only the accepted directions");
    check(window_history_eligibility(f.r, owners).move_targets[0] == 0, "an accepted node cools down too");
    planned(2);
    // A burst cancelled before COMMIT backs off without a node cooldown.
    check(f.admit(rows(5, 1)) == WindowHistoryAdmission::Admitted, "node 5");
    window_history_cancel(f.r, f.now, true); f.r.pending_actions.clear(); f.r.pending_bytes = 0;
    view = window_history_eligibility(f.r, owners);
    check(view.cooldown_nodes == 0 && view.move_targets[5] == 12, "a cancelled burst backs off its direction");
    const auto json = window_history_json(f.r, f.now);
    check(json.find("\"history_planning_windows\":9,\"history_node_cooldown_plans\":2,"
                    "\"history_backoff_base_plans\":2,\"history_backoff_max_plans\":8,"
                    "\"history_cooldown_entries\":0,\"history_backoff_entries\":1") != std::string::npos,
          "the journal reports the memory in planning windows and counts only entries in force");
    std::printf("rejection_memory OK\n");
}

// K servers: a direction key keeps 16 bits per owner, so moves such as 9 -> 13
// and 1 -> 5 (equal in the low two bits) never share a backoff, and the
// eligibility mask has one bit per owner up to 64.
void k_servers_keys_and_mask() {
    std::vector<uint64_t> keys;
    for (uint32_t node : {0u, 1u, 7u, 70000u})
        for (uint32_t source = 0; source < 16; ++source)
            for (uint32_t target = 0; target < 16; ++target)
                keys.push_back(window_history_direction_key(node, source, target));
    std::sort(keys.begin(), keys.end());
    check(std::adjacent_find(keys.begin(), keys.end()) == keys.end(), "no two (node, source, target) share a key");
    check(window_history_direction_key(3, 9, 13) != window_history_direction_key(3, 1, 5) &&
          window_history_direction_key(3, 9, 13) != window_history_direction_key(3, 13, 9),
          "9 -> 13 differs from 1 -> 5 and from 13 -> 9");
    Fixture f(16);
    std::vector<uint32_t> owners(1000, 9);
    owners[2] = 1;
    std::vector<WindowBurstAction> burst = {{0, 9, 13, 100}, {2, 1, 5, 100}};
    std::vector<WindowBurstAction> bad = {{0, 9, 16, 100}};
    check(f.admit(bad) == WindowHistoryAdmission::Invalid, "a target beyond the 16 owners is refused");
    window_history_count_planning_window(f.r);
    check(f.admit(burst) == WindowHistoryAdmission::Admitted, "a K = 16 burst is admitted");
    f.commit(); f.reject();
    for (int i = 0; i < 3; ++i) window_history_count_planning_window(f.r);
    auto view = window_history_eligibility(f.r, owners);
    check(view.move_targets.empty(), "after three planning windows the first backoff has ended");
    f.now += std::chrono::seconds(100);
    check(f.admit(burst) == WindowHistoryAdmission::Admitted, "the same burst again");
    f.commit(); f.reject();
    for (int i = 0; i < 3; ++i) window_history_count_planning_window(f.r);  // cooldown over, backoff (4) on
    view = window_history_eligibility(f.r, owners);
    const uint64_t all16 = (uint64_t{1} << 16) - 1;
    check(view.backoff_directions == 2 && view.cooldown_nodes == 0 &&
          view.move_targets[0] == (all16 ^ (uint64_t{1} << 9) ^ (uint64_t{1} << 13)) &&
          view.move_targets[2] == (all16 ^ (uint64_t{1} << 1) ^ (uint64_t{1} << 5)) &&
          view.move_targets[1] == (all16 ^ (uint64_t{1} << 9)),
          "each backoff closes exactly its own target bit among 16 owners");
    std::printf("k_servers_keys_and_mask OK\n");
}

// K = 64, the largest owner count: the eligibility mask uses all 64 bits (a node
// on owner 63 keeps bits 0..62); a backoff toward owner 63 closes bit 63 only; a
// backoff entry whose target is 64 or more cannot be a bit and is skipped; and a
// burst target beyond the 64 owners is refused at admission.
void k64_mask() {
    Fixture f(64);
    std::vector<uint32_t> owners(1000, 0);
    owners[3] = 63;
    const uint64_t all = ~uint64_t{0};
    window_history_count_planning_window(f.r);
    check(f.admit({{5, 0, 63, 100}}) == WindowHistoryAdmission::Admitted, "a K = 64 burst to owner 63");
    window_history_cancel(f.r, f.now, true); f.r.pending_actions.clear(); f.r.pending_bytes = 0;
    f.retained->direction_backoffs[window_history_direction_key(6, 0, 64)] = {1, 100};
    const auto view = window_history_eligibility(f.r, owners);
    check(view.backoff_directions == 1 && view.cooldown_nodes == 0 &&
          view.move_targets[5] == (all ^ 1u ^ (uint64_t{1} << 63)) && view.move_targets[2] == (all ^ 1u) &&
          view.move_targets[3] == (all ^ (uint64_t{1} << 63)) && view.move_targets[6] == (all ^ 1u),
          "64 owners use every mask bit; the backoff closes bit 63 only; a target of 64 is skipped");
    check(f.admit({{7, 0, 64, 100}}) == WindowHistoryAdmission::Invalid, "a target beyond the 64 owners is refused");
    std::printf("k64_mask OK\n");
}

void bind_identity() {
    WindowControllerRuntime r;
    r.config.ranks = 4;
    r.config.window_seconds = 3; r.config.cooldown_seconds = 1; r.config.journal_dir = "unit-history";
    r.config.repair = repair(); r.row_bytes.assign(1000, 100); r.total_bytes = 100000;
    std::shared_ptr<WindowActionHistory> history; std::string error;
    check(!window_history_bind(r, history, 3, error), "a first history starts at ownership epoch 0");
    check(window_history_bind(r, history, 0, error), "fresh history");
    const auto context = history->context_sha256;
    check(window_history_bind(r, history, 0, error), "the same context resumes");
    r.config.repair.refine_budget = 0.02;
    check(!window_history_bind(r, history, 0, error), "a changed repair budget is another context");
    r.config.repair.refine_budget = 0.01; r.config.repair.recall_guard = true; r.config.repair.recall_guard_pp = 0.1;
    check(!window_history_bind(r, history, 0, error), "turning the recall guard on is another context");
    r.config.repair.recall_guard = false; r.config.repair.recall_guard_pp = 0.0;
    r.config.repair.weight = r.config.repair.weight == WindowRepairWeight::W ? WindowRepairWeight::WNB
                                                                               : WindowRepairWeight::W;
    check(!window_history_bind(r, history, 0, error), "another weight is another context");
    r.config.repair.weight = repair().weight;
    // The context is the configuration only; each phase binds its own row bytes.
    r.row_bytes[7] = 104;
    check(window_history_bind(r, history, 0, error) && history->context_sha256 == context,
          "other row bytes (a later phase, after inserts) keep the context");
    r.row_bytes[7] = 100;
    r.config.vector_bytes_per_row = 384;
    check(!window_history_bind(r, history, 0, error), "another storage mode (vector bytes per row) is another context");
    r.config.vector_bytes_per_row = 0;
    r.config.ranks = 8;
    check(!window_history_bind(r, history, 0, error), "another server count is another context");
    r.config.ranks = 4;
    check(window_history_bind(r, history, 0, error) && history->context_sha256 == context, "restored context resumes");
    check(!window_history_bind(r, history, 1, error), "another ownership epoch refuses");
    history->pending = rows(0, 1);
    check(!window_history_bind(r, history, 0, error), "an unresolved burst refuses a new runtime");
    std::printf("bind_identity OK\n");
}

// controller-stop / controller-start in one process keep the epoch floor, the
// runtime counters and the planner state, while the caps restart with each
// phase; runs in a child because the history is per process.
void start_stop_preserves_history() {
    char pattern[] = "/tmp/window-history-XXXXXX";
    const char* made = mkdtemp(pattern); check(made, "owned directory");
    const std::filesystem::path dir(made);
    WindowControllerConfig config;
    config.journal_dir = dir.string();
    config.window_seconds = 3; config.cooldown_seconds = 1; config.repair = repair(); config.ranks = 4;
    std::string error;
    auto* r = window_controller_start(config, 1000, &error); check(r, "first start");
    check(r->nodes == 1000 && r->row_bytes.empty() && r->total_bytes == 0, "the row bytes wait for a harvest");
    std::string why;
    check(!window_controller_bind_row_bytes(*r, std::vector<uint64_t>(999, 100), &why) && !why.empty(),
          "row bytes of another node count are refused");
    check(window_controller_bind_row_bytes(*r, std::vector<uint64_t>(1000, 100), &why) &&
          r->total_bytes == 100000 && r->action_history->total_cap == 90000, "the phase binds its row bytes and caps");
    check(window_controller_bind_row_bytes(*r, std::vector<uint64_t>(1000, 100), &why), "the same bytes again");
    auto changed = std::vector<uint64_t>(1000, 100); changed[5] = 104;
    check(!window_controller_bind_row_bytes(*r, changed, &why) && r->row_bytes[5] == 100,
          "row bytes that change within a phase are refused");
    auto history = r->action_history;
    r->window_epoch = 9; r->commits = 2; r->reverts = 1; history->total_spent = 700; history->lifetime_spent = 700;
    r->moved_bytes = 700;
    r->repair_state.pred.assign(1000, 0.5); r->repair_state.cooldown_until.assign(1000, 0);
    r->repair_state.cooldown_until[3] = 7; r->repair_state.windows = 6;
    r->proposals_disabled = true;  // e.g. the phase reached its proposal limit
    check(window_controller_stop(), "stop");
    config.start_ownership_epoch = 5;
    check(window_controller_start(config, 1000, &error) == nullptr && !error.empty(),
          "a restart at another ownership epoch refuses the retained history");
    config.start_ownership_epoch = 0;
    // The second phase: 200 more ids (inserted) and other row bytes.
    r = window_controller_start(config, 1200, &error); check(r, "restart over more ids");
    check(r->action_history == history && r->window_epoch == 9 && r->commits == 2 && r->reverts == 1 &&
          r->nodes == 1200, "the restarted runtime continues the same history and window epochs");
    check(history->phases == 2 && history->total_spent == 0 && history->total_cap == 0 && r->moved_bytes == 0 &&
          history->lifetime_spent == 700 && !r->proposals_disabled, "a new phase: caps and spent bytes reset");
    check(r->repair_state.pred.size() == 1000 && r->repair_state.cooldown_until.size() == 1000 &&
          r->repair_state.cooldown_until[3] == 7 && r->repair_state.windows == 6 && !r->repair_state.price_frozen,
          "the planner's EMA, cooldowns and window counter carry over; prices refreeze");
    check(window_controller_bind_row_bytes(*r, std::vector<uint64_t>(1200, 140), &why) &&
          r->total_bytes == 168000 && history->total_cap == 151200, "the second phase binds its own row bytes");
    check(window_controller_stop(), "stop again");
    auto other = config; other.repair.balance_budget = 0.005;
    check(window_controller_start(other, 1200, &error) == nullptr, "another configuration cannot reuse the history");
    // A coordinator reset returns ownership to epoch 0: the retained history of
    // committed migrations would refuse every later start, so reset drops it.
    r = window_controller_start(config, 1200, &error); check(r, "restart once more");
    r->action_history->expected_epoch = 3;  // three committed migrations
    check(!window_controller_reset_history(), "the history is not dropped under an active controller");
    check(window_controller_stop(), "stop before the reset");
    check(window_controller_start(config, 1000, &error) == nullptr, "after the reset (epoch 0) the old history refuses");
    check(window_controller_reset_history(), "reset drops the retained history");
    r = window_controller_start(other, 1000, &error); check(r, "a fresh history binds at epoch 0, any configuration");
    check(r->action_history != history && r->action_history->expected_epoch == 0 &&
          r->action_history->total_spent == 0 && r->moved_bytes == 0 && r->commits == 0 && r->window_epoch == 9 &&
          r->repair_state.pred.empty() && r->repair_state.windows == 0,
          "fresh caps, counters and planner state; the window-epoch floor is kept");
    check(window_controller_stop(), "final stop");
    std::filesystem::remove_all(dir);
}
}  // namespace

void test_window_action_history_cases() {
    repair_caps();
    revert_reservation();
    rate_and_total_caps();
    rejection_memory();
    k_servers_keys_and_mask();
    k64_mask();
    bind_identity();
    const auto child = fork(); check(child >= 0, "fork");
    if (!child) { start_stop_preserves_history(); _exit(0); }
    int status = 0;
    check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "start/stop preserves the history");
    std::printf("start_stop_preserves_history OK\n");
}
