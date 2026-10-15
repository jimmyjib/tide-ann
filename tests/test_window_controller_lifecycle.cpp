// Lifecycle of the repair-only runtime controller, driven through the production
// tick, snapshot decode/load and planning workers, with the coordinator's
// stats-window and migration exchanges stubbed:
//   MOVE -> accept;  MOVE -> reject -> inverse MOVE;
//   a rejected burst is not re-proposed at the next planning window, however
//   long the windows last (the rejection memory counts planning windows);
//   PREPARE / COMMIT / ACCEPT failure injection (forward and inverse), and a
//   PREPARE failure whose rollback fails; an ownership-ambiguous terminal stalls
//   dispatch (controller_dispatch_stalled) once nothing is in flight; every
//   terminal ends by controller-abort without a restart: at once when
//   ownership is known, after a confirmed manual migration-accept or
//   migration-rollback when it was ambiguous;
//   the optional recall guard (unscored trigger, recall drop, unscored verify);
//   a window whose BEGIN ACKs arrive in another search command is abandoned;
//   a PREPARE-failed burst is proposed again once its backoff has passed;
//   an idle rank (no distances or requests: undefined prices) cannot block a
//   pending verdict: after verify_invalid_limit unmeasurable verify windows the
//   burst is rejected and reverted, and the controller stops; failed BEGINs and
//   both harvest_failed paths (refused exchange, decode-or-load deadline) count
//   the same way; so do abandoned windows (search commands shorter than
//   window_sec, BEGIN ACKs in the next command), a failed dump directory and a
//   command change while the harvest was measured: a pending verdict always
//   resolves and controller-stop is never blocked for good; an invalid planning
//   observation does not count: the burst is judged from the window's
//   measurement first;
//   K = 8: the opted-in underloaded balance target, MOVE -> reject -> inverse
//   MOVE, the rejection memory, then MOVE -> accept, with eight ranks' snapshots;
//   the weight: every scenario runs the default weight W (owner-histogram
//   harvests; the fixture frames' nbwork would be zero, so the plans are W^nb's
//   too), weight:wnb requests and merges the nbwork section, and a W controller
//   refuses a frame that carries it;
//   The row bytes bind from the first merged owner histogram of a phase,
//   and a second phase after inserts binds the same history over more ids with
//   other row bytes, restarted caps, the planner's EMA and cooldowns carried
//   and padded, and moves an inserted id;
//   a dump directory that stays unwritable (every lost window, then the
//   revert's own record) still reverts the burst: the fence PREPAREs from the
//   plan held in memory; a plan record that fails only at its close (full
//   disk) is journaled and removed, for the forward plan and the revert.
// Each scenario runs in a child process: the action history is per process.
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/wait.h>

#include "coordinator_types.h"
#include "coordinator_stats_window.h"
#include "membership_migration_coordinator.h"
#include "membership_migration_plan.h"
#include "npy_writer.h"
#include "window_controller_runtime.h"
#include "window_native_control.h"
#include "test_window_fixture.h"

namespace {
void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", why); std::abort(); }
}

struct Stubs {
    bool prepare_ok = true, commit_ok = true, accept_ok = true, rollback_ok = true;
    // A failed PREPARE that some servers had applied: the plan stays active and
    // the coordinator's rollback must undo it.
    bool prepare_failure_leaves_plan = false;
    unsigned begins = 0, harvests = 0, prepares = 0, commits = 0, accepts = 0, rollbacks = 0;
    std::vector<uint32_t> harvest_sections;                // the sections every snapshot request named
    uint32_t ranks = 4;                                    // the plans' server count
    std::vector<tide::MembershipMigrationPlan> prepared;  // every PREPARE's plan, failed ones included
    std::vector<bool> prepared_from_record;                 // per PREPARE: a plan file records it
    uint64_t epoch = 0;
    bool active = false, committed = false;   // like the coordinator's active plan
    tide::MembershipMigrationPlan plan;
    std::vector<tide::MembershipMigrationPlan> accepted;
} stubs;
}  // namespace

namespace tide {
DaemonState g_daemon;
void coordinator_stats_window_send_begin(uint64_t, uint32_t nodes) {
    ++stubs.begins;
    check(nodes == g_daemon.membership.size(), "BEGIN names the window's node count");
}
void coordinator_stats_window_send_harvest(uint64_t, uint32_t nodes, uint32_t sections) {
    ++stubs.harvests;
    stubs.harvest_sections.push_back(sections);
    check(nodes == g_daemon.membership.size(), "snapshot request names every node");
    const WindowControllerRuntime* active = window_controller_active();
    check(active && sections == window_repair_structure_sections(active->config.repair),
          "the snapshot request names the weight's structure sections");
}
// The controller's fence PREPAREs from the plan it holds in memory; its plan
// file, when written, is a record of the same bytes.
bool coordinator_membership_migration_prepare_controller(const std::vector<char>& bytes) {
    ++stubs.prepares;
    check(!stubs.active, "PREPARE never overlaps an active plan");
    MembershipMigrationPlan plan;
    std::string error;
    check(load_membership_migration_plan_bytes(bytes.data(), bytes.size(),
                                               static_cast<uint32_t>(g_daemon.membership.size()), stubs.ranks, plan,
                                               error), "the controller holds a loadable skeleton plan");
    const WindowControllerRuntime* active = window_controller_active();
    if (active && !active->pending_plan_path.empty()) {
        std::ifstream in(active->pending_plan_path, std::ios::binary);
        const std::vector<char> record((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        check(record == bytes, "the plan file records the bytes the fence prepares");
    }
    stubs.prepared_from_record.push_back(active && !active->pending_plan_path.empty());
    stubs.prepared.push_back(plan);
    check(plan.epoch_from == stubs.epoch && plan.mode == kMigrationPlanModeMove, "plan starts at the current epoch");
    for (const auto& move : plan.moves)
        check(g_daemon.membership[move.global_id] == move.from_partition, "plan source is the current owner");
    if (!stubs.prepare_ok) {
        if (stubs.prepare_failure_leaves_plan) { stubs.plan = plan; stubs.active = true; stubs.committed = false; }
        return false;
    }
    stubs.plan = plan; stubs.active = true; stubs.committed = false;
    return true;
}
bool coordinator_membership_migration_commit() {
    ++stubs.commits;
    check(stubs.active && !stubs.committed, "COMMIT follows PREPARE");
    if (!stubs.commit_ok) return false;
    for (const auto& move : stubs.plan.moves) g_daemon.membership[move.global_id] = move.to_partition;
    stubs.epoch = stubs.plan.epoch_to; stubs.committed = true;
    return true;
}
bool coordinator_membership_migration_accept() {
    ++stubs.accepts;
    check(stubs.active && stubs.committed, "ACCEPT follows COMMIT");
    if (!stubs.accept_ok) return false;
    stubs.accepted.push_back(stubs.plan);
    stubs.active = stubs.committed = false; stubs.plan = {};
    return true;
}
bool coordinator_membership_migration_rollback() {
    ++stubs.rollbacks;
    if (!stubs.active) return true;  // nothing active: the coordinator's no-op
    if (!stubs.rollback_ok) return false;
    if (stubs.committed) {
        for (const auto& move : stubs.plan.moves) g_daemon.membership[move.global_id] = move.from_partition;
        stubs.epoch = stubs.plan.epoch_from;
    }
    stubs.active = stubs.committed = false; stubs.plan = {};
    return true;
}
uint64_t coordinator_membership_ownership_epoch() { return stubs.epoch; }
// The fence payload of the last PREPARE: 100 bytes per move through the
// coordinator, 24 of them vector bytes (own-only storage).
const MigrationPayloadStats& coordinator_membership_migration_payload() {
    static MigrationPayloadStats payload;
    payload = MigrationPayloadStats{};
    const uint64_t moves = stubs.prepared.empty() ? 0 : stubs.prepared.back().moves.size();
    payload.moves = moves;
    payload.refresh_request_bytes = 30 * moves;
    payload.refresh_response_bytes = 30 * moves;
    payload.prepare_request_bytes = 40 * moves;
    payload.vector_bytes = payload.prepare_vector_bytes = 24 * moves;
    payload.exchange_deadline_s = 30.5;
    payload.refresh_deadline_s = 60.5;
    return payload;
}
}  // namespace tide

namespace {
using namespace tide;
using window_test::RankCounters;
using State = WindowControllerState;

// Row bytes come from the harvested owner histogram: every fixture frame
// gives each owned node one neighbour per owner, so a row is 4 + 4 x K bytes
// (20 at K = 4: 160 in all), plus the vector bytes under own-only storage.
// Ranks, start membership and per-node window work of a scenario.
struct Topology {
    uint32_t ranks;
    std::vector<uint32_t> start, work;
};
const Topology kK4{4, {0, 0, 0, 0, 0, 1, 2, 3}, {40, 30, 20, 10, 10, 5, 5, 5}};
const std::vector<uint32_t> kStart = kK4.start;
// refine 0 and every histogram row {1,1,1,1}: no refine gain. balance 0.125 x 160 = 20
// bytes: exactly one MOVE, the heaviest rank-0 node 0 to the lightest rank 1 (pred
// 0.4 <= half the 0.95 gap). Prices: every window has Td = D and Ttot = Td + R.
const std::string kToken = "algo:local,refine:0,balance:0.125,tau:0,eps:0.03,skew:1.02,cooldown:1,cum:1,"
                           "slack:0.10,hop_slack:0.05,ema:0.5,converge:3,guard:10";

// One window: per-rank distance and request deltas, completed queries, live recall.
struct Spec {
    std::vector<uint64_t> D, R;  // one entry per rank
    uint64_t queries = 100;
    bool scored = false;
    double recall = 100.0;
};
// Trigger: costs per query 11 / 1.5 / 1.5 / 1.5 (peak 11, total 15.5), hop 2.5.
const Spec kTrigger{{1000, 100, 100, 100}, {100, 50, 50, 50}};
// Verify, accepted: 9 / 3.6 / 1.5 / 1.5 (peak 0.818, total 1.006), hop 2.6 (1.04).
const Spec kKeep{{800, 300, 100, 100}, {100, 60, 50, 50}};
// Verify, rejected: hop 3.0 (1.2 > 1 + hop_slack).
const Spec kHopUp{{800, 300, 100, 100}, {100, 100, 50, 50}};

Spec scored(Spec spec, double recall) { spec.scored = true; spec.recall = recall; return spec; }

struct TestRig {
    std::filesystem::path dir;
    Topology topo;
    WindowControllerConfig config;
    WindowControllerRuntime* r = nullptr;
    std::vector<RankCounters> counters;
    uint64_t completed = 1000, command = 1, recall_count = 0;
    double recall_sum = 0.0;

    explicit TestRig(const std::string& token, uint64_t proposal_limit = 1150, double window_seconds = 0.001,
                     const Topology& topology = kK4, uint32_t vector_bytes_per_row = 0)
        : topo(topology), counters(topology.ranks) {
        char pattern[] = "/tmp/window-lifecycle-XXXXXX";
        const char* made = mkdtemp(pattern);
        check(made != nullptr, "owned journal directory");
        dir = made;
        g_daemon.membership = topo.start;
        stubs = Stubs{};
        stubs.ranks = topo.ranks;
        config.vector_bytes_per_row = vector_bytes_per_row;
        config.journal_dir = dir.string();
        config.window_seconds = window_seconds;
        config.cooldown_seconds = 0.0;
        config.proposal_query_limit = proposal_limit;
        config.ranks = topo.ranks;
        std::string error;
        check(window_repair_parse_token(token, &config.repair, &error), "repair token parses");
        r = window_controller_start(config, nodes(), &error);
        check(r != nullptr, "controller starts with the production loader");
    }
    size_t nodes() const { return topo.start.size(); }
    // One row's MOVE bytes: 4 + 4 x (the histogram row's sum) + the vector bytes.
    uint64_t row() const {
        uint64_t degree = 0;
        if (hist_row.empty()) degree = topo.ranks; else for (const auto count : hist_row) degree += count;
        return 4 + 4 * degree + config.vector_bytes_per_row;
    }
    std::vector<uint16_t> hist_row;  // every owned node's histogram row (empty: one neighbour per owner)
    ControllerTickInputs inputs() const {
        ControllerTickInputs in;
        in.completed_queries = completed; in.recall_sum = recall_sum; in.recall_count = recall_count;
        in.command_id = command;
        return in;  // nothing in flight: every fence drains at once
    }
    void until(const std::function<bool()>& done, const char* what) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!done()) {
            window_controller_tick(inputs());
            check(std::chrono::steady_clock::now() < deadline, what);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    std::vector<uint32_t> work_on(uint32_t rank, const std::vector<uint32_t>& membership = g_daemon.membership) const {
        std::vector<uint32_t> w(nodes(), 0);
        for (uint32_t v = 0; v < nodes(); ++v) if (membership[v] == rank) w[v] = topo.work[v];
        return w;
    }
    // Opens one window (every rank's BEGIN ACK), serves `spec` and closes it:
    // returns in HARVEST_WAIT before any snapshot frame arrives.
    void serve(const Spec& spec) {
        check(spec.D.size() == topo.ranks && spec.R.size() == topo.ranks, "one window entry per rank");
        until([&] { return r->state == State::BEGIN_WAIT; }, "BEGIN sent");
        for (uint32_t rank = 0; rank < topo.ranks; ++rank) {
            const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, r->window_epoch, counters[rank]);
            check(controller_offer(rank, &a, sizeof(a)), "BEGIN ACK is controller traffic");
        }
        until([&] { return r->state == State::WINDOW_OPEN; }, "window opens on every rank's BEGIN ACK");
        for (uint32_t rank = 0; rank < topo.ranks; ++rank) {
            auto& c = counters[rank];
            c.distances += spec.D[rank]; c.distance_us += spec.D[rank];
            c.requests += spec.R[rank]; c.total_us += spec.D[rank] + spec.R[rank];
        }
        completed += spec.queries;
        if (spec.scored) { recall_count += spec.queries; recall_sum += spec.queries * spec.recall / 100.0; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        until([&] { return r->state == State::HARVEST_WAIT; }, "window closes and requests snapshots");
    }
    // Every rank's snapshot frame, its owned ids taken from `membership`, with the
    // structure sections the controller's weight requests (or `sections` when given).
    void deliver_frames(const std::vector<uint32_t>& membership, uint32_t sections = 0) {
        if (!sections) sections = window_repair_structure_sections(r->config.repair);
        for (uint32_t rank = 0; rank < topo.ranks; ++rank) {
            auto bytes = window_test::frame(rank, r->window_epoch, counters[rank], work_on(rank, membership), membership,
                                            hist_row, sections, topo.ranks);
            check(!bytes.empty() && controller_offer_owned(rank, bytes), "snapshot frame is controller traffic");
        }
    }
    // Opens one window, serves `spec`, and delivers every rank's snapshot; returns
    // once the controller has handled the harvest (planning, verdict or deferral).
    // `frame_membership` (default: the current one) names the ids each frame owns:
    // a stale one makes the merged observation invalid while the ACK counters
    // still measure the window.
    void window(const Spec& spec, const std::vector<uint32_t>* frame_membership = nullptr) {
        serve(spec);
        deliver_frames(frame_membership ? *frame_membership : g_daemon.membership);
        until([&] { return r->state != State::HARVEST_WAIT; }, "harvest decoded, loaded and handled");
    }
    // A window whose BEGIN rank 0 refuses (ACK status 1): window_begin_failed.
    void begin_refused() {
        until([&] { return r->state == State::BEGIN_WAIT; }, "BEGIN sent");
        auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, 0, r->window_epoch, counters[0]);
        a.status = 1;
        check(controller_offer(0, &a, sizeof(a)), "a refused BEGIN ACK is controller traffic");
        until([&] { return r->state != State::BEGIN_WAIT; }, "the refused BEGIN is handled");
    }
    // A HARVEST refused: rank 0 answers with a bare HARVEST ACK (no validated
    // snapshot), so the exchange is rejected: harvest_failed with its ACK count.
    void harvest_refused(const Spec& spec) {
        serve(spec);
        const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_HARVEST_ACK, 0, r->window_epoch, counters[0]);
        check(controller_offer(0, &a, sizeof(a)), "a bare HARVEST ACK is controller traffic");
        until([&] { return r->state != State::HARVEST_WAIT; }, "the refused HARVEST is handled");
    }
    // BEGIN's ACKs from every rank: the window opens (in the current command).
    void open_window() {
        until([&] { return r->state == State::BEGIN_WAIT; }, "BEGIN sent");
        for (uint32_t rank = 0; rank < topo.ranks; ++rank) {
            const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, r->window_epoch, counters[rank]);
            check(controller_offer(rank, &a, sizeof(a)), "BEGIN ACK is controller traffic");
        }
        until([&] { return r->state == State::WINDOW_OPEN; }, "window opens on every rank's BEGIN ACK");
    }
    // Every frame arrives and validates, then the exchange deadline passes before
    // the merged structure is loaded: harvest_failed "decode-or-load-deadline".
    void harvest_load_deadline(const Spec& spec) {
        serve(spec);
        deliver_frames(g_daemon.membership);
        until([&] { return window_controller_exchange_complete(*r) || r->state != State::HARVEST_WAIT; },
              "every rank's snapshot validated");
        check(r->state == State::HARVEST_WAIT, "validated, not yet loaded");
        r->exchange_deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
        window_controller_tick(inputs());
        check(r->state != State::HARVEST_WAIT, "the passed deadline ends the harvest");
    }
    // Waits for the planned burst's fence, then runs it (nothing is in flight).
    void fence() {
        until([&] { return r->state == State::FENCE; }, "planning opens the fence");
        window_controller_tick(inputs());
    }
    std::string journal() const {
        std::ifstream in(dir / "controller-journal.jsonl");
        std::stringstream text; text << in.rdbuf();
        return text.str();
    }
    size_t events(const std::string& name) const {
        const std::string text = journal(), needle = "{\"event\":\"" + name + "\"";
        size_t count = 0;
        for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++count;
        return count;
    }
    void check_journal_clock() const {
        std::istringstream lines(journal());
        std::string line;
        uint64_t previous = 0, count = 0;
        while (std::getline(lines, line)) {
            const auto at = line.find("\"mono_ns\":");
            check(at != std::string::npos, "every journal line carries mono_ns");
            const uint64_t value = std::stoull(line.substr(at + 10));
            check(value >= previous, "mono_ns never decreases");
            previous = value; ++count;
        }
        check(count > 0, "journal written");
    }
    // Test teardown: the admin controller-stop sends the all-rank STOP first.
    void stop() {
        check(!window_controller_stop(), "stop refuses while the servers' stats windows are open");
        r->stats_off_confirmed = true;
        check(window_controller_stop(), "resolved controller stops");
        std::filesystem::remove_all(dir);
    }
    void check_failed(const char* reason) const {
        check(r->state == State::RECOVERY_FAILED && r->terminal_reason == reason, "terminal recovery state and reason");
        check(r->stopped_on_error && r->proposals_disabled, "no new intervention after a failure");
        check(window_controller_has_unresolved_action(*r), "recovery is still owed");
        r->stats_off_confirmed = true;
        check(!window_controller_stop() && window_controller_active() == r, "stop cannot discard the recovery identity");
    }
    // controller-abort after check_failed. While ownership is ambiguous it is
    // refused; `resolve` stands for the manual migration-accept or
    // migration-rollback whose success confirms ownership (the coordinator
    // then calls window_controller_confirm_ownership), which lifts the
    // dispatch freeze. Then the terminal controller ends without a coordinator
    // restart, its history is dropped and the window-epoch floor kept: a fresh
    // history binds again only at ownership epoch 0 (after reset).
    void check_abort(const std::function<bool()>& resolve = nullptr) {
        std::string why;
        if (r->ownership_ambiguous) {
            check(!window_controller_abort_allowed(*r, &why) && !window_controller_abort(&why) &&
                  window_controller_active() == r && why.find("ownership is ambiguous") != std::string::npos &&
                  events("abort_refused") == 1, "abort is refused while ownership is ambiguous");
            check(resolve && resolve(), "the manual migration command completes on every server");
            window_controller_confirm_ownership(stubs.epoch, "manual");
            check(!r->ownership_ambiguous && !window_controller_dispatch_frozen() && controller_dispatch_allowed() &&
                  !controller_dispatch_stalled(inputs(), nullptr) && events("ownership_confirmed") == 1,
                  "a confirmed migration command lifts the dispatch freeze");
            check(r->stopped_on_error && r->state == State::RECOVERY_FAILED && !window_controller_stop(),
                  "the controller stays terminal and controller-stop stays refused");
        }
        check(window_controller_abort_allowed(*r, &why) && why.empty(), "known ownership: abort allowed");
        const uint64_t floor = r->window_epoch;
        check(window_controller_abort(&why) && why.empty() && window_controller_active() == nullptr &&
              events("controller_aborted") == 1, "controller-abort ends a terminal controller with known ownership");
        r = nullptr;
        auto restart = config;
        restart.start_ownership_epoch = stubs.epoch;
        std::string error;
        auto* again = window_controller_start(restart, nodes(), &error);
        if (stubs.epoch == 0) {
            check(again && again->action_history->expected_epoch == 0 && again->action_history->pending.empty() &&
                  again->action_history->total_spent == 0 && again->window_epoch == floor,
                  "at epoch 0 a new controller binds a fresh history above the kept window-epoch floor");
            again->stats_off_confirmed = true;
            check(window_controller_stop(), "the new controller stops");
        } else {
            check(again == nullptr && error.find("epoch 0") != std::string::npos,
                  "after an abort at a later epoch the dropped history cannot be rebuilt without reset");
        }
        std::filesystem::remove_all(dir);
    }
    // An ownership-ambiguous terminal: dispatch stays frozen, so the pipeline
    // must end the command once nothing is in flight (and only then).
    void check_dispatch_stalls(const char* reason) const {
        check(r->ownership_ambiguous && window_controller_dispatch_frozen() && !controller_dispatch_allowed(),
              "unconfirmed ownership freezes dispatch");
        ControllerTickInputs busy = inputs();
        busy.logical_in_flight = 1;
        check(!controller_dispatch_stalled(busy, nullptr), "a query still in flight may complete first");
        busy = inputs(); busy.queued_responses = 1;
        check(!controller_dispatch_stalled(busy, nullptr), "a queued reply is still handled first");
        std::string why;
        check(controller_dispatch_stalled(inputs(), &why) && why == reason,
              "frozen dispatch with nothing in flight ends the command, naming the terminal reason");
    }
};

void move_then_accept() {
    TestRig h(kToken);
    check(h.r->config.repair.weight == WindowRepairWeight::W, "the token names no weight: the default W");
    h.window(kTrigger);
    check(h.r->state == State::PLAN_WAIT || h.r->state == State::FENCE, "a skewed trigger window plans");
    check(stubs.harvest_sections == std::vector<uint32_t>({kWindowStructureOwnerHist}),
          "weight W requests the owner histogram only");
    check(h.journal().find("\"structure_sections\":2,\"repair_config\"") != std::string::npos &&
          h.journal().find("\"verify_invalid_limit\":3,\"weight\":\"w\"}") != std::string::npos &&
          h.journal().find("\"event\":\"harvest_requested\"") != std::string::npos &&
          h.journal().find(",\"structure_sections\":2}") != std::string::npos &&
          h.journal().find("\"structure_sections\":2,\"weight\":\"w\",\"owners\":4") != std::string::npos &&
          h.journal().find("nb_work_total") == std::string::npos,
          "start, request and merge journal weight w with sections 2 and no nbwork total");
    h.fence();
    check(stubs.prepares == 1 && stubs.commits == 1 && stubs.accepts == 1 && stubs.rollbacks == 0,
          "one PREPARE, COMMIT, ACCEPT");
    check(stubs.accepted.size() == 1 && stubs.accepted[0].moves.size() == 1 &&
          stubs.accepted[0].moves[0].global_id == 0 && stubs.accepted[0].moves[0].from_partition == 0 &&
          stubs.accepted[0].moves[0].to_partition == 1 && stubs.accepted[0].moves[0].adjacency.empty(),
          "the planned MOVE is node 0 from rank 0 to rank 1, as a skeleton row");
    check(g_daemon.membership[0] == 1 && h.r->commits == 1 && h.r->verify_pending && h.r->moved_bytes == h.row(),
          "committed burst awaits its verdict");
    const auto& history = *h.r->action_history;
    check(history.committed && history.total_spent == h.row() && history.reverse_reserved == h.row(),
          "forward bytes charged, inverse bytes reserved");
    check(!window_controller_stop(), "stop refuses an unresolved burst");
    h.window(kKeep);
    check(!h.r->verify_pending && h.r->last_committed_actions.empty() && history.pending.empty() &&
          history.reverse_reserved == 0 && history.total_spent == h.row(), "accepted burst releases its reservation");
    check(h.events("verify_accepted") == 1 && h.events("repair_verdict") == 1 && h.events("verify_rejected") == 0,
          "verdict journaled as accepted");
    check(h.journal().find("\"recall_guard\":\"off\"") != std::string::npos, "verdict records the guard as off");
    check(h.r->proposals_disabled && h.events("observation_only") == 1, "the proposal limit leaves only observation");
    check(g_daemon.membership[0] == 1 && stubs.accepted.size() == 1, "the accepted MOVE stays");
    check(!window_controller_has_unresolved_action(*h.r) && !window_controller_dispatch_frozen(), "resolved");
    h.check_journal_clock();
    h.stop();
}

// Own-only storage: a MOVE also ships the node's vector, so a byte is row +
// vector. With 24 vector bytes per node every row costs 44 bytes (20 + 24) and
// the total is 352: the 0.125 balance budget still buys exactly the one MOVE,
// charged 44 bytes; the journal states the unit and carries the fence payload.
// The row bytes bind at the first merged harvest, not at start.
void vector_bytes_count_in_budgets() {
    TestRig h(kToken, 1150, 0.001, kK4, 24);
    check(h.r->total_bytes == 0 && h.r->row_bytes.empty() && h.r->nodes == 8, "no row bytes before a harvest");
    check(h.journal().find("\"vector_bytes_per_row\":24,\"row_bytes_unit\":\"adjacency+vector\"") !=
          std::string::npos, "controller_started states the byte unit");
    h.window(kTrigger);
    check(h.r->total_bytes == 8 * h.row() && h.r->row_bytes[3] == 20 + 24 && h.row() == 44 &&
          h.events("row_bytes_bound") == 1 &&
          h.journal().find("\"nodes\":8,\"total_bytes\":352,\"adjacency_bytes\":160,\"vector_bytes_per_row\":24") !=
          std::string::npos, "every row counts its histogram degree and its vector bytes");
    h.fence();
    check(stubs.accepted.size() == 1 && stubs.accepted[0].moves.size() == 1 && h.r->moved_bytes == h.row() &&
          h.r->action_history->total_spent == h.row(), "the one MOVE is charged row + vector bytes");
    const std::string journal = h.journal();
    check(journal.find("\"burst_bytes\":44") != std::string::npos &&
          journal.find("\"fence_payload_bytes\":100") != std::string::npos &&
          journal.find("\"vector_bytes\":24") != std::string::npos &&
          journal.find("\"exchange_deadline_s\":30.5") != std::string::npos &&
          journal.find("\"migration_s\":") != std::string::npos,
          "the committed event records the fence payload, its deadlines and its duration");
    h.window(kKeep);
    check(!window_controller_has_unresolved_action(*h.r), "resolved");
    h.stop();
}

void move_reject_inverse() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(g_daemon.membership[0] == 1 && h.r->verify_pending, "forward MOVE committed");
    h.window(kHopUp);
    check(h.r->state == State::FENCE && h.r->reverting && h.r->pending_actions.size() == 1 &&
          h.r->pending_actions[0].source_owner == 1 && h.r->pending_actions[0].target_owner == 0,
          "rejected burst opens the inverse MOVE fence");
    window_controller_tick(h.inputs());
    check(stubs.accepted.size() == 2 && stubs.accepted[1].moves.size() == 1 &&
          stubs.accepted[1].moves[0].global_id == 0 && stubs.accepted[1].moves[0].from_partition == 1 &&
          stubs.accepted[1].moves[0].to_partition == 0 && stubs.epoch == 2, "the inverse MOVE is a forward plan");
    check(g_daemon.membership == kStart, "membership restored");
    const auto& history = *h.r->action_history;
    check(!h.r->verify_pending && !h.r->reverting && h.r->reverts == 1 && h.r->commits == 1 &&
          history.pending.empty() && history.total_spent == 2 * h.row() && h.r->moved_bytes == 2 * h.row() &&
          history.expected_epoch == 2, "the inverse consumed exactly the reservation");
    check(h.events("verify_rejected") == 1 && h.events("reverted") == 1 && h.events("history_rejected_resolved") == 1,
          "rejection and revert journaled");
    check(h.journal().find("\"hop_ratio\":1.2") != std::string::npos, "rejected on the hop ratio");
    check(!window_controller_has_unresolved_action(*h.r), "resolved after the inverse MOVE");
    check(!controller_dispatch_stalled(h.inputs(), nullptr), "a resolved controller never stalls dispatch");
    h.check_journal_clock();
    h.stop();
}

// Long windows. The planner's own cooldown counts planning windows and a
// verify window does not advance it, so without the history's rejection memory
// the next planning window re-proposes the identical burst (node 0, rank 0 -> 1).
// The memory counts planning windows too, so it holds however long the windows
// last: here the configured period (window + cooldown) is 1 ms and 100 ms pass
// between the revert and the next planning window, far beyond the two-period
// memory a wall-clock rule would have kept. Node 0 cools down and the 0 -> 1
// direction backs off: the next burst moves the next-heaviest node instead.
void rejected_burst_not_reproposed_after_long_windows() {
    TestRig h(kToken, 0, 0.001);  // period 1 ms
    h.window(kTrigger);
    h.fence();
    check(stubs.accepted.size() == 1 && stubs.accepted[0].moves[0].global_id == 0, "first burst: node 0 to rank 1");
    h.window(kHopUp);
    check(h.r->reverting && h.events("verify_rejected") == 1, "rejected");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart && !window_controller_has_unresolved_action(*h.r), "inverse MOVE ran");
    const auto& history = *h.r->action_history;
    const auto key = window_history_direction_key(0, 0, 1);
    check(history.planning_windows == 1 && history.move_cooldowns.count(0) == 1 &&
          history.move_cooldowns.at(0) == 3 && history.direction_backoffs.count(key) == 1 &&
          history.direction_backoffs.at(key).failures == 1 && history.direction_backoffs.at(key).until_plan == 3,
          "the rejection is remembered per node and direction, for the next two planning windows");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // a long window cycle: 100 periods
    h.window(kTrigger);  // the next planning window
    h.fence();
    check(stubs.accepted.size() == 3, "the next window plans and commits a burst");
    const auto& next = stubs.accepted[2].moves;
    check(next.size() == 1 && next[0].global_id != 0, "the rejected node is not re-proposed");
    check(next[0].global_id == 1 && next[0].from_partition == 0 && next[0].to_partition == 1,
          "the next-heaviest node moves instead");
    check(h.journal().find("\"history_planning_window\":2,\"history_cooldown_nodes\":1,"
                           "\"history_backoff_directions\":1") != std::string::npos,
          "planning journals the rejection memory it applied");
    check(history.planning_windows == 2, "each installed plan counts one planning window");
    h.r->config.proposal_query_limit = 1;  // leave only the verdict
    h.window(kKeep);
    check(!window_controller_has_unresolved_action(*h.r), "the second burst resolves");
    h.stop();
}

void prepare_failure_recovers() {
    TestRig h(kToken, 0);
    h.window(kTrigger);
    stubs.prepare_ok = false;
    h.fence();
    check(stubs.prepares == 1 && stubs.commits == 0 && stubs.rollbacks == 1, "PREPARE failure rolls back");
    check(h.r->state == State::COOLDOWN && !h.r->verify_pending && h.r->action_history->pending.empty() &&
          h.r->action_history->total_spent == 0 && g_daemon.membership == kStart && h.r->commits == 0,
          "a failed PREPARE releases the admission and changes nothing");
    check(h.events("prepare_failed") == 1 && h.events("history_prepare_failed_resolved") == 1, "journaled");
    check(!window_controller_has_unresolved_action(*h.r), "nothing owed");
    // A burst cancelled before COMMIT backs off its directions (node 0 to rank 1)
    // for the next two planning windows: the next-heaviest node moves instead.
    stubs.prepare_ok = true;
    h.window(kTrigger);
    h.fence();
    check(stubs.accepted.size() == 1 && stubs.accepted[0].moves.size() == 1 &&
          stubs.accepted[0].moves[0].global_id == 1 && g_daemon.membership[0] == 0 &&
          g_daemon.membership[1] == 1 && h.r->verify_pending,
          "the next window plans and commits again, without the backed-off direction");
    check(h.journal().find("\"history_planning_window\":2,\"history_cooldown_nodes\":0,"
                           "\"history_backoff_directions\":1") != std::string::npos,
          "planning journals the backoff of the cancelled burst");
    h.r->config.proposal_query_limit = 1;  // leave only the verdict
    h.window(kKeep);
    check(!window_controller_has_unresolved_action(*h.r), "verdict resolves the second burst");
    h.stop();
}

// A PREPARE-failed burst backs off its direction for the next two planning
// windows and can be proposed again after that. PREPARE fails in planning
// windows 1-3: each proposes the heaviest rank-0 node whose 0 -> 1 direction is
// still open (node 0, then 1, then 2) and each failure backs that direction off.
// In planning window 4 node 0's backoff (until planning window 3) has passed,
// so node 0 -> rank 1 is proposed again, and this time it commits.
void prepare_failed_burst_reproposed_after_backoff() {
    TestRig h(kToken, 0);
    stubs.prepare_ok = false;
    for (uint32_t expected : {0u, 1u, 2u}) {
        h.window(kTrigger);
        h.fence();
        check(stubs.prepared.size() == expected + 1 && stubs.prepared.back().moves.size() == 1 &&
              stubs.prepared.back().moves[0].global_id == expected && stubs.prepared.back().moves[0].from_partition == 0 &&
              stubs.prepared.back().moves[0].to_partition == 1 && h.r->state == State::COOLDOWN &&
              !window_controller_has_unresolved_action(*h.r), "a failed PREPARE of the next open direction");
    }
    check(stubs.commits == 0 && g_daemon.membership == kStart && h.events("history_prepare_failed_resolved") == 3,
          "three PREPARE failures changed nothing");
    stubs.prepare_ok = true;
    h.window(kTrigger);
    h.fence();
    check(stubs.prepared.size() == 4 && stubs.accepted.size() == 1 && stubs.accepted[0].moves.size() == 1 &&
          stubs.accepted[0].moves[0].global_id == 0 && stubs.accepted[0].moves[0].to_partition == 1 &&
          g_daemon.membership[0] == 1 && h.r->verify_pending,
          "after its backoff node 0 -> rank 1 is proposed again and commits");
    check(h.journal().find("\"history_planning_window\":4,\"history_cooldown_nodes\":0,"
                           "\"history_backoff_directions\":2") != std::string::npos,
          "planning window 4 still backs off nodes 1 and 2, no longer node 0");
    h.r->config.proposal_query_limit = 1;  // leave only the verdict
    h.window(kKeep);
    check(!window_controller_has_unresolved_action(*h.r) && h.events("verify_accepted") == 1, "the burst resolves");
    h.stop();
}

// A rank that serves nothing in the verify windows (no distances or requests):
// its prices are undefined, so no verify window can be measured. Each one is
// counted; the first verify_invalid_limit - 1 (default limit 3) defer the
// verdict, the third resolves the burst as REJECTED and reverts it, and the
// controller can then stop.
void idle_rank_rejects_after_invalid_limit() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(h.r->verify_pending && g_daemon.membership[0] == 1, "burst committed");
    const Spec idle{{800, 300, 100, 0}, {100, 60, 50, 0}};
    h.window(idle);
    check(h.r->verify_pending && h.r->verify_invalid_windows == 1 && h.r->state == State::COOLDOWN &&
          h.events("verify_deferred_measurement") == 1 &&
          h.journal().find("\"verify_invalid_windows\":1,\"verify_invalid_limit\":3") != std::string::npos,
          "the first unmeasurable verify window defers and is counted");
    h.r->stats_off_confirmed = true;
    check(!window_controller_stop() && window_controller_active() == h.r, "stop is refused: the burst is unresolved");
    h.window(idle);
    check(h.r->verify_pending && h.r->verify_invalid_windows == 2 && h.events("verify_deferred_measurement") == 2,
          "the second defers too");
    h.window(idle);
    check(h.r->reverting && h.r->state == State::FENCE && h.events("verify_rejected") == 1 &&
          h.events("verify_deferred_measurement") == 2 && h.events("repair_verdict") == 1,
          "the third resolves the burst as rejected and opens the inverse MOVE");
    check(h.journal().find("\"reason\":\"verify-invalid-limit\",\"last_invalid\":\"measurement-invalid\","
                           "\"verify_invalid_windows\":3,\"verify_invalid_limit\":3") != std::string::npos,
          "the verdict journals why");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart && !h.r->verify_pending && !h.r->reverting && h.r->reverts == 1 &&
          h.r->repair_state.reverts == 1 && h.r->verify_invalid_windows == 0 && h.events("reverted") == 1 &&
          !window_controller_has_unresolved_action(*h.r), "the inverse MOVE ran: nothing is owed");
    h.stop();
}

// A verify window whose planning observation is invalid (the frames own the ids
// of the membership before the burst, so the merge refuses them) but whose ACK
// counters measure it: the burst is judged from the measurement, not counted as
// an unjudgeable window; planning then waits for the next observation.
void verdict_accepted_despite_invalid_observation() {
    TestRig h(kToken, 0);
    h.window(kTrigger);
    h.fence();
    check(h.r->verify_pending && g_daemon.membership[0] == 1, "burst committed");
    h.window(kKeep, &kStart);
    check(!h.r->verify_pending && h.r->verify_invalid_windows == 0 && h.events("verify_accepted") == 1 &&
          h.events("repair_verdict") == 1 && h.events("verify_deferred_measurement") == 0,
          "the burst is accepted from the measured window");
    check(h.events("structure_unavailable") == 1 && h.events("observation_evidence_unavailable") == 1 &&
          h.r->state == State::COOLDOWN, "the invalid observation only defers planning");
    check(h.journal().find("\"last_invalid\":\"observation-invalid\"") == std::string::npos,
          "no window is counted as unjudgeable");
    h.stop();
}

// Search commands shorter than window_sec: every window is abandoned when the
// command changes before it closes. While a burst awaits its verdict each such
// window counts toward verify_invalid_limit (3), so the burst is rejected and
// reverted at the third and the controller can stop (before, these windows
// never counted: the verdict, and controller-stop, could wait for ever).
void short_commands_reject_after_invalid_limit() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(h.r->verify_pending && g_daemon.membership[0] == 1, "burst committed");
    h.r->config.window_seconds = 1000.0;  // no window lasts its window_sec
    for (uint32_t lost = 1; lost <= 3; ++lost) {
        h.open_window();
        ++h.command; h.completed = 5;  // the next short command restarts the counters
        h.until([&] { return h.r->state != State::WINDOW_OPEN; }, "the window is abandoned");
        check(h.events("window_abandoned") == lost, "every window is abandoned");
        if (lost < 3)
            check(h.r->verify_pending && h.r->verify_invalid_windows == lost && h.r->state == State::COOLDOWN,
                  "an abandoned verify window is counted");
    }
    check(h.journal().find("\"reason\":\"search-command-changed\",\"verify_invalid_windows\":1,"
                           "\"verify_invalid_limit\":3") != std::string::npos, "journaled with the count");
    check(h.r->reverting && h.r->state == State::FENCE && h.events("verify_rejected") == 1 &&
          h.journal().find("\"reason\":\"verify-invalid-limit\",\"last_invalid\":\"window-abandoned\","
                           "\"verify_invalid_windows\":3") != std::string::npos,
          "the third abandoned window resolves the burst as rejected");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart && !window_controller_has_unresolved_action(*h.r), "the inverse MOVE ran");
    h.stop();
}

// The other windows that cannot judge a pending burst count the same way: a
// dump directory that cannot be created (window 1), a search command that
// changed while the harvest was decoded and measured (window 2), and BEGIN
// ACKs that arrive in the next command (window 3) reach the limit.
void lost_windows_count_toward_invalid_limit() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(h.r->verify_pending, "burst committed");
    // Window 1: a regular file where the window's dump directory goes.
    h.open_window();
    std::ofstream(h.dir / ("window-" + std::to_string(h.r->window_epoch))) << "blocks the directory";
    for (uint32_t rank = 0; rank < h.topo.ranks; ++rank) {
        h.counters[rank].distances += kKeep.D[rank]; h.counters[rank].distance_us += kKeep.D[rank];
        h.counters[rank].requests += kKeep.R[rank]; h.counters[rank].total_us += kKeep.D[rank] + kKeep.R[rank];
    }
    h.completed += kKeep.queries;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    h.until([&] { return h.r->state != State::WINDOW_OPEN; }, "the dump directory fails");
    check(h.events("dump_dir_failed") == 1 && h.r->verify_pending && h.r->verify_invalid_windows == 1 &&
          h.r->state == State::COOLDOWN, "a failed dump directory is counted");
    // Window 2: every frame arrived and the endpoints froze (first tick after the
    // arrivals; the load of the merged structure is asynchronous, so the state
    // is still HARVEST_WAIT), then the command changes before the measurement.
    h.serve(kKeep);
    h.deliver_frames(g_daemon.membership);
    window_controller_tick(h.inputs());
    check(h.r->state == State::HARVEST_WAIT && h.r->native_control->endpoints_frozen,
          "endpoints frozen in the window's command, not yet measured");
    ++h.command; h.completed = 7;
    h.until([&] { return h.r->state != State::HARVEST_WAIT; }, "harvest loaded and handled");
    check(h.journal().find("\"reason\":\"measurement-command-changed\",\"verify_invalid_windows\":1") !=
          std::string::npos && h.r->verify_pending && h.r->verify_invalid_windows == 2, "counted");
    // Window 3: BEGIN goes out in one command, its ACKs arrive in the next.
    h.until([&] { return h.r->state == State::BEGIN_WAIT; }, "BEGIN sent");
    ++h.command; h.completed = 3;
    for (uint32_t rank = 0; rank < h.topo.ranks; ++rank) {
        const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, h.r->window_epoch,
                                        h.counters[rank]);
        controller_offer(rank, &a, sizeof(a));
    }
    h.until([&] { return h.r->state != State::BEGIN_WAIT; }, "BEGIN ACKs handled");
    check(h.r->reverting && h.r->state == State::FENCE && h.events("verify_rejected") == 1 &&
          h.journal().find("\"reason\":\"verify-invalid-limit\",\"last_invalid\":\"window-abandoned\","
                           "\"verify_invalid_windows\":3") != std::string::npos,
          "the third lost window resolves the burst as rejected");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart && !window_controller_has_unresolved_action(*h.r), "the inverse MOVE ran");
    h.stop();
}

// A dump directory that stays unwritable: a regular file sits
// where each verify window's directory goes, so all three windows fail with
// dump_dir_failed, and the third reaches verify_invalid_limit; its revert
// cannot write revert-plan.bin into that same directory either. The record
// failure is journaled and the fence PREPAREs the inverse plan the controller
// holds in memory: the burst is reverted and controller-stop succeeds.
void persistent_dump_dir_failure_reverts() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(h.r->verify_pending && g_daemon.membership[0] == 1, "burst committed");
    check(stubs.prepared_from_record.size() == 1 && stubs.prepared_from_record[0],
          "the forward plan file records the PREPAREd bytes");
    const unsigned prepares = stubs.prepares;
    for (uint32_t lost = 1; lost <= 3; ++lost) {
        h.open_window();
        std::ofstream(h.dir / ("window-" + std::to_string(h.r->window_epoch))) << "blocks the directory";
        for (uint32_t rank = 0; rank < h.topo.ranks; ++rank) {
            h.counters[rank].distances += kKeep.D[rank]; h.counters[rank].distance_us += kKeep.D[rank];
            h.counters[rank].requests += kKeep.R[rank]; h.counters[rank].total_us += kKeep.D[rank] + kKeep.R[rank];
        }
        h.completed += kKeep.queries;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        h.until([&] { return h.r->state != State::WINDOW_OPEN; }, "the dump directory fails");
        check(h.events("dump_dir_failed") == lost, "every verify window's dump directory fails");
        if (lost < 3)
            check(h.r->verify_pending && h.r->verify_invalid_windows == lost && h.r->state == State::COOLDOWN,
                  "a failed dump directory is counted");
    }
    check(h.events("revert_plan_record_failed") == 1 && h.events("revert_plan_write_failed") == 0 &&
          !h.r->stopped_on_error && h.r->reverting && h.r->state == State::FENCE &&
          h.r->pending_plan_path.empty() && !h.r->pending_plan.empty() && h.events("verify_rejected") == 1,
          "the record failure is journaled and the revert goes on from the plan in memory");
    check(h.journal().find("\"reason\":\"verify-invalid-limit\",\"last_invalid\":\"dump-dir-failed\","
                           "\"verify_invalid_windows\":3") != std::string::npos,
          "the third failed dump directory resolves the burst as rejected");
    window_controller_tick(h.inputs());
    check(stubs.prepares == prepares + 1 && !stubs.prepared_from_record.back() && stubs.accepted.size() == 2 &&
          stubs.accepted[1].moves.size() == 1 && stubs.accepted[1].moves[0].from_partition == 1 &&
          stubs.accepted[1].moves[0].to_partition == 0, "the inverse MOVE was PREPAREd from memory");
    check(g_daemon.membership == kStart && h.events("reverted") == 1 && !h.r->stopped_on_error &&
          !window_controller_has_unresolved_action(*h.r), "the burst is reverted: nothing is owed");
    h.stop();
}

// A plan record that fails only at its close: a full disk, simulated by a record
// that is a link to /dev/full, so the open and the buffered write of a small plan
// succeed and the flush at the close fails. The failure is journaled, the record
// removed, and the revert goes on from the plan held in memory.
void revert_record_fails_at_close() {
    if (!std::filesystem::exists("/dev/full")) { std::printf("(no /dev/full: skipped) "); return; }
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(g_daemon.membership[0] == 1 && h.r->verify_pending, "forward MOVE committed");
    const unsigned prepares = stubs.prepares;
    h.serve(kHopUp);
    const auto record = h.dir / ("window-" + std::to_string(h.r->window_epoch)) / "revert-plan.bin";
    std::filesystem::create_symlink("/dev/full", record);
    h.deliver_frames(g_daemon.membership);
    h.until([&] { return h.r->state != State::HARVEST_WAIT; }, "harvest decoded, loaded and handled");
    check(h.events("revert_plan_record_failed") == 1 && h.events("verify_rejected") == 1 && h.r->reverting &&
          h.r->state == State::FENCE && h.r->pending_plan_path.empty() && !h.r->pending_plan.empty() &&
          !h.r->stopped_on_error && !std::filesystem::exists(std::filesystem::symlink_status(record)),
          "a revert record that fails at its close is journaled and removed; the revert goes on");
    window_controller_tick(h.inputs());
    check(stubs.prepares == prepares + 1 && !stubs.prepared_from_record.back() && h.events("reverted") == 1 &&
          g_daemon.membership == kStart && !window_controller_has_unresolved_action(*h.r),
          "the inverse MOVE was PREPAREd from memory and ran");
    h.stop();
}

// The forward plan's record failing at its close: plan_write_failed, cooldown,
// no fence and no PREPARE, and the failed record is removed.
void plan_record_fails_at_close() {
    if (!std::filesystem::exists("/dev/full")) { std::printf("(no /dev/full: skipped) "); return; }
    TestRig h(kToken);
    h.serve(kTrigger);
    const auto record = h.dir / ("window-" + std::to_string(h.r->window_epoch)) / "plan.bin";
    std::filesystem::create_symlink("/dev/full", record);
    h.deliver_frames(g_daemon.membership);
    h.until([&] { return h.events("plan_write_failed") > 0 || h.r->state == State::FENCE; },
            "the planned burst's record is written");
    check(h.events("plan_write_failed") == 1 && h.r->state != State::FENCE && h.events("fence_open") == 0 &&
          stubs.prepares == 0 && h.r->pending_plan_path.empty() && h.r->commits == 0 &&
          !std::filesystem::exists(std::filesystem::symlink_status(record)),
          "a plan record that fails at its close is journaled and removed, and no fence opens");
    h.stop();
}

// The same with a burst the measurement rejects: rejected and reverted at once.
void verdict_rejected_despite_invalid_observation() {
    TestRig r2(kToken, 0);
    r2.window(kTrigger);
    r2.fence();
    r2.window(kHopUp, &kStart);
    check(r2.r->reverting && r2.r->state == State::FENCE && r2.events("verify_rejected") == 1 &&
          r2.r->verify_invalid_windows == 0, "a measurable rejection reverts at once");
    window_controller_tick(r2.inputs());
    check(g_daemon.membership == kStart && !window_controller_has_unresolved_action(*r2.r), "inverse MOVE ran");
    r2.stop();
}

// A failed BEGIN while a burst awaits its verdict is a window that cannot judge
// it: counted, and at verify_invalid_limit (3) the burst is rejected and reverted.
// The third BEGIN never opened a window, so its dump directory did not exist:
// the revert creates it for its plan (this test first found the revert failing
// terminally with revert_plan_write_failed).
void begin_failures_count_toward_invalid_limit() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(h.r->verify_pending, "burst committed");
    h.begin_refused();
    check(h.r->verify_pending && h.r->verify_invalid_windows == 1 && h.events("window_begin_failed") == 1 &&
          h.r->state == State::COOLDOWN, "the first failed BEGIN is counted");
    h.begin_refused();
    check(h.r->verify_pending && h.r->verify_invalid_windows == 2, "the second too");
    h.begin_refused();
    check(h.events("revert_plan_record_failed") == 0 && !h.r->stopped_on_error && !h.r->pending_plan_path.empty(),
          "the revert plan is recorded");
    check(h.r->reverting && h.r->state == State::FENCE && h.events("verify_rejected") == 1 &&
          h.events("window_begin_failed") == 3, "the third resolves the burst as rejected");
    check(h.journal().find("\"reason\":\"verify-invalid-limit\",\"last_invalid\":\"window-begin-failed\","
                           "\"verify_invalid_windows\":3") != std::string::npos, "journaled with the BEGIN failure");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart && !window_controller_has_unresolved_action(*h.r) &&
          h.r->verify_invalid_windows == 0, "the inverse MOVE ran: nothing is owed");
    h.stop();
}

// Both harvest_failed paths count while a burst awaits its verdict: a refused
// exchange (window 1), the decode-or-load deadline after every frame validated
// (window 2), and a refused exchange again (window 3) reaches the limit.
void harvest_failures_count_toward_invalid_limit() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(h.r->verify_pending, "burst committed");
    h.harvest_refused(kKeep);
    check(h.r->verify_pending && h.r->verify_invalid_windows == 1 && h.events("harvest_failed") == 1 &&
          h.journal().find("\"rejected\":true") != std::string::npos, "a refused HARVEST is counted");
    h.harvest_load_deadline(kKeep);
    check(h.r->verify_pending && h.r->verify_invalid_windows == 2 && h.events("harvest_failed") == 2 &&
          h.journal().find("\"reason\":\"decode-or-load-deadline\"") != std::string::npos,
          "the decode-or-load deadline is counted");
    h.harvest_refused(kKeep);
    check(h.r->reverting && h.r->state == State::FENCE && h.events("verify_rejected") == 1 &&
          h.events("harvest_failed") == 3, "the third failed harvest resolves the burst as rejected");
    check(h.journal().find("\"reason\":\"verify-invalid-limit\",\"last_invalid\":\"harvest-failed\","
                           "\"verify_invalid_windows\":3") != std::string::npos, "journaled with the harvest failure");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart && !window_controller_has_unresolved_action(*h.r), "the inverse MOVE ran");
    h.stop();
}

void inverse_prepare_failure_is_terminal() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    h.window(kHopUp);
    check(h.r->reverting, "inverse fence open");
    stubs.prepare_ok = false;
    window_controller_tick(h.inputs());
    check(!h.r->ownership_ambiguous && !window_controller_dispatch_frozen(), "the rollback succeeded: ownership known");
    check(h.r->verify_pending && g_daemon.membership[0] == 1, "the rejected burst is still in place");
    h.check_failed("revert_prepare_failed");
    h.check_abort();
    check(g_daemon.membership[0] == 1 && stubs.epoch == 1, "the aborted controller leaves the rejected burst in place");
}

void commit_failure_is_terminal() {
    TestRig h(kToken);
    h.window(kTrigger);
    stubs.commit_ok = false;
    h.fence();
    check(stubs.commits == 1 && stubs.accepts == 0 && stubs.rollbacks == 1 && g_daemon.membership == kStart,
          "a failed COMMIT rolls back");
    check(!h.r->ownership_ambiguous, "rollback confirmed");
    h.check_failed("commit_failed");
    h.check_abort();
}

void commit_and_rollback_failure_freezes_dispatch() {
    TestRig h(kToken);
    h.window(kTrigger);
    stubs.commit_ok = false; stubs.rollback_ok = false;
    h.fence();
    h.check_dispatch_stalls("commit_failed");
    h.check_failed("commit_failed");
    h.check_abort([] { stubs.rollback_ok = true; return coordinator_membership_migration_rollback(); });
    check(g_daemon.membership == kStart && stubs.epoch == 0, "the manual rollback restored ownership");
}

void accept_failure_freezes_dispatch() {
    TestRig h(kToken);
    h.window(kTrigger);
    stubs.accept_ok = false;
    h.fence();
    check(stubs.accepts == 1 && stubs.rollbacks == 0, "a failed ACCEPT is not rolled back");
    h.check_dispatch_stalls("accept_failed");
    h.check_failed("accept_failed");
    h.check_abort([] { stubs.accept_ok = true; return coordinator_membership_migration_accept(); });
    check(stubs.accepted.size() == 1 && stubs.epoch == 1 && g_daemon.membership[0] == 1,
          "the manual ACCEPT completed the burst; the aborted controller leaves it in place");
}

// A PREPARE that some servers applied, then a rollback that fails: ownership
// is unknown, so dispatch freezes (prepare_rollback_failed).
void prepare_failure_with_failed_rollback_freezes_dispatch() {
    TestRig h(kToken);
    h.window(kTrigger);
    stubs.prepare_ok = false; stubs.prepare_failure_leaves_plan = true; stubs.rollback_ok = false;
    h.fence();
    check(stubs.prepares == 1 && stubs.commits == 0 && stubs.rollbacks == 1, "PREPARE failed, rollback attempted");
    check(h.events("prepare_failed") == 1 && !h.r->verify_pending && g_daemon.membership == kStart,
          "nothing committed");
    h.check_dispatch_stalls("prepare_rollback_failed");
    h.check_failed("prepare_rollback_failed");
    h.check_abort([] { stubs.rollback_ok = true; return coordinator_membership_migration_rollback(); });
}

// The inverse MOVE's COMMIT fails and its rollback succeeds: the rejected
// burst is still in place and owed, ownership is known (no freeze).
void inverse_commit_failure_is_terminal() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    h.window(kHopUp);
    check(h.r->reverting, "inverse fence open");
    stubs.commit_ok = false;
    window_controller_tick(h.inputs());
    check(stubs.commits == 2 && stubs.rollbacks == 1 && stubs.accepted.size() == 1, "inverse COMMIT failed, rolled back");
    check(!h.r->ownership_ambiguous && !window_controller_dispatch_frozen() &&
          !controller_dispatch_stalled(h.inputs(), nullptr), "the rollback succeeded: ownership known");
    check(h.r->verify_pending && h.r->reverting && g_daemon.membership[0] == 1, "the rejected burst is still in place");
    h.check_failed("commit_failed");
    h.check_abort();
}

// The inverse MOVE committed but its ACCEPT failed: dispatch freezes.
void inverse_accept_failure_freezes_dispatch() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    h.window(kHopUp);
    check(h.r->reverting, "inverse fence open");
    stubs.accept_ok = false;
    window_controller_tick(h.inputs());
    check(stubs.accepts == 2 && stubs.rollbacks == 0 && g_daemon.membership == kStart,
          "the inverse committed; its ACCEPT failed and is not rolled back");
    check(h.r->verify_pending && h.r->reverting, "the revert is not recorded as done");
    h.check_dispatch_stalls("accept_failed");
    h.check_failed("accept_failed");
    h.check_abort([] { stubs.accept_ok = true; return coordinator_membership_migration_accept(); });
    check(stubs.accepted.size() == 2 && stubs.epoch == 2 && g_daemon.membership == kStart,
          "the manual ACCEPT completed the inverse MOVE");
}

const std::string kGuard = kToken + ",recall_guard:0.10";

void guard_requires_scored_trigger() {
    TestRig h(kGuard, 1250);  // plans at 1,200 completed queries, judges at 1,300
    h.window(kTrigger);
    check(h.r->state == State::COOLDOWN && h.events("recall_guard_unscored") == 1 && h.events("planning_submitted") == 0,
          "an unscored window cannot trigger a guarded burst");
    h.window(scored(kTrigger, 90.0));
    h.fence();
    check(stubs.accepted.size() == 1 && h.r->verify_pending && h.r->verify_recall_percent == 90.0,
          "a scored window plans; its recall is the guard's reference");
    h.window(scored(kKeep, 80.0));
    check(h.r->reverting && h.events("verify_rejected") == 1, "recall 80 < 90 - 0.10 rejects despite the costs");
    check(h.journal().find("\"recall_guard\":0.10000000000000001,\"recall\":80,\"reference\":90") != std::string::npos,
          "verdict carries guard, recall and reference");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart && !window_controller_has_unresolved_action(*h.r), "inverse MOVE ran");
    h.stop();
}

void guard_fails_closed_on_unscored_verify() {
    TestRig h(kGuard);
    h.window(scored(kTrigger, 90.0));
    h.fence();
    h.window(kKeep);  // the verify window has no ground truth
    check(h.r->reverting && h.events("verify_rejected") == 1, "a guarded verdict without recall rejects");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kStart, "inverse MOVE ran");
    h.stop();
}

void guard_off_ignores_recall() {
    TestRig h(kToken);
    h.window(scored(kTrigger, 90.0));
    h.fence();
    h.window(scored(kKeep, 80.0));
    check(!h.r->verify_pending && h.events("verify_accepted") == 1, "without the guard recall never decides");
    h.stop();
}

void command_change_abandons_window() {
    TestRig h(kToken);
    h.until([&] { return h.r->state == State::BEGIN_WAIT; }, "BEGIN sent");
    for (uint32_t rank = 0; rank < h.topo.ranks; ++rank) {
        const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, h.r->window_epoch, h.counters[rank]);
        controller_offer(rank, &a, sizeof(a));
    }
    h.until([&] { return h.r->state == State::WINDOW_OPEN; }, "window open");
    ++h.command; h.completed = 5;  // a new search command restarts the counters
    h.until([&] { return h.r->state != State::WINDOW_OPEN; }, "window abandoned");
    check(h.events("window_abandoned") == 1 && h.r->state == State::COOLDOWN, "a window never spans two commands");
    h.r->stats_off_confirmed = true;
    check(window_controller_stop(), "stop");
}

// BEGIN goes out at the tail of one search command and its ACKs arrive in the
// next: the server counters started in the first command, the completed-query
// and recall counters would start in the second. The window's command is the
// one BEGIN was sent in, so the window is abandoned instead of opened.
void begin_acks_in_next_command_abandon_window() {
    TestRig h(kToken);
    h.until([&] { return h.r->state == State::BEGIN_WAIT; }, "BEGIN sent");
    const uint64_t begin_command = h.command;
    ++h.command; h.completed = 3;  // the next command starts before the ACKs arrive
    for (uint32_t rank = 0; rank < h.topo.ranks; ++rank) {
        const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, h.r->window_epoch, h.counters[rank]);
        controller_offer(rank, &a, sizeof(a));
    }
    h.until([&] { return h.r->state != State::BEGIN_WAIT; }, "BEGIN ACKs handled");
    check(h.r->state == State::COOLDOWN && h.events("window_open") == 0 && h.events("window_abandoned") == 1,
          "the window is not opened in a command it did not begin in");
    check(h.journal().find("\"reason\":\"search-command-changed-before-begin-ack\",\"begin_command_id\":" +
                           std::to_string(begin_command)) != std::string::npos, "journaled with both commands");
    // The next window begins and opens inside the new command.
    h.until([&] { return h.r->state == State::BEGIN_WAIT; }, "next BEGIN sent");
    for (uint32_t rank = 0; rank < h.topo.ranks; ++rank) {
        const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, h.r->window_epoch, h.counters[rank]);
        controller_offer(rank, &a, sizeof(a));
    }
    h.until([&] { return h.r->state == State::WINDOW_OPEN; }, "the next window opens");
    check(h.r->window_begin_command == h.command, "the open window belongs to the new command");
    h.r->stats_off_confirmed = true;
    check(window_controller_stop(), "stop");
}

// K = 8 servers. 16 nodes, 36-byte rows (4 + 4 x 8; 576 bytes); rank 0 holds nodes 0..8
// (work 40, 30, 20, 10 x 6), ranks 1..7 one node each (work 5). pred = work /
// 100 queries, price 1: W = {1.5, .05 x 7}, mean .23125, limit 1.03 x mean =
// .2381875. Balance budget .0625 x 576 = 36 bytes: one MOVE. balance_target
// underloaded (opted in; the default is lightest): nodes 0, 1, 2 would lift a rank-1..7 owner to .45,
// .35, .25 > limit; node 3 fits (.15); every owner holds one neighbour of it
// and the same load, so the lowest index wins: node 3 -> rank 1. (The K = 4
// rule would have moved node 0 to the single lightest rank.)
const Topology kK8{8, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7},
                   {40, 30, 20, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5}};
const std::string kToken8 = "algo:local,refine:0,balance:0.0625,tau:0,eps:0.03,skew:1.02,cooldown:1,cum:1,"
                            "slack:0.10,hop_slack:0.05,ema:0.5,converge:3,guard:10,balance_target:underloaded";
// Trigger: costs per query 11 / 1.5 x 7 (total 21.5), hop 4.5. Keep: 9 / 3.6 / 1.5 x 6
// (total 21.6, 1.0047), hop 4.6 (1.022). Hop up: hop 5.0 (1.11 > 1.05).
const Spec kTrigger8{{1000, 100, 100, 100, 100, 100, 100, 100}, {100, 50, 50, 50, 50, 50, 50, 50}};
const Spec kKeep8{{800, 300, 100, 100, 100, 100, 100, 100}, {100, 60, 50, 50, 50, 50, 50, 50}};
const Spec kHopUp8{{800, 300, 100, 100, 100, 100, 100, 100}, {100, 100, 50, 50, 50, 50, 50, 50}};

void k8_underloaded_reject_then_accept() {
    TestRig h(kToken8, 0, 0.001, kK8);
    h.window(kTrigger8);
    h.fence();
    check(stubs.accepted.size() == 1 && stubs.accepted[0].n_servers == 8 && stubs.accepted[0].moves.size() == 1 &&
          stubs.accepted[0].moves[0].global_id == 3 && stubs.accepted[0].moves[0].from_partition == 0 &&
          stubs.accepted[0].moves[0].to_partition == 1 && h.r->verify_pending,
          "K = 8: the underloaded rule moves node 3 to rank 1 in an eight-server plan");
    check(h.journal().find("\"ranks\":8") != std::string::npos &&
          h.journal().find("\"balance_target\":\"underloaded\"") != std::string::npos,
          "eight ranks; the plan journals the underloaded balance target");
    h.window(kHopUp8);
    check(h.r->reverting && h.events("verify_rejected") == 1 && h.r->pending_actions.size() == 1 &&
          h.r->pending_actions[0].source_owner == 1 && h.r->pending_actions[0].target_owner == 0,
          "rejected on the hop ratio: the inverse MOVE");
    window_controller_tick(h.inputs());
    check(g_daemon.membership == kK8.start && !window_controller_has_unresolved_action(*h.r), "inverse MOVE ran");
    // Planning window 2: node 3 cools down and its 0 -> 1 direction backs off; node 4 moves instead.
    h.window(kTrigger8);
    h.fence();
    check(stubs.accepted.size() == 3 && stubs.accepted[2].moves.size() == 1 && stubs.accepted[2].moves[0].global_id == 4 &&
          stubs.accepted[2].moves[0].to_partition == 1 && g_daemon.membership[4] == 1 && h.r->verify_pending,
          "the rejection memory holds at K = 8: node 4 moves instead of node 3");
    check(h.journal().find("\"history_planning_window\":2,\"history_cooldown_nodes\":1,"
                           "\"history_backoff_directions\":1") != std::string::npos, "memory journaled");
    h.r->config.proposal_query_limit = 1;  // leave only the verdict
    h.window(kKeep8);
    check(!window_controller_has_unresolved_action(*h.r) && h.events("verify_accepted") == 1 &&
          g_daemon.membership[4] == 1 && h.r->commits == 2 && h.r->reverts == 1, "the second burst is accepted");
    h.check_journal_clock();
    h.stop();
}

// weight:wnb: the request and the merge carry the nbwork section (sections 6);
// the fixture's nbwork is zero, so the plan is the same single MOVE.
void wnb_move_then_accept() {
    TestRig h(kToken + ",weight:wnb");
    check(h.r->config.repair.weight == WindowRepairWeight::WNB, "weight:wnb parses");
    h.window(kTrigger);
    h.fence();
    check(stubs.harvest_sections == std::vector<uint32_t>({kWindowStructureOwnerHist | kWindowStructureNbWork}),
          "weight WNB requests the owner histogram and nbwork");
    check(stubs.accepted.size() == 1 && stubs.accepted[0].moves.size() == 1 &&
          stubs.accepted[0].moves[0].global_id == 0 && stubs.accepted[0].moves[0].to_partition == 1,
          "the same MOVE: node 0 to rank 1");
    check(h.journal().find("\"structure_sections\":6,\"weight\":\"wnb\",\"owners\":4") != std::string::npos &&
          h.journal().find("\"nb_work_total\":0") != std::string::npos &&
          h.journal().find("\"weight\":\"wnb\"}") != std::string::npos,
          "the merge journals sections 6 and the nbwork total; the config says wnb");
    h.window(kKeep);
    check(h.events("verify_accepted") == 1 && !window_controller_has_unresolved_action(*h.r), "accepted");
    h.stop();
}

// A weight-W controller decodes frames against the owner histogram alone: a
// frame that also carries nbwork never satisfies a rank (snapshot_rejected),
// the harvest fails and nothing is planned.
void weight_w_refuses_nbwork_frames() {
    TestRig h(kToken, 0);  // no proposal limit: the second window still plans
    h.serve(kTrigger);
    h.deliver_frames(g_daemon.membership, kWindowStructureOwnerHist | kWindowStructureNbWork);
    h.until([&] { return h.r->state != State::HARVEST_WAIT; }, "the refused harvest is handled");
    check(h.events("snapshot_rejected") >= 1 && h.events("harvest_failed") == 1 && h.events("planning_submitted") == 0 &&
          stubs.prepares == 0, "an nbwork section refuses the frame under weight W; nothing is planned");
    h.window(kTrigger);  // the next window with owner-histogram frames plans again
    check(h.events("planning_submitted") == 1, "the next well-formed harvest plans");
    h.fence();
    h.r->config.proposal_query_limit = 1;  // leave only the verdict
    h.window(kKeep);
    check(!window_controller_has_unresolved_action(*h.r), "resolved");
    h.stop();
}

// A second controller phase after inserts. The first phase moves node 0 (rank 0 ->
// 1) and accepts it, then stops. Two ids are "inserted" (8 on rank 0, 9 on
// rank 2) and the rows change (two neighbours per owner: 36-byte rows). The
// second phase binds the same history (configuration-only context) over 10 ids: its caps
// restart, its row bytes bind from its first harvest (10 x 36 = 360), the
// planner's EMA carries the 8 known ids and pads the 2 new ones (node 3: work
// 30 now, pred 0.5 x 0.3 + 0.5 x 0.1 = 0.2, not a reset's 0.3; node 8: its
// first measurement 0.45), and the carried cooldowns are padded. Balance
// (0.125 x 360 = 45 bytes): rank 0 (1.25) to the lightest rank 3 (0.05), gap
// 1.2: node 8 (0.45 <= 0.6) moves first, then rank 0 (0.8) to rank 2 (0.1):
// node 1 (0.3 <= 0.35); 72 bytes reach the budget. An inserted id moves.
void second_phase_binds_more_nodes() {
    TestRig h(kToken);
    h.window(kTrigger);
    h.fence();
    check(g_daemon.membership[0] == 1 && h.events("row_bytes_bound") == 1 &&
          h.journal().find("\"nodes\":8,\"total_bytes\":160,") != std::string::npos, "phase 1 binds 8 x 20 bytes");
    h.window(kKeep);
    check(!window_controller_has_unresolved_action(*h.r) && h.r->proposals_disabled, "phase 1 resolved");
    const auto history = h.r->action_history;
    h.r->stats_off_confirmed = true;
    check(window_controller_stop(), "phase 1 stops");
    // The insert phase: two more ids, other rows, a new search command.
    g_daemon.membership.push_back(0); g_daemon.membership.push_back(2);
    h.topo.start.push_back(0); h.topo.start.push_back(2);
    h.topo.work.push_back(45); h.topo.work.push_back(5);
    h.topo.work[3] = 30;
    h.hist_row = {2, 2, 2, 2};
    h.completed = 1000; ++h.command;
    h.config.start_ownership_epoch = stubs.epoch;
    std::string error;
    h.r = window_controller_start(h.config, h.nodes(), &error);
    check(h.r && h.r->action_history == history && history->phases == 2 && h.r->nodes == 10 &&
          history->total_spent == 0 && history->total_cap == 0 && history->lifetime_spent == 20 &&
          !h.r->proposals_disabled && h.r->row_bytes.empty(), "phase 2 binds the same history; caps restart");
    check(h.r->repair_state.pred.size() == 8 && h.r->repair_state.cooldown_until.size() == 8 &&
          h.r->repair_state.cooldown_until[0] == 2 && h.r->repair_state.windows == 1,
          "the planner state of phase 1 carries over");
    h.window(kTrigger);
    check(h.events("row_bytes_bound") == 2 &&
          h.journal().find("\"nodes\":10,\"total_bytes\":360,") != std::string::npos &&
          history->total_cap == 720 && h.r->row_bytes[9] == 36, "phase 2 binds its own row bytes and caps");
    h.fence();  // the planning job's state is installed before the fence opens
    const auto& pred = h.r->repair_state.pred;
    check(pred.size() == 10 && std::fabs(pred[3] - 0.2) < 1e-12 && std::fabs(pred[8] - 0.45) < 1e-12 &&
          std::fabs(pred[0] - 0.4) < 1e-12 && h.r->repair_state.cooldown_until.size() == 10 &&
          h.r->repair_state.cooldown_until[0] == 2, "EMA kept for known ids, new ids padded, cooldowns kept");
    check(stubs.accepted.size() == 2 && stubs.accepted[1].moves.size() == 2 &&
          stubs.accepted[1].moves[0].global_id == 8 && stubs.accepted[1].moves[0].from_partition == 0 &&
          stubs.accepted[1].moves[0].to_partition == 3 && stubs.accepted[1].moves[1].global_id == 1 &&
          stubs.accepted[1].moves[1].to_partition == 2 && g_daemon.membership[8] == 3,
          "phase 2 moves the inserted id 8 and node 1");
    check(h.r->moved_bytes == 72 && history->total_spent == 72 && history->lifetime_spent == 92 &&
          h.r->last_committed_actions.size() == 2 && h.r->last_committed_actions[0].adjacency_bytes == 36,
          "phase 2's bytes are its own rows'");
    h.window(kKeep);
    check(!window_controller_has_unresolved_action(*h.r) && h.events("verify_accepted") == 2, "phase 2 resolved");
    h.check_journal_clock();
    h.stop();
}

void isolated(const char* name, void (*scenario)()) {
    std::fflush(stdout);  // a child must not repeat the parent's buffered lines
    const pid_t child = fork();
    check(child >= 0, "fork");
    if (child == 0) { scenario(); std::fflush(stdout); _exit(0); }
    int status = 0;
    check(waitpid(child, &status, 0) == child, "wait");
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "scenario %s failed\n", name);
        std::abort();
    }
    std::printf("%s OK\n", name);
}
}  // namespace

void test_window_snapshot_transport_cases();
void test_window_native_control_cases();

int main() {
    isolated("move_then_accept", move_then_accept);
    isolated("move_reject_inverse", move_reject_inverse);
    isolated("vector_bytes_count_in_budgets", vector_bytes_count_in_budgets);
    isolated("rejected_burst_not_reproposed_after_long_windows",
             rejected_burst_not_reproposed_after_long_windows);
    isolated("prepare_failure_recovers", prepare_failure_recovers);
    isolated("prepare_failed_burst_reproposed_after_backoff", prepare_failed_burst_reproposed_after_backoff);
    isolated("idle_rank_rejects_after_invalid_limit", idle_rank_rejects_after_invalid_limit);
    isolated("verdict_accepted_despite_invalid_observation", verdict_accepted_despite_invalid_observation);
    isolated("verdict_rejected_despite_invalid_observation", verdict_rejected_despite_invalid_observation);
    isolated("begin_failures_count_toward_invalid_limit", begin_failures_count_toward_invalid_limit);
    isolated("harvest_failures_count_toward_invalid_limit", harvest_failures_count_toward_invalid_limit);
    isolated("short_commands_reject_after_invalid_limit", short_commands_reject_after_invalid_limit);
    isolated("lost_windows_count_toward_invalid_limit", lost_windows_count_toward_invalid_limit);
    isolated("persistent_dump_dir_failure_reverts", persistent_dump_dir_failure_reverts);
    isolated("revert_record_fails_at_close", revert_record_fails_at_close);
    isolated("plan_record_fails_at_close", plan_record_fails_at_close);
    isolated("prepare_failure_with_failed_rollback_freezes_dispatch",
             prepare_failure_with_failed_rollback_freezes_dispatch);
    isolated("inverse_prepare_failure_is_terminal", inverse_prepare_failure_is_terminal);
    isolated("commit_failure_is_terminal", commit_failure_is_terminal);
    isolated("inverse_commit_failure_is_terminal", inverse_commit_failure_is_terminal);
    isolated("commit_and_rollback_failure_freezes_dispatch", commit_and_rollback_failure_freezes_dispatch);
    isolated("accept_failure_freezes_dispatch", accept_failure_freezes_dispatch);
    isolated("inverse_accept_failure_freezes_dispatch", inverse_accept_failure_freezes_dispatch);
    isolated("guard_requires_scored_trigger", guard_requires_scored_trigger);
    isolated("guard_fails_closed_on_unscored_verify", guard_fails_closed_on_unscored_verify);
    isolated("guard_off_ignores_recall", guard_off_ignores_recall);
    isolated("command_change_abandons_window", command_change_abandons_window);
    isolated("begin_acks_in_next_command_abandon_window", begin_acks_in_next_command_abandon_window);
    isolated("k8_underloaded_reject_then_accept", k8_underloaded_reject_then_accept);
    isolated("wnb_move_then_accept", wnb_move_then_accept);
    isolated("weight_w_refuses_nbwork_frames", weight_w_refuses_nbwork_frames);
    isolated("second_phase_binds_more_nodes", second_phase_binds_more_nodes);
    isolated("snapshot_transport", test_window_snapshot_transport_cases);
    isolated("native_control", test_window_native_control_cases);
    std::printf("window controller lifecycle tests passed\n");
    return 0;
}
