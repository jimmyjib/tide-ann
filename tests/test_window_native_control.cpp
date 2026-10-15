// Snapshot decoding, the in-memory merge and planning run on the controller's worker thread and
// never block the pipeline owner: dispatch continues, the window's end
// boundary is frozen at the first owner tick after every rank's frame arrived,
// and a stale or cancelled job cannot publish into a later window.
#include "window_native_control.h"
#include "window_snapshot_transport.h"
#include "coordinator_stats_window.h"
#include "npy_writer.h"
#include "test_window_fixture.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <thread>
#include <unistd.h>

namespace {
using namespace tide;
void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", why); std::abort(); }
}
const std::vector<uint32_t> kMembership = {0, 1, 2, 3};
// The runtime below runs the default weight W: its frames carry the owner histogram only.
std::vector<char> frame(uint32_t rank, uint64_t epoch, uint32_t work = 3) {
    window_test::RankCounters counters{10, 20, 30, 40};
    std::vector<uint32_t> w(4, 0);
    w[rank] = work;
    return window_test::frame(rank, epoch, counters, w, kMembership, {},
                              window_repair_structure_sections(WindowRepairConfig{}));
}
void wait_receipts(WindowControllerRuntime& runtime) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (;;) {
        window_native_poll_snapshots(runtime);
        bool pending = false;
        for (auto& job : runtime.native_control->snapshots) pending |= job.valid();
        if (!pending) return;
        check(std::chrono::steady_clock::now() < deadline, "bounded snapshot worker completion");
        std::this_thread::yield();
    }
}
}  // namespace

void test_window_native_control_cases() {
    using namespace tide;
    char pattern[] = "/tmp/window-control-worker-XXXXXX";
    const auto* made = mkdtemp(pattern); check(made, "owned native-control test directory");
    const std::filesystem::path dir(made);
    WindowControllerConfig config; config.journal_dir = dir.string();
    config.repair.enabled = true; config.ranks = 4;
    std::string error;
    auto* runtime = window_controller_start(config, 4, &error); check(runtime, "native-control runtime starts");
    runtime->window_epoch = 31; runtime->state = WindowControllerState::HARVEST_WAIT;
    runtime->exchange_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    window_controller_arm_exchange(*runtime, WindowExchangeKind::HARVEST);
    std::promise<void> release, started;
    auto gate = release.get_future().share(); auto started_future = started.get_future();
    auto blocker = runtime->native_control->worker.submit([gate, &started] { started.set_value(); gate.wait(); return true; });
    check(started_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready, "deterministic worker barrier starts");
    for (unsigned rank = 0; rank < 4; ++rank)
        check(window_snapshot_offer_owned(*runtime, rank, frame(rank, 31)), "owned frame accepted without decode wait");
    check(!runtime->exchange_ack_count, "arrival does not masquerade as validated receipt");
    ControllerTickInputs endpoint;
    endpoint.completed_queries = 80; endpoint.recall_count = 80; endpoint.recall_sum = 79; endpoint.command_id = 1000;
    window_controller_tick(endpoint);
    for (unsigned refill = 0; refill < 16; ++refill) {
        ++endpoint.completed_queries;
        window_controller_tick(endpoint);
        check(!window_controller_dispatch_frozen() && controller_dispatch_allowed(),
              "delayed snapshot work permits continued refill on every tick");
    }
    const auto frozen = runtime->native_control->endpoints;
    check(frozen.completed_queries == 80 && frozen.recall_count == 80 && frozen.recall_sum == 79 && frozen.command_id == 1000,
          "the window end never follows background completion");
    release.set_value(); check(blocker.get(), "release worker gate"); wait_receipts(*runtime);
    check(runtime->exchange_ack_count == 4 && runtime->exchange_counter_sums.requests == 40 &&
          runtime->native_control->endpoints.completed_queries == 80, "validated receipt closes frozen counter tuple");
    window_snapshot_offer_owned(*runtime, 0, frame(0, 31)); wait_receipts(*runtime);
    check(runtime->exchange_ack_count == 4 && !runtime->exchange_rejected, "identical delayed duplicate is idempotent");
    window_snapshot_offer_owned(*runtime, 0, frame(0, 31, 4)); wait_receipts(*runtime);
    check(runtime->exchange_rejected && runtime->exchange_ack_count == 4, "conflicting delayed duplicate rejects exchange");

    runtime->window_epoch = 32; window_controller_arm_exchange(*runtime, WindowExchangeKind::HARVEST);
    std::promise<void> release_old;
    auto old_gate = release_old.get_future().share();
    auto old_blocker = runtime->native_control->worker.submit([old_gate] { old_gate.wait(); return true; });
    window_snapshot_offer_owned(*runtime, 0, frame(0, 32));
    runtime->window_epoch = 33; window_controller_arm_exchange(*runtime, WindowExchangeKind::HARVEST);
    release_old.set_value(); check(old_blocker.get(), "old worker generation released");
    window_snapshot_offer_owned(*runtime, 0, frame(0, 32));
    window_snapshot_offer_owned(*runtime, 1, frame(1, 33)); wait_receipts(*runtime);
    check(runtime->exchange_ack_count == 1 && runtime->snapshot_digests[0].empty() &&
          !std::filesystem::exists(dir / "window-32"), "cancelled and stale epochs cannot write or ACK the new epoch");

    runtime->action_history->expected_epoch = 2;
    runtime->state = WindowControllerState::PLAN_WAIT;
    auto& work = *runtime->native_control;
    work.ownership_epoch = 2; work.command_id = 1000; work.endpoints = frozen; work.endpoints_frozen = true;
    work.planning_started = true; work.planning_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    work.planning_input = std::make_shared<WindowPlanningInput>();
    std::promise<void> release_plan; auto plan_gate = release_plan.get_future().share();
    work.planning = work.worker.submit([plan_gate] { plan_gate.wait(); return WindowPlanningResult{}; });
    check(window_native_plan_applicable(*runtime, 2, endpoint), "matching command and ownership accept the result");
    check(!window_native_plan_applicable(*runtime, 3, endpoint), "ownership change rejects result");
    auto wrong_command = endpoint; ++wrong_command.command_id;
    check(!window_native_plan_applicable(*runtime, 2, wrong_command), "command change rejects result");
    runtime->config.proposal_query_limit = endpoint.completed_queries;
    check(!window_native_plan_applicable(*runtime, 2, endpoint), "late completion cannot cross proposal cutoff");
    runtime->config.proposal_query_limit = 0;
    runtime->verify_pending = true;
    check(!window_native_plan_applicable(*runtime, 2, endpoint), "a pending verdict blocks result application");
    runtime->verify_pending = false;
    runtime->stats_stop_pending = true;
    window_controller_tick(endpoint);
    check(!work.planning.valid() && !work.planning_started, "a pending stats STOP invalidates unpublished planning");
    release_plan.set_value();
    runtime->stats_stop_pending = false; runtime->stats_off_confirmed = true;
    runtime->state = WindowControllerState::IDLE;
    check(window_controller_stop(), "stop joins the owned worker without late runtime access");
    std::filesystem::remove_all(dir);
}
