// Pure pieces of the repair-only runtime controller: the in-memory merge of
// the window's frames (K = 8), skew, skeleton/inverse plans, the ACK record,
// the counter-domain measurement, the per-rank costs and prices (K = 4 and
// K = 8), and the guard's window recall.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "npy_writer.h"
#include "membership_migration_plan.h"
#include "protocol/message_types.h"
#include "window_controller_runtime.h"
#include "window_controller.h"
#include "test_window_fixture.h"

namespace {
using namespace tide;

void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", what); std::abort(); }
}

// K = 8: frames of eight ranks merge against the membership; owners >= 4
// keep their own histogram columns and work.
void test_merge_frames_k8() {
    constexpr uint32_t kRanks = 8, kNodes = 20;
    std::vector<uint32_t> membership(kNodes);
    for (uint32_t v = 0; v < kNodes; ++v) membership[v] = (v * 3 + 1) % kRanks;
    std::vector<uint32_t> w_all(kNodes);
    for (uint32_t v = 0; v < kNodes; ++v) w_all[v] = 10 + v;
    std::vector<std::shared_ptr<const WindowRankFrame>> frames(kRanks);
    for (uint32_t rank = 0; rank < kRanks; ++rank) {
        std::vector<uint16_t> row(kRanks, 0); row[7 - rank] = 3;
        const auto bytes = window_test::frame(rank, 9, {1, 1, 1, 2}, w_all, membership, row,
                                              kWindowStructureOwnerHist | kWindowStructureNbWork, kRanks);
        auto frame = std::make_shared<WindowRankFrame>();
        std::string error;
        check(window_snapshot_decode(bytes.data(), bytes.size(), rank, 9, kNodes, kRanks,
                                     kWindowStructureOwnerHist | kWindowStructureNbWork, *frame, error),
              "K = 8 frame decodes");
        frames[rank] = frame;
    }
    WindowTables tables; WindowStructure structure; std::string error;
    check(window_merge_frames(frames, membership, kRanks, kWindowStructureOwnerHist | kWindowStructureNbWork,
                              &tables, &structure, &error), "eight frames merge");
    bool ok = tables.nodes == kNodes && tables.per_rank_work.size() == kRanks && structure.owners == kRanks;
    std::vector<uint64_t> per_rank(kRanks, 0);
    for (uint32_t v = 0; v < kNodes; ++v) {
        ok &= tables.work[v] == 10 + v;
        per_rank[membership[v]] += 10 + v;
        for (uint32_t p = 0; p < kRanks; ++p)
            ok &= structure.owner_hist[v * kRanks + p] == (p == 7 - membership[v] ? 3u : 0u);
    }
    ok &= per_rank == tables.per_rank_work;
    check(ok, "work and histogram rows of owners 4..7 land on their global ids");
    auto moved = membership; std::swap(moved[0], moved[1]);
    check(!window_merge_frames(frames, moved, kRanks, kWindowStructureOwnerHist | kWindowStructureNbWork,
                               &tables, &structure, &error) && error.find("owned ids differ") != std::string::npos,
          "a membership the frames were not captured under is refused");
    check(!window_merge_frames(frames, membership, 4, kWindowStructureOwnerHist | kWindowStructureNbWork,
                               &tables, &structure, &error), "a frame count other than K is refused");
    std::printf("test_merge_frames_k8 OK\n");
}

void test_plans() {
    check(window_controller_skew({100, 100, 100, 100}) == 1.0, "flat skew");
    const double skew = window_controller_skew({120, 100, 100, 80});
    check(skew > 1.19 && skew < 1.21, "skew 1.2");
    check(window_controller_skew({}) == 0.0 && window_controller_skew({0, 0}) == 0.0, "empty skew");
    std::vector<WindowBurstAction> actions = {{7, 0, 1, 40}, {9, 2, 3, 24}};
    std::vector<char> bytes;
    std::string error;
    check(window_controller_skeleton_plan(actions, 5, 4, &bytes, &error), "skeleton serializes");
    MembershipMigrationPlan plan;
    check(load_membership_migration_plan_bytes(bytes.data(), bytes.size(), 100, 4, plan, error),
          "skeleton round-trips through the loader");
    check(plan.mode == kMigrationPlanModeMove && plan.epoch_from == 5 && plan.epoch_to == 6, "MOVE plan epochs");
    check(plan.moves.size() == 2 && plan.moves[0].global_id == 7 && plan.moves[0].from_partition == 0 &&
          plan.moves[0].to_partition == 1 && plan.moves[0].adjacency.empty(), "moves with empty rows");
    check(!window_controller_skeleton_plan({}, 5, 4, &bytes, &error), "an empty burst has no plan");
    const auto inverse = window_controller_inverse_actions(actions);
    check(inverse.size() == 2 && inverse[0].node == 7 && inverse[0].source_owner == 1 && inverse[0].target_owner == 0 &&
          inverse[0].adjacency_bytes == 40 && inverse[1].source_owner == 3 && inverse[1].target_owner == 2,
          "inverse swaps owners and keeps the row bytes");
    std::printf("test_plans OK\n");
}

void test_fence_and_hooks() {
    ControllerTickInputs inputs;
    check(window_controller_fence_drained(inputs), "zeroed state drains");
    inputs = {}; inputs.logical_in_flight = 1;
    check(!window_controller_fence_drained(inputs), "logical in-flight holds");
    inputs = {}; inputs.transport_in_flight = 1;
    check(!window_controller_fence_drained(inputs), "transport in-flight holds");
    inputs = {}; inputs.queued_search = 1;
    check(!window_controller_fence_drained(inputs), "queued search holds");
    inputs = {}; inputs.queued_responses = 1;
    check(!window_controller_fence_drained(inputs), "queued responses holds");
    check(!window_controller_dispatch_frozen() && controller_dispatch_allowed() && !controller_running(),
          "no controller never freezes the pipeline");
    const auto begin = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, 0, 1, {});
    check(controller_offer(0, &begin, sizeof(begin)), "a stale stats ACK after stop is still consumed");
    ann::StatsWindowAck search{}; search.msg_type = ann::MessageType::SEARCH_RESPONSE;
    check(!controller_offer(0, &search, sizeof(search)), "a search reply is never controller traffic");
    std::printf("test_fence_and_hooks OK\n");
}

WindowControllerRuntime make_exchange_runtime() {
    WindowControllerRuntime runtime;
    runtime.config.ranks = 4;
    runtime.window_epoch = 7;
    return runtime;
}

void test_offer_ack_collection() {
    WindowControllerRuntime runtime = make_exchange_runtime();
    window_controller_arm_exchange(runtime, WindowExchangeKind::BEGIN);
    check(!window_controller_exchange_complete(runtime) && !runtime.stats_off_confirmed, "BEGIN opens the stats windows");
    std::vector<char> junk(sizeof(ann::StatsWindowAck), 0);
    check(!window_controller_offer_ack(runtime, 0, junk.data(), junk.size()), "non-window message refused");
    const auto stale = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, 0, 6, {});
    check(window_controller_offer_ack(runtime, 0, &stale, sizeof(stale)) && runtime.exchange_ack_count == 0,
          "stale epoch consumed and ignored");
    const auto wrong_kind = window_test::ack(ann::MessageType::STATS_WINDOW_HARVEST_ACK, 0, 7, {});
    check(window_controller_offer_ack(runtime, 0, &wrong_kind, sizeof(wrong_kind)) && runtime.exchange_ack_count == 0,
          "wrong-kind ACK consumed and ignored");
    for (uint32_t rank = 0; rank < 4; ++rank) {
        const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, 7,
                                        {10 * (rank + 1), 100 * (rank + 1), 200, 300});
        check(window_controller_offer_ack(runtime, rank, &a, sizeof(a)) &&
              window_controller_offer_ack(runtime, rank, &a, sizeof(a)), "matching and duplicate ACK consumed");
    }
    check(window_controller_exchange_complete(runtime) && runtime.exchange_ack_count == 4, "every rank once");
    check(runtime.exchange_counter_sums.requests == 100 && runtime.exchange_counter_sums.distances == 1000 &&
          runtime.exchange_rank_counters[2].requests == 30, "counters summed and kept per rank");
    window_controller_arm_exchange(runtime, WindowExchangeKind::HARVEST);
    const auto bare = window_test::ack(ann::MessageType::STATS_WINDOW_HARVEST_ACK, 1, 7, {});
    check(window_controller_offer_ack(runtime, 1, &bare, sizeof(bare)) && runtime.exchange_rejected,
          "a bare HARVEST ACK without its snapshot rejects the exchange");
    window_controller_arm_exchange(runtime, WindowExchangeKind::BEGIN);
    auto rejected = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, 1, 7, {});
    rejected.status = 1;
    check(window_controller_offer_ack(runtime, 1, &rejected, sizeof(rejected)) && runtime.exchange_rejected &&
          !window_controller_exchange_complete(runtime), "a rejected ACK poisons the exchange");
    std::printf("test_offer_ack_collection OK\n");
}

// A runtime whose last BEGIN/HARVEST exchanges measured `queries` completed
// queries with per-rank deltas D and R (Td = D, Ttot = D + R).
WindowControllerRuntime measured_runtime(const std::array<uint64_t, 4>& D, const std::array<uint64_t, 4>& R,
                                         uint64_t queries) {
    WindowControllerRuntime runtime = make_exchange_runtime();
    window_controller_arm_exchange(runtime, WindowExchangeKind::HARVEST);
    runtime.window_begin_rank_counters.assign(4, {});
    runtime.window_begin_completed = 1000; runtime.window_begin_command = 3;
    for (uint32_t r = 0; r < 4; ++r) {
        runtime.exchange_acked[r] = 1;
        runtime.exchange_rank_counters[r] = {R[r], D[r], D[r], D[r] + R[r]};
        runtime.exchange_counter_sums.requests += R[r]; runtime.exchange_counter_sums.distances += D[r];
        runtime.exchange_counter_sums.distance_us += D[r]; runtime.exchange_counter_sums.total_us += D[r] + R[r];
    }
    runtime.exchange_ack_count = 4;
    runtime.harvest_total_w = 1;
    (void)queries;
    return runtime;
}

ControllerTickInputs end_inputs(uint64_t queries, uint64_t scored, double recall_sum) {
    ControllerTickInputs inputs;
    inputs.completed_queries = 1000 + queries; inputs.command_id = 3;
    inputs.recall_count = scored; inputs.recall_sum = recall_sum;
    return inputs;
}

void test_measurement_and_costs() {
    auto runtime = measured_runtime({1000, 100, 100, 100}, {100, 50, 50, 50}, 100);
    check(window_controller_measurement_valid(runtime, end_inputs(100, 0, 0.0)),
          "an unscored window is a valid measurement: recall never enters the counter domain");
    auto other_command = end_inputs(100, 0, 0.0); other_command.command_id = 4;
    check(!window_controller_measurement_valid(runtime, other_command), "a window never spans two commands");
    check(!window_controller_measurement_valid(runtime, end_inputs(0, 0, 0.0)), "no completed query, no window");
    runtime.config.min_window_queries = 101;
    check(!window_controller_measurement_valid(runtime, end_inputs(100, 0, 0.0)), "below the query minimum");
    runtime.config.min_window_queries = 0;
    runtime.harvest_total_w = 0;
    check(!window_controller_measurement_valid(runtime, end_inputs(100, 0, 0.0)), "no window work");
    runtime.harvest_total_w = 1;
    runtime.exchange_rank_counters[3].distances = 0;  // rank 3 computed nothing
    check(!window_controller_measurement_valid(runtime, end_inputs(100, 0, 0.0)), "every rank needs D and R");
    runtime = measured_runtime({1000, 100, 100, 100}, {100, 50, 50, 50}, 100);

    check(std::isnan(window_controller_window_recall(runtime, end_inputs(100, 0, 0.0))), "unscored window: NaN");
    check(std::isnan(window_controller_window_recall(runtime, end_inputs(100, 99, 90.0))), "partly scored window: NaN");
    check(window_controller_window_recall(runtime, end_inputs(100, 100, 90.0)) == 90.0, "fully scored window recall");
    auto changed = end_inputs(100, 100, 90.0); changed.command_id = 9;
    check(std::isnan(window_controller_window_recall(runtime, changed)), "recall of another command: NaN");

    WindowRankCostSnapshot trigger;
    check(window_controller_rank_snapshot(runtime, 100, &trigger) && window_rank_time_valid(trigger.rank_time),
          "rank snapshot with per-rank prices");
    check(trigger.distances_per_query == std::vector<double>({10, 1, 1, 1}) &&
          trigger.requests_per_query == std::vector<double>({1, 0.5, 0.5, 0.5}), "per-query rates");
    std::vector<double> costs;
    check(window_controller_rank_costs(trigger, trigger, &costs) &&
          costs == std::vector<double>({11, 1.5, 1.5, 1.5}), "trigger costs priced with its own prices");
    auto verify = measured_runtime({800, 300, 100, 100}, {100, 60, 50, 50}, 100);
    for (auto& c : verify.exchange_rank_counters) {  // the verify window's own prices differ
        c.total_us += c.distance_us; c.distance_us *= 2;
    }
    WindowRankCostSnapshot measured;
    check(window_controller_rank_snapshot(verify, 100, &measured), "verify snapshot");
    check(window_controller_rank_costs(trigger, measured, &costs) &&
          costs == std::vector<double>({9, 3.6, 1.5, 1.5}), "verify costs priced with the frozen trigger prices");
    WindowRankCostSnapshot invalid;
    check(!window_controller_rank_costs(invalid, measured, &costs), "no frozen prices, no costs");

    WindowControllerRuntime priced;
    priced.config.ranks = 4;
    window_controller_repair_freeze_prices(priced, trigger);
    check(priced.repair_state.price_frozen && priced.repair_state.price == std::vector<double>(4, 1.0) &&
          priced.repair_state.price_r == std::vector<double>(4, 1.0), "pooled repair prices");
    check(window_controller_repair_peak_us(priced, measured) == 9.0, "repair peak with the pooled prices");
    window_controller_repair_freeze_prices(priced, measured);
    check(priced.repair_state.price == std::vector<double>(4, 1.0), "prices freeze once");
    std::printf("test_measurement_and_costs OK\n");
}

// K = 8: the per-rank rows, costs, pooled prices and peak have eight entries;
// a K = 4 price context cannot price a K = 8 window.
void test_measurement_and_costs_k8() {
    WindowControllerRuntime runtime;
    runtime.config.ranks = 8; runtime.window_epoch = 7;
    window_controller_arm_exchange(runtime, WindowExchangeKind::HARVEST);
    runtime.window_begin_rank_counters.assign(8, {});
    runtime.window_begin_completed = 1000; runtime.window_begin_command = 3;
    for (uint32_t r = 0; r < 8; ++r) {
        const uint64_t D = 100 * (r + 1), R = 10 * (r + 1);
        runtime.exchange_acked[r] = 1;
        runtime.exchange_rank_counters[r] = {R, D, D, D + R};
    }
    runtime.exchange_ack_count = 8;
    WindowRankCostSnapshot snapshot;
    check(window_controller_rank_snapshot(runtime, 100, &snapshot) && snapshot.rank_time.ranks.size() == 8 &&
          snapshot.distances_per_query.size() == 8 && snapshot.distances_per_query[7] == 8.0,
          "eight per-rank rows");
    std::vector<double> costs;
    check(window_controller_rank_costs(snapshot, snapshot, &costs) && costs.size() == 8 && costs[7] == 8.8,
          "eight per-rank costs");
    auto four = measured_runtime({1000, 100, 100, 100}, {100, 50, 50, 50}, 100);
    WindowRankCostSnapshot four_prices;
    check(window_controller_rank_snapshot(four, 100, &four_prices), "K = 4 snapshot");
    check(!window_controller_rank_costs(four_prices, snapshot, &costs), "four prices cannot price eight ranks");
    window_controller_repair_freeze_prices(runtime, snapshot);
    check(runtime.repair_state.price == std::vector<double>(8, 1.0) &&
          runtime.repair_state.price_r == std::vector<double>(8, 1.0), "pooled prices for eight ranks");
    check(window_controller_repair_peak_us(runtime, snapshot) == 8.8, "peak over eight ranks");
    check(window_controller_exchange_deadline_sec(4) == 25.0 && window_controller_exchange_deadline_sec(2) == 25.0 &&
          window_controller_exchange_deadline_sec(8) == 50.0 && window_controller_exchange_deadline_sec(16) == 100.0,
          "the exchange deadline grows with K");
    std::printf("test_measurement_and_costs_k8 OK\n");
}

// window_rank_time_seal / window_rank_time_valid (include/window_rank_time_cost.h): an
// undefined price never becomes a number: a zero R, D or Q, a negative residual
// or a non-finite price fails closed.
void test_rank_time_seal_fails_closed() {
    WindowRankTimeContext base;
    base.Q = 10;
    base.ranks = {WindowRankTimeRow{100, 10, 100, 110}, WindowRankTimeRow{50, 10, 100, 120},
                  WindowRankTimeRow{40, 10, 40, 50}, WindowRankTimeRow{30, 10, 30, 40}};
    check(!window_rank_time_valid(base), "an unsealed context is not valid");
    check(window_rank_time_seal(&base) && window_rank_time_valid(base) && base.reason.empty(), "a complete context seals");
    check(base.ranks[0].c_d_us == 1.0 && base.ranks[0].c_r_us == 1.0 && base.ranks[1].c_d_us == 2.0 &&
          base.ranks[1].c_r_us == 2.0, "prices are Td/D and (Ttot - Td)/R per rank");
    auto invalid = base; invalid.ranks[0].R = 0;
    check(!window_rank_time_seal(&invalid) && invalid.reason == "rank-0-zero-R", "zero R (including 0/0) is undefined");
    invalid = base; invalid.ranks[1].D = 0;
    check(!window_rank_time_seal(&invalid) && invalid.reason == "rank-1-zero-D", "zero D is undefined");
    invalid = base; invalid.ranks[2].Ttot_us = invalid.ranks[2].Td_us - 1;
    check(!window_rank_time_seal(&invalid) && invalid.reason == "rank-2-negative-residual",
          "a negative per-rank residual fails closed");
    invalid = base; invalid.Q = 0;
    check(!window_rank_time_seal(&invalid) && invalid.reason == "zero-query-denominator", "zero Q fails closed");
    invalid = base; invalid.ranks[3].R = 0; invalid.ranks[0].D = 0;
    check(!window_rank_time_seal(&invalid) && invalid.reason == "rank-0-zero-D", "the first undefined rank is named");
    invalid = base; invalid.ranks[0] = WindowRankTimeRow{100, 10, 0, 0};
    check(window_rank_time_seal(&invalid) && invalid.ranks[0].c_d_us == 0 && invalid.ranks[0].c_r_us == 0,
          "defined zero time prices stay zero, never replaced with global prices");
    invalid.ranks[0].c_r_us = std::numeric_limits<double>::infinity();
    check(!window_rank_time_valid(invalid), "a nonfinite price cannot pass on the validity bit");
    invalid = base; invalid.ranks[3].c_d_us = 99.0;
    check(!window_rank_time_valid(invalid), "a price that is not the rank's own Td/D is invalid");
    invalid = base; invalid.ranks[2].valid = false;
    check(!window_rank_time_valid(invalid), "an invalid rank invalidates the context");
    check(!window_rank_time_seal(nullptr), "no context");
    std::printf("test_rank_time_seal_fails_closed OK\n");
}

// window_controller_rank_snapshot over the BEGIN/HARVEST ACK counters fails closed on
// partial coverage, a duplicate rank and a counter regression.
void test_rank_snapshot_fails_closed() {
    // BEGIN ACKs fill the same per-rank exchange counters a HARVEST does (a HARVEST
    // ACK additionally needs its snapshot frame); the window baseline is set directly.
    WindowControllerRuntime runtime = make_exchange_runtime();
    window_controller_arm_exchange(runtime, WindowExchangeKind::BEGIN);
    runtime.window_begin_rank_counters.assign(4, {10, 100, 100, 200});
    WindowRankCostSnapshot snapshot;
    for (uint32_t rank = 0; rank < 3; ++rank) {
        const auto a = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, rank, 7, {20, 300, 500, 900});
        check(window_controller_offer_ack(runtime, rank, &a, sizeof(a)), "rank ACK consumed");
    }
    check(!window_controller_rank_snapshot(runtime, 10, &snapshot), "partial rank coverage cannot score");
    const auto last = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, 3, 7, {20, 300, 500, 900});
    window_controller_offer_ack(runtime, 3, &last, sizeof(last));
    check(window_controller_rank_snapshot(runtime, 10, &snapshot) &&
          snapshot.distances_per_query == std::vector<double>(4, 20.0) &&
          snapshot.requests_per_query == std::vector<double>(4, 1.0) &&
          snapshot.rank_time.ranks[2].c_d_us == 2.0 && snapshot.rank_time.ranks[2].c_r_us == 30.0,
          "D and R share the exact ACK interval");
    const auto duplicate = window_test::ack(ann::MessageType::STATS_WINDOW_BEGIN_ACK, 3, 7, {999, 9999, 500, 900});
    check(window_controller_offer_ack(runtime, 3, &duplicate, sizeof(duplicate)) &&
          runtime.exchange_rank_counters[3].requests == 20 && runtime.exchange_rank_counters[3].distances == 300 &&
          runtime.exchange_ack_count == 4, "a duplicate rank cannot overwrite its counter vector");
    // (WindowControllerRuntime owns its journal stream and is not copyable: edit, check, restore.)
    const auto counters = runtime.exchange_rank_counters;
    const auto acked = runtime.exchange_acked;
    const auto begin = runtime.window_begin_rank_counters;
    auto restore = [&] {
        runtime.exchange_rank_counters = counters;
        runtime.exchange_acked = acked;
        runtime.window_begin_rank_counters = begin;
    };
    runtime.exchange_rank_counters[2].distances = 99;
    check(!window_controller_rank_snapshot(runtime, 10, &snapshot) &&
          snapshot.rank_time.reason == "rank-2-counter-regression", "a distance counter regression fails closed");
    restore();
    runtime.exchange_rank_counters[1].total_us = 150;
    check(!window_controller_rank_snapshot(runtime, 10, &snapshot) &&
          snapshot.rank_time.reason == "rank-1-counter-regression", "a time counter regression fails closed");
    restore();
    runtime.exchange_rank_counters[0].requests = 10;  // R = 0 over the window
    check(!window_controller_rank_snapshot(runtime, 10, &snapshot) &&
          snapshot.rank_time.reason == "rank-0-zero-R", "a rank with no request in the window has no price");
    restore();
    check(!window_controller_rank_snapshot(runtime, 0, &snapshot), "no completed query, no snapshot");
    runtime.exchange_acked[1] = 0;
    check(!window_controller_rank_snapshot(runtime, 10, &snapshot), "an unacknowledged rank cannot score");
    restore();
    runtime.window_begin_rank_counters.pop_back();
    check(!window_controller_rank_snapshot(runtime, 10, &snapshot), "a missing BEGIN baseline cannot score");
    restore();
    check(!window_controller_rank_snapshot(runtime, 10, nullptr), "no output");
    check(window_controller_rank_snapshot(runtime, 10, &snapshot), "the unmodified exchange still scores");
    std::printf("test_rank_snapshot_fails_closed OK\n");
}
}  // namespace

void test_window_action_history_cases();

int main() {
    test_merge_frames_k8();
    test_plans();
    test_fence_and_hooks();
    test_offer_ack_collection();
    test_measurement_and_costs();
    test_measurement_and_costs_k8();
    test_rank_time_seal_fails_closed();
    test_rank_snapshot_fails_closed();
    test_window_action_history_cases();
    std::printf("window controller tests passed\n");
    return 0;
}
