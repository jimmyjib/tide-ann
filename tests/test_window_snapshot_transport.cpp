// The snapshot receipt paths (controller_offer of a raw frame, and the owned
// path through the control worker): validation against the expected rank,
// epoch, owner count, node count and sections; the in-memory merge of every
// rank's frame (owned counts and owned-id digests checked against the
// membership); a 16-frame harvest through the worker queue; and the all-rank
// STOP receipt collector at K = 4 and K = 8.
#include "window_snapshot_transport.h"
#include "window_native_control.h"
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
// Weight WNB's sections (owner histogram + nbwork): the controllers below pin weight:wnb,
// except weight_w_frames (the default weight W: the owner histogram only).
constexpr uint32_t kWnbSections = kWindowStructureOwnerHist | kWindowStructureNbWork;
window_test::RankCounters counters(uint32_t rank) { return {31 + rank, 41 + rank, 51 + rank, 61 + rank}; }
// Rank r executed 3 distances for its own node r; its owned row has a 1 at column (r + 1) % 4.
std::vector<char> snapshot(uint32_t rank, uint64_t epoch, uint32_t work = 3,
                           uint32_t sections = kWindowStructureOwnerHist | kWindowStructureNbWork) {
    std::vector<uint32_t> w(4, 0);
    std::vector<uint16_t> row(4, 0);
    w[rank] = work; row[(rank + 1) % 4] = 1;
    return window_test::frame(rank, epoch, counters(rank), w, kMembership, row, sections);
}
std::filesystem::path temp_dir(const char* stem) {
    std::string pattern = std::string("/tmp/") + stem + "-XXXXXX";
    const char* made = mkdtemp(pattern.data()); check(made, "owned fixture directory");
    return made;
}
WindowControllerRuntime* start(const std::filesystem::path& dir, uint32_t ranks, uint32_t nodes,
                               WindowRepairWeight weight = WindowRepairWeight::WNB) {
    WindowControllerConfig config; config.journal_dir = dir.string();
    config.repair.enabled = true; config.ranks = ranks; config.repair.weight = weight;
    std::string error;
    auto* r = window_controller_start(config, nodes, &error); check(r, "production runtime starts");
    return r;
}

// K = 16: every rank's frame queues behind a busy worker (16 frames plus the
// blocker, more than 8 queued at once), then all are validated and merged.
void sixteen_frame_harvest() {
    const auto dir = temp_dir("window-k16");
    constexpr uint32_t kRanks = 16, kNodes = 48;
    // Another controller context in this process: drop the retained history
    // first, as the coordinator's reset does.
    check(window_controller_reset_history(), "history reset with no controller active");
    auto* r = start(dir, kRanks, kNodes);
    std::vector<uint32_t> membership(kNodes);
    for (uint32_t v = 0; v < kNodes; ++v) membership[v] = (v * 7) % kRanks;  // three ids per rank
    std::vector<uint32_t> w_all(kNodes);
    for (uint32_t v = 0; v < kNodes; ++v) w_all[v] = v + 1;
    r->window_epoch = 5; r->state = WindowControllerState::HARVEST_WAIT;
    window_controller_arm_exchange(*r, WindowExchangeKind::HARVEST);
    check(window_control_queue_capacity(kRanks) == 34, "queue capacity is 2K + 2");
    std::promise<void> release; auto gate = release.get_future().share();
    auto blocker = r->native_control->worker.submit([gate] { gate.wait(); return true; });
    check(blocker.valid(), "blocker queued");
    for (uint32_t rank = 0; rank < kRanks; ++rank) {
        std::vector<uint16_t> row(kRanks, 0); row[(rank + 3) % kRanks] = 2; row[rank] = 1;
        auto bytes = window_test::frame(rank, 5, counters(rank), w_all, membership, row,
                                        kWnbSections, kRanks);
        check(!bytes.empty() && window_snapshot_offer_owned(*r, rank, std::move(bytes)) && !r->exchange_rejected,
              "every one of the 16 frames is queued (the queue holds 2K + 2 jobs)");
    }
    release.set_value(); check(blocker.get(), "worker released");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        window_native_poll_snapshots(*r);
        bool pending = false;
        for (auto& job : r->native_control->snapshots) pending |= job.valid();
        if (!pending) break;
        check(std::chrono::steady_clock::now() < deadline, "bounded decode");
        std::this_thread::yield();
    }
    check(r->exchange_ack_count == kRanks && !r->exchange_rejected && window_controller_exchange_complete(*r),
          "a 16-frame harvest completes");
    while (!window_native_load(*r, membership)) {
        check(std::chrono::steady_clock::now() < deadline, "bounded merge");
        std::this_thread::yield();
    }
    const auto& obs = r->native_control->observation;
    check(obs.valid && obs.structure.valid && obs.structure.owners == kRanks && obs.tables.nodes == kNodes,
          "the 16 frames merge in memory");
    bool scattered = true;
    for (uint32_t v = 0; v < kNodes; ++v) {
        const uint32_t own = membership[v];
        scattered &= obs.tables.work[v] == v + 1;
        for (uint32_t p = 0; p < kRanks; ++p)
            scattered &= obs.structure.owner_hist[v * kRanks + p] == (p == (own + 3) % kRanks ? 2u : p == own ? 1u : 0u);
    }
    check(scattered && obs.structure.cut_edges == 2 * kNodes, "rows scatter to their global ids with 16 columns");
    r->exchange = WindowExchangeKind::NONE; r->state = WindowControllerState::IDLE;
    check(window_controller_stop(), "K = 16 fixture teardown");
    std::filesystem::remove_all(dir);
}

// Weight W (the default): a frame carries the work and the owner histogram only
// (header + 4 x owned + 2 x K x owned bytes, no nbwork), the merge leaves nb_work
// empty, and a frame that carries nbwork never satisfies a rank.
void weight_w_frames() {
    const auto dir = temp_dir("window-weight-w");
    check(window_controller_reset_history(), "history reset with no controller active");
    auto* r = start(dir, 4, 4, WindowRepairWeight::W);
    check(window_repair_structure_sections(r->config.repair) == kWindowStructureOwnerHist, "W: owner histogram only");
    r->window_epoch = 1;
    window_controller_arm_exchange(*r, WindowExchangeKind::HARVEST);
    for (uint32_t rank = 0; rank < 4; ++rank) {
        const auto bytes = snapshot(rank, 1, 3, kWindowStructureOwnerHist);
        const auto wnb = snapshot(rank, 1, 3, kWnbSections);
        check(bytes.size() == sizeof(WindowSnapshotHeader) + 4 + 4 * 2 &&
              wnb.size() == bytes.size() + 4 * 8, "a W frame is the WNB frame without its 8 x N nbwork bytes");
        check(controller_offer(rank, bytes.data(), bytes.size()) && r->exchange_ack_count == rank + 1 &&
              !r->exchange_rejected, "an owner-histogram frame satisfies its rank under weight W");
    }
    WindowTables tables; WindowStructure structure; std::string error;
    check(window_merge_frames(r->snapshot_frames, kMembership, 4, kWindowStructureOwnerHist, &tables, &structure,
                              &error) && structure.valid && structure.nb_work.empty() && structure.nb_work_total == 0 &&
          structure.owner_hist.size() == 16 && structure.cut_edges == 4, "the W merge has no nbwork");
    r->window_epoch = 2; window_controller_arm_exchange(*r, WindowExchangeKind::HARVEST);
    const auto wnb = snapshot(0, 2, 3, kWnbSections);
    check(controller_offer(0, wnb.data(), wnb.size()) && r->exchange_rejected && !r->exchange_ack_count,
          "a frame with the nbwork section is refused under weight W");
    r->exchange = WindowExchangeKind::NONE; r->state = WindowControllerState::IDLE;
    check(window_controller_stop(), "weight W fixture teardown");
    std::filesystem::remove_all(dir);
}

void stop_collector(uint32_t ranks) {
    for (uint64_t epoch : {uint64_t{0}, uint64_t{10}}) {
        WindowStopCollector collector(epoch, ranks);
        WindowStopAck ack; ack.status = 0; ack.requested_epoch = epoch;
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            ack.rank = rank; ack.actual_epoch = rank ? epoch : 0;
            check(collector.offer(rank, &ack, sizeof(ack)) && collector.offer(rank, &ack, sizeof(ack)), "identical STOP ACK idempotent");
            check(collector.complete() == (rank == ranks - 1), "missing rank never clean");
        }
        std::string expected = "\"stats_off_ranks\":[";
        for (uint32_t rank = 0; rank < ranks; ++rank) expected += (rank ? "," : "") + std::to_string(rank);
        check(collector.json().find(expected + "]") != std::string::npos, "STOP JSON sorted receipts");
        ack.rank = ranks; ack.actual_epoch = epoch;
        check(!collector.offer(ranks, &ack, sizeof(ack)) && !collector.complete(), "a rank beyond K invalidates the STOP");
    }
    WindowStopCollector conflict(3, ranks);
    WindowStopAck ack; ack.status = 0; ack.requested_epoch = 3; ack.actual_epoch = 3;
    for (uint32_t rank = 0; rank < ranks; ++rank) { ack.rank = rank; conflict.offer(rank, &ack, sizeof(ack)); }
    ack.rank = ranks - 1; ack.active = 1;
    check(!conflict.offer(ranks - 1, &ack, sizeof(ack)) && !conflict.complete(), "conflicting active ACK invalidates the STOP");
}
}  // namespace

void test_window_snapshot_transport_cases() {
    using namespace tide;
    const auto dir = temp_dir("window-native");
    auto* r = start(dir, 4, 4);
    r->window_epoch = 1;
    window_controller_arm_exchange(*r, WindowExchangeKind::HARVEST);
    auto stale = snapshot(0, 99);
    check(controller_offer(0, stale.data(), stale.size()) && !r->exchange_ack_count && !r->exchange_rejected,
          "wrong epoch cannot satisfy a rank");
    ann::StatsWindowAck ordinary{}; ordinary.msg_type = ann::MessageType::SEARCH_RESPONSE;
    check(!controller_offer(0, &ordinary, sizeof(ordinary)), "ordinary search stays on production path");
    for (uint32_t rank = 0; rank < 4; ++rank) {
        const auto bytes = snapshot(rank, 1);
        check(bytes.size() == sizeof(WindowSnapshotHeader) + 4 + 4 * 2 + 4 * 8, "exact v3 frame size");
        check(window_snapshot_queue_bytes(bytes.data(), bytes.size(), 4, 4) == bytes.size(), "one bounded body retained");
        check(controller_offer(rank, bytes.data(), bytes.size()) && controller_offer(rank, bytes.data(), bytes.size()) &&
              r->exchange_ack_count == rank + 1 && !r->exchange_rejected && r->snapshot_frames[rank],
              "real dispatch entry validates and counts each rank exactly once");
    }
    check(!std::filesystem::exists(dir / "window-1"), "nothing is persisted: the merge is in memory");
    WindowTables tables; WindowStructure structure; std::string error;
    check(window_merge_frames(r->snapshot_frames, kMembership, 4, kWnbSections, &tables, &structure, &error) &&
          tables.work == std::vector<uint64_t>({3, 3, 3, 3}) && tables.per_rank_work == std::vector<uint64_t>({3, 3, 3, 3}) &&
          r->exchange_counter_sums.requests == 130 && r->exchange_counter_sums.distances == 170,
          "merge scatters rank work; ACK counters summed");
    check(structure.valid && structure.owner_hist == std::vector<uint32_t>({0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}) &&
          structure.cut_edges == 4 && structure.nb_work == std::vector<uint64_t>({0, 0, 0, 0}),
          "structure scatters owned rows to global ids and sums nbwork");
    check(!window_merge_frames(r->snapshot_frames, {0, 0, 2, 3}, 4, kWnbSections, &tables, &structure, &error) &&
          !structure.valid && error.find("owned ids differ from the membership at rank 0") != std::string::npos,
          "an owned count disagreement leaves the structure absent");
    check(!window_merge_frames(r->snapshot_frames, {1, 0, 2, 3}, 4, kWnbSections, &tables, &structure, &error) &&
          error.find("owned ids differ") != std::string::npos,
          "equal owned counts but other ids: the owned-id digest refuses the scatter");
    auto missing = r->snapshot_frames; missing[2].reset();
    check(!window_merge_frames(missing, kMembership, 4, kWnbSections, &tables, &structure, &error) &&
          error.find("missing at rank 2") != std::string::npos, "a missing rank refuses the merge");
    check(!window_merge_frames(r->snapshot_frames, kMembership, 4, kWindowStructureOwnerHist, &tables, &structure, &error),
          "other sections than the frames carry refuse the merge");
    auto conflict = snapshot(0, 1, 4);
    controller_offer(0, conflict.data(), conflict.size());
    check(r->exchange_rejected && r->exchange_ack_count == 4, "valid conflicting duplicate invalidates exchange");
    for (unsigned failure = 0; failure < 9; ++failure) {
        r->window_epoch = failure + 2; window_controller_arm_exchange(*r, WindowExchangeKind::HARVEST);
        auto bytes = snapshot(0, r->window_epoch); uint32_t server = 0;
        if (failure == 0) bytes.back() ^= 1;                       // digest
        if (failure == 1) bytes.resize(7);                         // truncated
        if (failure == 2) bytes.push_back(0);                      // oversize
        if (failure == 3) server = 1;                              // another rank's frame
        if (failure == 4) bytes = snapshot(0, r->window_epoch, 3, kWindowStructureOwnerHist);  // sections
        if (failure == 5) bytes = window_test::frame(0, r->window_epoch, counters(0), {3, 0, 0, 0}, kMembership,
                                                     {}, kWnbSections, 8);  // another owner count
        if (failure == 6) bytes = window_test::frame(0, r->window_epoch, counters(0), {3, 0, 0, 0, 0}, {0, 1, 2, 3, 0},
                                                     {}, kWnbSections, 4);  // another node count
        if (failure == 7) {  // work on an id this rank does not own (re-signed, so only that gate fails)
            auto* h = reinterpret_cast<WindowSnapshotHeader*>(bytes.data());
            h->non_owned_w = 1; h->ack.total_w += 1;
            std::memset(h->sha256, 0, 64);
            const auto digest = sha256_hex(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
            std::memcpy(h->sha256, digest.data(), 64);
        }
        if (failure == 8) {  // the work total disagrees with the ACK (re-signed)
            auto* h = reinterpret_cast<WindowSnapshotHeader*>(bytes.data());
            h->ack.total_w += 1;
            std::memset(h->sha256, 0, 64);
            const auto digest = sha256_hex(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
            std::memcpy(h->sha256, digest.data(), 64);
        }
        check(controller_offer(server, bytes.data(), bytes.size()) && r->exchange_rejected && !r->exchange_ack_count,
              "digest/truncated/oversize/rank/sections/owners/nodes/non-owned/total errors never satisfy a rank");
    }
    r->window_epoch = 20; window_controller_arm_exchange(*r, WindowExchangeKind::HARVEST);
    auto next = snapshot(0, 20);
    controller_offer(0, next.data(), next.size());
    check(r->exchange_ack_count == 1 && !r->exchange_rejected, "fresh next epoch independent of rejected predecessor");
    ordinary.msg_type = ann::MessageType::STATS_WINDOW_HARVEST_ACK; ordinary.window_epoch = 20; ordinary.server_rank = 1;
    controller_offer(1, &ordinary, sizeof(ordinary));
    check(r->exchange_rejected && r->exchange_ack_count == 1, "a bare HARVEST ACK cannot replace the snapshot");
    std::vector<char> oversized(sizeof(WindowSnapshotHeader) + 4 * 4 + 4 * 4 * 2 + 4 * 8 + 1, 0);
    oversized[0] = static_cast<char>(ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE);
    check(window_snapshot_queue_bytes(oversized.data(), oversized.size(), 4, 4) == sizeof(WindowSnapshotHeader),
          "oversized single frame never creates an unbounded second queue copy");
    oversized.pop_back();
    check(window_snapshot_queue_bytes(oversized.data(), oversized.size(), 4, 4) == oversized.size() &&
          window_snapshot_queue_bytes(oversized.data(), oversized.size(), 4, 1) == sizeof(WindowSnapshotHeader),
          "the bound is every node owned in every section, for a valid owner count only");
    r->exchange = WindowExchangeKind::NONE;
    r->stats_off_confirmed = false;  // as after any BEGIN
    check(!window_controller_stop() && window_controller_active() == r, "stop requires the all-rank stats STOP");
    r->stats_stop_pending = true; r->state = WindowControllerState::IDLE;
    const auto old_epoch = r->window_epoch;
    window_controller_tick({});
    check(r->window_epoch == old_epoch && r->state == WindowControllerState::IDLE,
          "a pending STOP blocks new windows until it is retried");
    r->stats_stop_pending = false;
    r->stats_off_confirmed = true; r->verify_pending = true;
    check(!window_controller_stop(), "the STOP receipt cannot bypass a pending verdict");
    r->verify_pending = false;
    check(window_controller_stop(), "resolved fixture teardown");
    std::filesystem::remove_all(dir);
    {
        WindowControllerConfig bad; bad.journal_dir = "/tmp"; bad.repair.enabled = true; bad.ranks = 1;
        std::string why;
        check(!window_controller_start(bad, 4, &why) && why.find("2..64") != std::string::npos,
              "one server is refused");
        bad.ranks = 65;
        check(!window_controller_start(bad, 4, &why), "more than 64 servers are refused");
    }
    sixteen_frame_harvest();
    weight_w_frames();
    stop_collector(4);
    stop_collector(8);
}
