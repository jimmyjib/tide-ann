// Server window statistics (stats_window.{h,cpp}) through the same free
// functions the server uses: BEGIN (version, node count, epochs), exact batch
// accounting, the owned capture and its snapshot wire v3 frame at K = 4, K = 8
// and K = 64 (owners >= 4 are counted, never folded or dropped; K = 65 is
// refused), STOP at K = 4, 8 and 64 and its epoch rules (idempotent after a
// harvest and at epoch 0, a stale STOP cannot close a newer window, a later
// epoch closes the older window, restart after a BEGIN-only STOP), the
// capture's refusals, and the asynchronous single-slot encoder; BEGIN
// freezes N up to the ids the server knows (above it: refused), the delta table
// of inserted ids grows at BEGIN (across a chunk boundary, while other threads
// count), counts and captures inserted ids, and is emptied by the RESET path.
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "stats_window.h"

namespace {
using namespace tide;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "CHECK FAILED: %s\n", what);
        std::abort();
    }
}

void allocate(ServerData& data, uint32_t nodes, uint32_t servers = 4, uint32_t rank = 0) {
    data.n_servers = servers;
    data.my_partition = rank;
    data.base_static_capacity = nodes;
    data.membership.resize(nodes, rank);
    n_base_set(data, nodes);
    stats_window_allocate(data, nodes);
}
bool begin(ServerData& data, uint64_t epoch) {
    return stats_window_begin_local(data, stats_window_begin_request(epoch, data.stats_window_nodes));
}
WindowSnapshotRequest snapshot_request(uint64_t epoch, uint32_t nodes,
                                       uint32_t sections = kWindowStructureOwnerHist | kWindowStructureNbWork) {
    WindowSnapshotRequest request; request.epoch = epoch; request.nodes = nodes;
    request.sections = static_cast<uint16_t>(sections);
    return request;
}
uint64_t digest_of(const std::vector<uint32_t>& ids) {
    uint64_t digest = kWindowOwnedDigestSeed;
    for (const uint32_t id : ids) digest = window_owned_digest_add(digest, id);
    return digest;
}

void test_begin_rejects_and_activates() {
    ServerData data;
    check(!begin(data, 1), "unallocated table refused");
    allocate(data, 8);
    check(!begin(data, 0), "epoch zero refused");
    check(begin(data, 5) && data.stats_window_active.load(), "first window");
    check(!begin(data, 5) && !begin(data, 4), "non-monotone epochs refused");
    check(begin(data, 6), "next epoch");
    auto other_nodes = stats_window_begin_request(7, 9);
    check(!stats_window_begin_local(data, other_nodes), "a node count above the ids this server knows is refused");
    check(!stats_window_begin_local(data, stats_window_begin_request(7, 0)), "zero nodes refused");
    auto other_version = stats_window_begin_request(7, 8); other_version.version = 2;
    check(!stats_window_begin_local(data, other_version), "another request version refused");
    auto padded = stats_window_begin_request(7, 8); padded.pad[1] = 1;
    check(!stats_window_begin_local(data, padded), "nonzero padding refused");
    check(data.stats_window_epoch.load() == 6, "refusals keep the epoch");
    std::printf("test_begin_rejects_and_activates OK\n");
}

// Full replication's frontier: every neighbour scored exactly, no re-rank.
void count_batch(ServerData& data, const std::vector<std::pair<uint32_t, uint32_t>>& spans, size_t touched) {
    stats_window_count_work(data, spans, std::vector<uint8_t>(touched, kScoreExact),
                            std::vector<uint8_t>(spans.size(), 0));
}

// Hybrid PQ: W(v) counts the evaluations of v's expansion, the re-rank of v
// included and missing codes (FLT_MAX) excluded, like the distance counter.
void test_work_accounting_kinds_and_reranks() {
    ServerData data; allocate(data, 8);
    check(begin(data, 1), "begin");
    // 0 re-ranked, neighbours exact, ADC, missing; 3 not re-ranked, one missing
    // neighbour; 5 re-ranked, no neighbour; 9 beyond the table.
    const std::vector<uint8_t> kinds = {kScoreExact, kScoreAdc, kScoreMissing, kScoreMissing, kScoreAdc};
    stats_window_count_work(data, {{0, 0}, {3, 3}, {5, 4}, {9, 4}}, kinds, {1, 0, 1, 1});
    check(data.stats_window_w[0].load() == 3, "re-rank + exact + ADC; the missing code is not work");
    check(data.stats_window_w[3].load() == 0, "a missing code alone is no work");
    check(data.stats_window_w[5].load() == 1, "a re-rank-only expansion counts its re-rank");
    uint64_t total = 0;
    for (uint32_t v = 0; v < 8; ++v) total += data.stats_window_w[v].load();
    check(total == 4, "the out-of-table source adds nothing");
    std::printf("test_work_accounting_kinds_and_reranks OK\n");
}

void test_batch_accounting_exact() {
    ServerData data; allocate(data, 8);
    check(begin(data, 1), "begin");
    // Expansions of 0 (2 neighbours), 3 (1), 9 (beyond the table: ignored), 5 (0).
    count_batch(data, {{0, 0}, {3, 2}, {9, 3}, {5, 5}}, 5);
    check(data.stats_window_w[0].load() == 2 && data.stats_window_w[3].load() == 1 &&
          data.stats_window_w[5].load() == 0, "span lengths land on their sources");
    uint64_t total = 0;
    for (uint32_t v = 0; v < 8; ++v) total += data.stats_window_w[v].load();
    check(total == 3, "the out-of-table source adds nothing");
    std::printf("test_batch_accounting_exact OK\n");
}

// K = 4, rank 0 owns 0, 4 and 7 (node 4 has no local row).
void build_k4(ServerData& data) {
    allocate(data, 8);
    data.membership = {0, 1, 2, 3, 0, 2, 3, 0};
    n_base_set(data, 8); n_local_set(data, 2);
    data.global_to_local.ensure_capacity_for(7, UINT32_MAX);
    data.global_to_local[0] = 0; data.global_to_local[7] = 1;
    data.adj.ensure_capacity_for(1);
    data.adj[0] = {1, 2, 7, 3};
    data.adj[1] = {0, 5, 6};
    ensure_node_locks(data, 2);
}

void test_owned_capture_and_v3_wire_k4() {
    ServerData data; build_k4(data);
    StatsWindowCapturedSnapshot out;
    const std::vector<uint32_t> w_all = {2, 0, 0, 0, 0, 0, 0, 1};
    check(stats_window_capture_owned(data, kWindowStructureOwnerHist | kWindowStructureNbWork, w_all, out),
          "owned capture");
    // Owners of {1,2,7,3} = {1,2,0,3}; zero row; owners of {0,5,6} = {0,2,3}.
    check(out.w == std::vector<uint32_t>({2, 0, 1}) && out.owners == 4 && out.nodes == 8 &&
          out.owned_digest == digest_of({0, 4, 7}) && out.non_owned_w == 0, "owned work and the owned-id digest");
    check(out.hist == std::vector<uint16_t>({1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1}),
          "owner histogram as u16 rows in ascending owned-id order with a rowless zero row");
    // nbwork: row 0 (w 2) adds 2 to 1,2,7,3; row 7 (w 1) adds 1 to 0,5,6.
    check(out.nbwork == std::vector<uint64_t>({1, 2, 2, 2, 0, 1, 1, 2}), "neighbour-aggregated window work");
    check(stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out) && out.nbwork.empty() &&
          out.hist.size() == 12, "histogram-only section");
    check(stats_window_capture_owned(data, kWindowStructureNbWork, w_all, out) && out.hist.empty() &&
          out.nbwork.size() == 8 && out.w.size() == 3, "nbwork-only section");
    check(!stats_window_capture_owned(data, 0, w_all, out) && !stats_window_capture_owned(data, 1, w_all, out) &&
          !stats_window_capture_owned(data, 8, w_all, out), "empty or unknown sections refused");
    check(!stats_window_capture_owned(data, kWindowStructureNbWork, std::vector<uint32_t>(7, 0), out),
          "a counter array of another length refused");
    const std::vector<uint32_t> stray = {2, 5, 0, 0, 0, 0, 0, 1};
    check(stats_window_capture_owned(data, kWindowStructureNbWork, stray, out) && out.non_owned_w == 5,
          "work on an id this rank does not own is reported, not folded in");

    // The whole path: BEGIN, counted batches, SNAPSHOT -> frame -> decode.
    check(begin(data, 1), "begin");
    count_batch(data, {{0, 0}, {7, 2}}, 3);
    ann::StatsWindowAck counters{}; counters.server_rank = 0;
    counters.search_request_count = 5; counters.search_distance_count = 6;
    const auto bytes = stats_window_snapshot_response(data, snapshot_request(1, 8), counters);
    check(!data.stats_window_active, "the snapshot closes the window");
    check(bytes.size() == sizeof(WindowSnapshotHeader) + 3 * 4 + 12 * 2 + 8 * 8, "exact v3 frame size");
    WindowRankFrame frame; std::string error;
    check(window_snapshot_decode(bytes.data(), bytes.size(), 0, 1, 8, 4,
                                 kWindowStructureOwnerHist | kWindowStructureNbWork, frame, error),
          "the frame decodes");
    const auto& h = frame.header;
    check(h.version == 3 && h.owners == 4 && h.nodes == 8 && h.owned_count == 3 && h.owned_digest == digest_of({0, 4, 7}) &&
          h.ack.total_w == 3 && h.ack.search_request_count == 5 && h.ack.msg_type == ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE,
          "header identity");
    check(frame.w == std::vector<uint32_t>({2, 0, 1}) && frame.hist.size() == 12 &&
          frame.nbwork == std::vector<uint64_t>({1, 2, 2, 2, 0, 1, 1, 2}), "sections round-trip");
    check(window_snapshot_queue_bytes(bytes.data(), bytes.size(), 8, 4) == bytes.size(), "within the queue bound");
    check(!window_snapshot_decode(bytes.data(), bytes.size(), 0, 1, 8, 8,
                                  kWindowStructureOwnerHist | kWindowStructureNbWork, frame, error),
          "a coordinator expecting another K refuses");
    check(!window_snapshot_decode(bytes.data(), bytes.size(), 0, 1, 9, 4,
                                  kWindowStructureOwnerHist | kWindowStructureNbWork, frame, error),
          "another node count refuses");
    check(!window_snapshot_decode(bytes.data(), bytes.size(), 1, 1, 8, 4,
                                  kWindowStructureOwnerHist | kWindowStructureNbWork, frame, error),
          "another rank refuses");
    check(!window_snapshot_decode(bytes.data(), bytes.size(), 0, 1, 8, 4, kWindowStructureOwnerHist, frame, error) &&
          error.find("sections") != std::string::npos, "other sections refuse");
    for (size_t at : {size_t(70), sizeof(WindowSnapshotHeader) + 1, bytes.size() - 1}) {
        auto tampered = bytes; tampered[at] ^= 1;
        check(!window_snapshot_decode(tampered.data(), tampered.size(), 0, 1, 8, 4,
                                      kWindowStructureOwnerHist | kWindowStructureNbWork, frame, error),
              "any flipped byte refuses");
    }
    check(window_snapshot_encode(counters, 1, 8, 6, 0, 0, {1}, {1}, std::vector<uint64_t>(8, 0)).empty() &&
          window_snapshot_encode(counters, 4, 8, 6, 0, 0, {1}, {1, 1, 1}, std::vector<uint64_t>(8, 0)).empty() &&
          window_snapshot_encode(counters, 4, 8, 6, 0, 0, {1}, {1, 1, 1, 1}, std::vector<uint64_t>(7, 0)).empty(),
          "inconsistent owner counts or section lengths encode no frame");
    std::printf("test_owned_capture_and_v3_wire_k4 OK\n");
}

// K = 8: rank 5 owns 2, 5 and 13; neighbours on owners 4..7 count in their own
// columns (a capture limited to four columns would drop every owner >= 4).
void test_capture_k8_counts_high_owners() {
    ServerData data; allocate(data, 16, 8, 5);
    for (uint32_t v = 0; v < 16; ++v) data.membership[v] = v % 8;
    data.membership[13] = 5;  // 13 % 8 = 5 already; 2 is moved onto rank 5
    data.membership[2] = 5;
    n_base_set(data, 16); n_local_set(data, 3);
    data.global_to_local.ensure_capacity_for(15, UINT32_MAX);
    data.global_to_local[2] = 0; data.global_to_local[5] = 1; data.global_to_local[13] = 2;
    data.adj.ensure_capacity_for(2);
    data.adj[0] = {4, 12, 7, 6, 0};      // owners 4, 4, 7, 6, 0
    data.adj[1] = {13, 15, 1};           // owners 5, 7, 1
    data.adj[2] = {11, 3, 14, 2};        // owners 3, 3, 6, 5
    ensure_node_locks(data, 3);
    check(begin(data, 3), "K = 8 window");
    count_batch(data, {{2, 0}, {5, 5}, {13, 8}}, 12);
    ann::StatsWindowAck counters{}; counters.server_rank = 5;
    const auto bytes = stats_window_snapshot_response(data, snapshot_request(3, 16), counters);
    WindowRankFrame frame; std::string error;
    check(window_snapshot_decode(bytes.data(), bytes.size(), 5, 3, 16, 8,
                                 kWindowStructureOwnerHist | kWindowStructureNbWork, frame, error),
          "K = 8 frame decodes");
    const std::vector<uint16_t> expected = {
        1, 0, 0, 0, 2, 0, 1, 1,   // id 2
        0, 1, 0, 0, 0, 1, 0, 1,   // id 5
        0, 0, 0, 2, 0, 1, 1, 0};  // id 13
    check(frame.header.owners == 8 && frame.header.owned_count == 3 && frame.hist == expected,
          "owners 4..7 are counted in their own columns");
    check(frame.w == std::vector<uint32_t>({5, 3, 4}) && frame.header.ack.total_w == 12 &&
          frame.header.owned_digest == digest_of({2, 5, 13}), "owned work and digest at K = 8");
    // STOP at K = 8: the request must name eight ranks.
    check(begin(data, 4), "next window");
    WindowStopRequest stop; stop.epoch = 4; stop.ranks = 4;
    check(stats_window_stop_local(data, stop, 5).status && data.stats_window_active, "a four-rank STOP is refused at K = 8");
    stop.ranks = 8;
    check(stats_window_stop_local(data, stop, 8).status, "a rank beyond K is refused");
    const auto ack = stats_window_stop_local(data, stop, 5);
    check(!ack.status && !ack.active && ack.actual_epoch == 4 && !data.stats_window_active, "an eight-rank STOP closes the window");
    std::printf("test_capture_k8_counts_high_owners OK\n");
}

// STOP around a snapshot harvest on wire v3: a STOP of a closed window is idempotent.
void test_stop_epochs_after_snapshot() {
    ServerData data; build_k4(data);
    WindowStopRequest stop; stop.ranks = 4;  // epoch 0
    auto off = stats_window_stop_local(data, stop, 0);
    check(!off.status && !off.active && off.actual_epoch == 0 && off.requested_epoch == 0,
          "a STOP at epoch 0 with nothing open is idempotent");
    check(!stats_window_stop_local(data, stop, 0).status && !data.stats_window_active, "and stays so");
    check(begin(data, 1), "window 1");
    count_batch(data, {{0, 0}, {7, 2}}, 3);
    ann::StatsWindowAck counters{}; counters.server_rank = 0;
    const auto bytes = stats_window_snapshot_response(data, snapshot_request(1, 8), counters);
    WindowRankFrame frame; std::string error;
    check(window_snapshot_decode(bytes.data(), bytes.size(), 0, 1, 8, 4,
                                 kWindowStructureOwnerHist | kWindowStructureNbWork, frame, error) &&
          frame.header.ack.total_w == 3, "window 1 harvested");
    check(!data.stats_window_active && !begin(data, 1), "the harvest closes the window and keeps its epoch");
    stop.epoch = 1;
    off = stats_window_stop_local(data, stop, 0);
    check(!off.status && !off.active && off.actual_epoch == 1, "a STOP after the harvest is idempotent");
    check(!stats_window_stop_local(data, stop, 0).status, "a repeated STOP too");
    check(stats_window_capture_snapshot(data, snapshot_request(1, 8), counters).ack.status &&
          data.stats_window_w[0].load() == 2 && data.stats_window_w[7].load() == 1,
          "no second capture of a harvested window; the STOPs leave the counters alone");
    check(begin(data, 2), "the next window remains usable");
    check(stats_window_stop_local(data, stop, 0).status && data.stats_window_active,
          "a stale STOP (epoch 1) cannot close the newer window 2");
    stop.epoch = 3;  // BEGIN 3 reached other ranks but not this one
    off = stats_window_stop_local(data, stop, 0);
    check(!off.status && off.requested_epoch == 3 && off.actual_epoch == 2 && !off.active && !data.stats_window_active,
          "a STOP naming a later epoch closes the older window and reports its actual epoch");
    check(begin(data, 4) && data.stats_window_active, "a window restarts after a BEGIN-only STOP");
    stop.epoch = 4;
    check(!stats_window_stop_local(data, stop, 0).status && !data.stats_window_active, "STOP closes the restarted window");
    std::printf("test_stop_epochs_after_snapshot OK\n");
}

// K = 64, the largest owner count: rank 63 owns 63 and 127; its rows' neighbours
// count in 64 columns; the frame round-trips and fits the queue bound; K = 65 is
// refused by the encoder, the decoder, the queue bound, STOP and the capture.
void test_capture_k64() {
    ServerData data; allocate(data, 128, 64, 63);
    for (uint32_t v = 0; v < 128; ++v) data.membership[v] = v % 64;
    n_base_set(data, 128); n_local_set(data, 2);
    data.global_to_local.ensure_capacity_for(127, UINT32_MAX);
    data.global_to_local[63] = 0; data.global_to_local[127] = 1;
    data.adj.ensure_capacity_for(1);
    data.adj[0] = {0, 1, 62, 63, 126};  // owners 0, 1, 62, 63, 62
    data.adj[1] = {64, 127, 125};       // owners 0, 63, 61
    ensure_node_locks(data, 2);
    check(begin(data, 1), "K = 64 window");
    count_batch(data, {{63, 0}, {127, 5}}, 8);
    ann::StatsWindowAck counters{}; counters.server_rank = 63;
    const uint32_t sections = kWindowStructureOwnerHist | kWindowStructureNbWork;
    const auto bytes = stats_window_snapshot_response(data, snapshot_request(1, 128), counters);
    check(bytes.size() == sizeof(WindowSnapshotHeader) + 2 * 4 + 2 * 64 * 2 + 128 * 8, "exact K = 64 frame size");
    WindowRankFrame frame; std::string error;
    check(window_snapshot_decode(bytes.data(), bytes.size(), 63, 1, 128, 64, sections, frame, error),
          "the K = 64 frame decodes");
    std::vector<uint16_t> expected(128, 0);
    expected[0] = 1; expected[1] = 1; expected[62] = 2; expected[63] = 1;  // id 63
    expected[64 + 0] = 1; expected[64 + 61] = 1; expected[64 + 63] = 1;   // id 127
    check(frame.header.owners == 64 && frame.header.owned_count == 2 && frame.hist == expected &&
          frame.w == std::vector<uint32_t>({5, 3}) && frame.header.ack.total_w == 8 &&
          frame.header.owned_digest == digest_of({63, 127}), "owners up to 63 are counted in their own columns");
    check(window_snapshot_queue_bytes(bytes.data(), bytes.size(), 128, 64) == bytes.size(), "within the K = 64 queue bound");
    check(window_snapshot_queue_bytes(bytes.data(), bytes.size(), 128, 65) == sizeof(WindowSnapshotHeader),
          "the queue bound keeps only the header for K = 65");
    check(!window_snapshot_decode(bytes.data(), bytes.size(), 63, 1, 128, 65, sections, frame, error),
          "a coordinator expecting K = 65 refuses");
    check(window_snapshot_encode(counters, 65, 128, kWindowStructureNbWork, 0, 0, {1}, {},
                                 std::vector<uint64_t>(128, 0)).empty(), "no frame encodes 65 owners");
    WindowStopRequest stop; stop.epoch = 1; stop.ranks = 64;
    check(!stats_window_stop_local(data, stop, 63).status, "a 64-rank STOP");
    data.n_servers = 65; stop.ranks = 65;
    StatsWindowCapturedSnapshot out;
    check(stats_window_stop_local(data, stop, 63).status &&
          !stats_window_capture_owned(data, kWindowStructureOwnerHist, std::vector<uint32_t>(128, 0), out),
          "a 65-server rank refuses STOP and capture");
    std::printf("test_capture_k64 OK\n");
}

void test_capture_refusals() {
    ServerData data; build_k4(data);
    StatsWindowCapturedSnapshot out;
    const std::vector<uint32_t> w_all(8, 0);
    data.membership[2] = 7;  // an owner label >= K on an out-neighbour of row 0 is corrupt: never folded
    check(!stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out), "owner >= K fails the capture");
    // The coordinator prices a row as 4 + 4 x its histogram row sum, so a
    // neighbour this server cannot label fails the capture instead of under-counting the row:
    // an entry never set (UINT32_MAX) and an id beyond the server's membership. Without the
    // histogram (nbwork only) the row is not priced and such a neighbour is ignored.
    data.membership[2] = UINT32_MAX;
    check(!stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out) &&
          !stats_window_capture_owned(data, kWindowStructureOwnerHist | kWindowStructureNbWork, w_all, out),
          "an out-neighbour without an owner label fails the histogram capture");
    check(stats_window_capture_owned(data, kWindowStructureNbWork, w_all, out) && out.hist.empty(),
          "the nbwork-only capture does not price rows");
    data.membership[2] = 2;
    data.adj[0] = {1, 2, 7, 3, 9};  // 9: beyond this server's membership (8 ids)
    check(!stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out),
          "an out-neighbour beyond the server's membership fails the histogram capture");
    check(stats_window_capture_owned(data, kWindowStructureNbWork, w_all, out) && out.nbwork.size() == 8,
          "the nbwork-only capture ignores it (v >= N)");
    data.adj[0] = {1, 2, 7, 3};
    check(stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out) &&
          out.hist[0] + out.hist[1] + out.hist[2] + out.hist[3] == 4, "every neighbour counts: the row sum is the degree");
    data.n_servers = 1;
    check(!stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out), "one server refused");
    data.n_servers = 65;
    check(!stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out), "65 servers refused");
    data.n_servers = 4; data.my_partition = 4;
    check(!stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out), "a rank outside K refused");
    data.my_partition = 0;
    data.adj[0] = std::vector<uint32_t>(70000, 1);  // 70,000 neighbours on owner 1
    check(!stats_window_capture_owned(data, kWindowStructureOwnerHist, w_all, out), "a count above 65535 refused");
    data.adj[0] = {1, 2, 7, 3};
    // Request validation: version, sections, nodes, epoch.
    ann::StatsWindowAck counters{};
    check(begin(data, 1), "window");
    auto v2 = snapshot_request(1, 8); v2.version = 2;
    check(stats_window_capture_snapshot(data, v2, counters).ack.status && data.stats_window_active,
          "a version-2 request is refused without closing the window");
    check(stats_window_capture_snapshot(data, snapshot_request(1, 8, 1), counters).ack.status,
          "an unknown structure section is refused");
    check(stats_window_capture_snapshot(data, snapshot_request(1, 9), counters).ack.status, "another node count refused");
    check(stats_window_capture_snapshot(data, snapshot_request(2, 8), counters).ack.status && data.stats_window_active,
          "another epoch refused");
    const auto failure = stats_window_snapshot_response(data, snapshot_request(2, 8), counters);
    WindowSnapshotHeader header; std::memcpy(&header, failure.data(), sizeof(header));
    check(failure.size() == sizeof(WindowSnapshotHeader) && header.ack.status == 1 &&
          header.ack.msg_type == ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE && header.ack.window_epoch == 2,
          "a refusal is a typed failure header");
    WindowStopRequest stop; stop.epoch = 1; stop.ranks = 4;
    check(!stats_window_stop_local(data, stop, 0).status, "K = 4 STOP");
    stop.epoch = 0;
    check(stats_window_stop_local(data, stop, 0).status, "a STOP below the actual epoch is refused");
    std::printf("test_capture_refusals OK\n");
}

// N grows past the load-time ids. Static table N0 = 8 (rank 0 owns every
// load-time id); ids 8 .. 100,009 are inserted (delta indices 0 .. 100,001,
// two delta chunks); the server knows them (n_base) and owns 8 and 100,009.
// BEGIN with N above n_base is refused; BEGIN with N = 100,010 grows the delta
// table, counts inserted ids, ignores ids >= N, and the frame carries them;
// the next BEGIN zeroes them; an id whose mapping chunk is absent (100,009:
// global_to_local chunk 1 was never allocated here) is an owned id without a
// row (checked lookup), never a crash.
void test_begin_node_count_and_delta_table() {
    ServerData data; allocate(data, 8);
    check(data.stats_window_static_nodes == 8 && data.stats_window_nodes.load() == 8 &&
          data.stats_window_delta.allocated_chunks() == 0, "load: the static table only");
    constexpr uint32_t kNodes = 100010, kLast = kNodes - 1;
    check(!stats_window_begin_local(data, stats_window_begin_request(1, kNodes)),
          "BEGIN with N above n_base is refused (inserts not applied here)");
    check(data.stats_window_delta.allocated_chunks() == 0 && data.stats_window_nodes.load() == 8 &&
          !data.stats_window_active, "the refusal changed nothing");
    server_ensure_mem_capacity(data, kLast);
    mem_set(data, 8, 0); mem_set(data, kLast, 0);
    n_base_set(data, kNodes);
    check(stats_window_begin_local(data, stats_window_begin_request(1, kNodes)) && data.stats_window_active,
          "BEGIN over the inserted ids");
    check(data.stats_window_nodes.load() == kNodes && data.stats_window_delta.allocated_chunks() == 2 &&
          stats_window_counter(data, kLast) != nullptr && stats_window_counter(data, 8) != nullptr,
          "N frozen; the delta table covers [N0, N) across its chunk boundary");
    stats_window_count_work(data, {{0, 0}, {8, 1}, {kLast, 3}, {kNodes, 6}},
                            std::vector<uint8_t>(9, kScoreExact), std::vector<uint8_t>(4, 0));
    check(stats_window_counter(data, 0)->load() == 1 && stats_window_counter(data, 8)->load() == 2 &&
          stats_window_counter(data, kLast)->load() == 3, "inserted ids count in the delta table");
    ann::StatsWindowAck counters{}; counters.server_rank = 0;
    check(stats_window_capture_snapshot(data, snapshot_request(1, 8, kWindowStructureOwnerHist), counters).ack.status &&
          data.stats_window_active, "a snapshot naming the load-time count is refused: N is frozen at BEGIN");
    const auto bytes = stats_window_snapshot_response(data, snapshot_request(1, kNodes, kWindowStructureOwnerHist),
                                                      counters);
    WindowRankFrame frame; std::string error;
    check(window_snapshot_decode(bytes.data(), bytes.size(), 0, 1, kNodes, 4, kWindowStructureOwnerHist, frame, error),
          "the frame over N ids decodes");
    std::vector<uint32_t> expected_w(10, 0);
    expected_w[0] = 1; expected_w[8] = 2; expected_w[9] = 3;
    std::vector<uint32_t> owned = {0, 1, 2, 3, 4, 5, 6, 7, 8, kLast};
    check(frame.header.nodes == kNodes && frame.header.owned_count == 10 && frame.w == expected_w &&
          frame.header.ack.total_w == 6 && frame.header.owned_digest == digest_of(owned) &&
          frame.hist == std::vector<uint16_t>(40, 0), "owned inserted ids are in the frame (rowless: zero rows)");
    check(stats_window_begin_local(data, stats_window_begin_request(2, kNodes)) &&
          stats_window_counter(data, kLast)->load() == 0 && stats_window_counter(data, 8)->load() == 0 &&
          stats_window_counter(data, 0)->load() == 0, "the next BEGIN zeroes the static and the delta counters");
    check(stats_window_begin_local(data, stats_window_begin_request(3, 8)) && data.stats_window_nodes.load() == 8,
          "a later window may cover fewer ids");
    stats_window_count_work(data, {{8, 0}}, {kScoreExact}, {0});
    check(stats_window_counter(data, 8)->load() == 0, "an id at or beyond N is not counted");
    stats_window_allocate(data, 8);  // the RESET path
    check(data.stats_window_delta.allocated_chunks() == 0 && data.stats_window_nodes.load() == 8 &&
          stats_window_counter(data, 8) == nullptr, "RESET empties the delta table");
    stats_window_count_work(data, {{9, 0}}, {kScoreExact}, {0});  // no chunk: ignored, no crash
    std::printf("test_begin_node_count_and_delta_table OK\n");
}

// The delta table grows at BEGIN while other threads count (inserts are fenced
// out of a controller phase, searches are not): BEGINs with growing N (8 ->
// 50,008 -> 100,010 -> 250,008: up to three delta chunks), each followed by a
// capture, run while four threads count expansions of random sources over
// [0, 300,000), some beyond N or in chunks not allocated yet. A stress test
// (it also runs under ThreadSanitizer, docs/LIMITATIONS.md "Race detection");
// afterwards a quiescent window counts exactly.
void test_delta_growth_while_counting() {
    ServerData data; allocate(data, 8);
    n_base_set(data, 300000);
    std::atomic<bool> running{true};
    std::atomic<uint64_t> batches{0};
    std::vector<std::thread> counters;
    for (uint32_t t = 0; t < 4; ++t) {
        counters.emplace_back([&, t] {
            uint64_t state = 0x9E3779B97F4A7C15ull * (t + 1);
            std::vector<std::pair<uint32_t, uint32_t>> spans(16);
            const std::vector<uint8_t> kinds(32, kScoreExact), reranked(16, 1);
            while (running.load(std::memory_order_relaxed)) {
                for (uint32_t i = 0; i < 16; ++i) {
                    state = state * 6364136223846793005ull + 1442695040888963407ull;
                    spans[i] = {static_cast<uint32_t>((state >> 33) % 300000), 2 * i};
                }
                stats_window_count_work(data, spans, kinds, reranked);
                batches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    uint64_t epoch = 0;
    ann::StatsWindowAck ack{}; ack.server_rank = 0;
    for (const uint32_t nodes : {8u, 50008u, 100010u, 250008u, 250008u}) {
        check(stats_window_begin_local(data, stats_window_begin_request(++epoch, nodes)), "BEGIN while counting");
        const uint64_t before = batches.load();
        while (batches.load() < before + 64) std::this_thread::yield();
        auto snapshot = stats_window_capture_snapshot(data, snapshot_request(epoch, nodes, kWindowStructureOwnerHist), ack);
        check(!snapshot.ack.status, "capture while counting");
    }
    running = false;
    for (auto& thread : counters) thread.join();
    check(data.stats_window_delta.allocated_chunks() == 3, "three delta chunks cover 250,008 ids");
    // Quiescent: one exact window over N = 250,008.
    check(stats_window_begin_local(data, stats_window_begin_request(++epoch, 250008)), "quiescent BEGIN");
    stats_window_count_work(data, {{3, 0}, {100007, 1}, {250007, 3}, {250008, 4}},
                            std::vector<uint8_t>(6, kScoreExact), std::vector<uint8_t>(4, 0));
    const auto captured = stats_window_capture_snapshot(data, snapshot_request(epoch, 250008, kWindowStructureOwnerHist),
                                                        ack);
    // Rank 0 owns only the load-time ids here (inserted ids have no label): their work is non-owned.
    check(!captured.ack.status && captured.ack.total_w == 1 + 2 + 1 && captured.w.size() == 8 && captured.w[3] == 1 &&
          captured.non_owned_w == 3, "the quiescent window is exact; nothing counted beyond N");
    std::printf("test_delta_growth_while_counting OK (%llu batches)\n", (unsigned long long)batches.load());
}

struct EncodeBarrier {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
    void await_entry() {
        std::unique_lock<std::mutex> lock(mutex);
        check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "bounded encoder entered");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};
EncodeBarrier* encode_barrier = nullptr;  // Set before start, cleared after join.
std::vector<char> blocked_encode(const StatsWindowCapturedSnapshot& snapshot) {
    {
        std::unique_lock<std::mutex> lock(encode_barrier->mutex);
        encode_barrier->entered = true;
        encode_barrier->cv.notify_all();
        if (!encode_barrier->cv.wait_for(lock, std::chrono::seconds(5), [] { return encode_barrier->released; }))
            throw std::runtime_error("encoder test release deadline");
    }
    return stats_window_encode_snapshot(snapshot);
}
std::vector<char> throwing_encode(const StatsWindowCapturedSnapshot&) {
    throw std::runtime_error("injected encoding failure");
}
std::vector<char> empty_encode(const StatsWindowCapturedSnapshot&) { return {}; }

void test_async_snapshot_owned_cut_and_single_slot() {
    ServerData data, reference;
    build_k4(data); build_k4(reference);
    check(begin(data, 1) && begin(reference, 1), "async and reference windows");
    count_batch(data, {{0, 0}, {7, 2}}, 3);
    count_batch(reference, {{0, 0}, {7, 2}}, 3);
    ann::StatsWindowAck counters{}; counters.server_rank = 0;
    counters.search_request_count = 17; counters.search_distance_count = 23;
    const auto expected = stats_window_snapshot_response(reference, snapshot_request(1, 8), counters);
    auto captured = stats_window_capture_snapshot(data, snapshot_request(1, 8), counters);
    check(!data.stats_window_active && !captured.ack.status && captured.ack.total_w == 3, "the capture is the cut");
    int fds[2]; check(::pipe(fds) == 0, "owned test wake pipe");
    EncodeBarrier barrier; encode_barrier = &barrier;
    StatsWindowSnapshotTask task;
    const std::string identity("coord\0epoch-1", 13);
    check(task.start(identity, std::move(captured), fds[1], blocked_encode), "first encoder admitted");
    barrier.await_entry();
    StatsWindowSnapshotReply reply;
    check(task.pending() && !task.ready() && !task.finish(reply, false),
          "I/O completion poll does not wait for a blocked encoder");
    check(!task.start("second", {}, fds[1]), "single slot: a second capture is refused while one is pending");
    barrier.release();
    check(task.finish(reply, true), "ordered control barrier joins pending work");
    check(reply.identity == identity && reply.response == expected, "identity and authenticated bytes preserved");
    check(!task.pending() && !task.ready() && !task.finish(reply, true), "the slot is free after finish");
    char signal = 0;
    check(::read(fds[0], &signal, 1) == 1 && signal == 1, "completion wakes the I/O pipe");
    encode_barrier = nullptr;
    check(begin(data, 2), "BEGIN successor after completion");
    WindowStopRequest stop; stop.epoch = 2; stop.ranks = 4;
    check(!stats_window_stop_local(data, stop, 0).status && !data.stats_window_active, "STOP closes the successor");
    ::close(fds[0]); ::close(fds[1]);
    std::printf("test_async_snapshot_owned_cut_and_single_slot OK\n");
}

void test_async_snapshot_failures_and_shutdown_join() {
    ServerData data; build_k4(data);
    ann::StatsWindowAck counters{}; counters.server_rank = 0;
    for (auto encoder : {throwing_encode, empty_encode}) {
        check(begin(data, data.stats_window_epoch.load() + 1), "window");
        const auto epoch = data.stats_window_epoch.load();
        auto failed = stats_window_capture_snapshot(data, snapshot_request(epoch, 8), counters);
        StatsWindowSnapshotTask task;
        check(task.start("failure-owner", failed, -1, encoder), "failing encoder admitted");
        StatsWindowSnapshotReply reply;
        check(task.finish(reply, true) && reply.response.size() == sizeof(WindowSnapshotHeader),
              "an encoder failure leaves the preallocated typed failure");
        WindowSnapshotHeader header; std::memcpy(&header, reply.response.data(), sizeof(header));
        check(header.ack.msg_type == ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE && header.ack.status == 1 &&
              header.ack.window_epoch == epoch, "typed, epoch-exact failure");
    }
    EncodeBarrier barrier; encode_barrier = &barrier;
    std::promise<void> started_promise; auto started = started_promise.get_future();
    auto owner = std::async(std::launch::async, [&] {
        StatsWindowSnapshotTask task;
        check(task.start("shutdown-owner", {}, -1, blocked_encode), "shutdown encoder admitted");
        barrier.await_entry();
        started_promise.set_value();
        // Destructor must join; no detached access to destroyed task state.
    });
    check(started.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "owner reached destructor");
    check(owner.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready,
          "shutdown cannot finish while the encoder remains blocked");
    barrier.release();
    check(owner.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "shutdown joins the released encoder within the test deadline");
    owner.get(); encode_barrier = nullptr;
    std::printf("test_async_snapshot_failures_and_shutdown_join OK\n");
}
}  // namespace

int main() {
    test_begin_rejects_and_activates();
    test_batch_accounting_exact();
    test_work_accounting_kinds_and_reranks();
    test_owned_capture_and_v3_wire_k4();
    test_capture_k8_counts_high_owners();
    test_stop_epochs_after_snapshot();
    test_capture_k64();
    test_capture_refusals();
    test_async_snapshot_owned_cut_and_single_slot();
    test_async_snapshot_failures_and_shutdown_join();
    test_begin_node_count_and_delta_table();
    test_delta_growth_while_counting();
    std::printf("ALL STATS WINDOW TESTS PASSED\n");
    return 0;
}
