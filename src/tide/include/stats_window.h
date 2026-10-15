// Runtime window statistics on a server: shared core for the server control
// handlers and the search hot path, kept in free functions so the unit test
// exercises exactly the production logic.
//
// One counter per node id v < N: the distances computed while expanding v
// during the window (v is expanded only on its owner). The load-time ids
// [0, N0) count in the static table, the ids inserted after load in the chunked
// delta table. BEGIN freezes the window's node count N (at most the ids
// this server knows: inserts are fenced out of a controller phase), grows the
// delta table to cover it, zeroes [0, N) and activates the window mid-serving;
// a SNAPSHOT request deactivates it and answers with one snapshot wire v3 frame
// (window_snapshot_wire.h): the owned ids' work, their owner histogram over
// K = n_servers owners, and the neighbour-aggregated work. STOP deactivates
// without a frame.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "server_types.h"
#include "window_snapshot_wire.h"

namespace tide {

// Allocates the static counter table for the load-time ids [0, static_nodes),
// zeroed, empties the delta table and sets the window node count to
// static_nodes (load and RESET; the caller excludes search).
void stats_window_allocate(ServerData& data, uint32_t static_nodes);
// The counter of node `id` (the static table below N0, the delta table above),
// or nullptr when no counter exists for it (beyond the delta table's chunks).
inline std::atomic<uint32_t>* stats_window_counter(const ServerData& data, uint32_t id) {
    if (id < data.stats_window_static_nodes)
        return data.stats_window_w ? &data.stats_window_w[id] : nullptr;
    return data.stats_window_delta.counter(id - data.stats_window_static_nodes);
}
// Freezes the window's node count N = request.nodes, grows the delta table to
// cover [N0, N), zeroes [0, N) and activates window `request.window_epoch`.
// Refuses another request version, N = 0, N above the ids this server knows
// (n_base_load: the coordinator's N counts the ids it inserted, and a server
// that has not applied every insert's broadcast cannot measure them), epoch 0,
// a non-monotone epoch, an unallocated static table and a failed allocation
// (every refusal leaves the previous window as it was). Runs without
// quiescence: counters are relaxed atomics and the window boundary may smear by
// in-flight increments; growth publishes zeroed chunks only.
bool stats_window_begin_local(ServerData& data, const ann::StatsWindowBeginRequest& request);
ann::StatsWindowBeginRequest stats_window_begin_request(uint64_t window_epoch, uint32_t nodes);

// The captured cut of one window, owning every encoder input (no ServerData view).
struct StatsWindowCapturedSnapshot {
    ann::StatsWindowAck ack{};       // status 0 only when the capture succeeded
    uint32_t owners = 0, nodes = 0, sections = 0;
    uint64_t owned_digest = 0, non_owned_w = 0;
    std::vector<uint32_t> w;         // owned ids, ascending
    std::vector<uint16_t> hist;      // owned ids x owners
    std::vector<uint64_t> nbwork;    // every node id
};
// Capture on the caller/I/O thread: deactivates the window (epoch must match),
// reads the counters of [0, N) and, from the rows this rank owns, the
// structure sections. Refuses (ack.status 1) another request version, unknown
// sections, a node count other than the window's N (frozen at BEGIN), a server
// count outside [2, 64] or a rank outside it.
StatsWindowCapturedSnapshot stats_window_capture_snapshot(ServerData& data,
    const WindowSnapshotRequest& request, ann::StatsWindowAck counters);
// The owned part of a full counter array `w_all` (one per window node) plus the
// structure sections: over every owned id u (mem_at(u) == my_partition, u < N,
// the window's node count) in ascending order, its work, its out-neighbours counted
// by owner (hist, K = n_servers columns: every out-neighbour counts, so a
// histogram row sums to the row's degree, the coordinator's row bytes; an
// out-neighbour this server cannot label, an id beyond its membership or an
// entry never set (UINT32_MAX), fails the capture instead of under-counting
// the row, and so does any other owner >= K) and w[u] added
// onto every out-neighbour v < N (nbwork). An owned id without a local row
// contributes a zero row. Rows are read one at a time under the same per-node
// lock the search expansion uses. False for invalid sections, a bad server
// layout, an unlabelled or corrupt neighbour owner (with the histogram), a
// histogram count above 65535 or a w_all of another length.
bool stats_window_capture_owned(ServerData& data, uint32_t sections, const std::vector<uint32_t>& w_all,
                                StatsWindowCapturedSnapshot& out);
// The frame of a capture (typed failure header when the capture failed).
std::vector<char> stats_window_encode_snapshot(const StatsWindowCapturedSnapshot& snapshot);
std::vector<char> stats_window_snapshot_response(ServerData& data,
    const WindowSnapshotRequest& request, ann::StatsWindowAck counters);

struct StatsWindowSnapshotReply {
    std::string identity;
    std::vector<char> response;
};

// One owned encoder, not a query-worker pool. start/finish/pending are I/O-thread
// methods. Completion is published before waking the existing nonblocking pipe;
// only the I/O thread sends. A pending slot remains occupied until finish().
// The owner drains before BEGIN/SNAPSHOT/STOP/RESET/shutdown to retain reply order.
class StatsWindowSnapshotTask {
public:
    using Encoder = std::vector<char> (*)(const StatsWindowCapturedSnapshot&);
    ~StatsWindowSnapshotTask();
    bool start(std::string identity, StatsWindowCapturedSnapshot snapshot, int wake_fd,
               Encoder encoder = stats_window_encode_snapshot);
    bool pending() const { return pending_; }
    bool ready() const { return ready_.load(std::memory_order_acquire); }
    bool finish(StatsWindowSnapshotReply& reply, bool wait);
private:
    std::thread worker_;
    std::atomic<bool> ready_{false};
    bool pending_ = false;
    StatsWindowSnapshotReply reply_;
};

// Deactivates the window. Refused (status 1) unless request.ranks equals this
// server's n_servers and `rank` is below it; an epoch above the request's is
// never accepted (the receipt carries the actual epoch).
WindowStopAck stats_window_stop_local(ServerData& data,
    const WindowStopRequest& request, uint32_t rank);

// Accounts one frontier's distance work: `spans` holds (source global id,
// start offset into the scored batch) per expansion, in order; each span ends
// where the next begins (the last at kinds.size(), the batch length). `kinds`
// is hybrid_score's per-neighbour kind and `reranked[span]` whether that
// expansion re-scored its source exactly. Adds to w(source) the span's scored
// neighbours (exact or ADC; a missing code, scored FLT_MAX, is no work) plus 1
// for a re-rank: the same evaluations the distance counter counts. Ids at or
// beyond the window's node count N are ignored, and so is an id whose delta
// chunk is absent (a search still counting into a window a BEGIN is replacing).
void stats_window_count_work(ServerData& data,
    const std::vector<std::pair<uint32_t, uint32_t>>& spans, const std::vector<uint8_t>& kinds,
    const std::vector<uint8_t>& reranked);

}  // namespace tide
