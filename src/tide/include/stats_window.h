// Per-node work counters for controller windows (BEGIN / SNAPSHOT / STOP).
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

// Allocates the zeroed static table for [0, static_nodes) and empties the delta table (load, RESET).
void stats_window_allocate(ServerData& data, uint32_t static_nodes);
// counter of node `id`, or nullptr when none exists
inline std::atomic<uint32_t>* stats_window_counter(const ServerData& data, uint32_t id) {
    if (id < data.stats_window_static_nodes)
        return data.stats_window_w ? &data.stats_window_w[id] : nullptr;
    return data.stats_window_delta.counter(id - data.stats_window_static_nodes);
}
// BEGIN: fixes N, grows the delta table, zeroes [0, N) and activates the window. N above the ids
// this server knows, a stale epoch or a failed allocation rejects it and keeps the previous window.
bool stats_window_begin_local(ServerData& data, const ann::StatsWindowBeginRequest& request);
ann::StatsWindowBeginRequest stats_window_begin_request(uint64_t window_epoch, uint32_t nodes);

// one window's captured counters and sections
struct StatsWindowCapturedSnapshot {
    ann::StatsWindowAck ack{};       // status 0 only when the capture succeeded
    uint32_t owners = 0, nodes = 0, sections = 0;
    uint64_t owned_digest = 0, non_owned_w = 0;
    std::vector<uint32_t> w;         // owned ids, ascending
    std::vector<uint16_t> hist;      // owned ids x owners
    std::vector<uint64_t> nbwork;    // every node id
};
// SNAPSHOT on the I/O thread: deactivates the window and captures [0, N) and the sections.
StatsWindowCapturedSnapshot stats_window_capture_snapshot(ServerData& data,
    const WindowSnapshotRequest& request, ann::StatsWindowAck counters);
// Owned ids' work, owner histogram and nbwork. False if a neighbor has no valid owner.
bool stats_window_capture_owned(ServerData& data, uint32_t sections, const std::vector<uint32_t>& w_all,
                                StatsWindowCapturedSnapshot& out);
// the frame of a capture (a failure header when it failed)
std::vector<char> stats_window_encode_snapshot(const StatsWindowCapturedSnapshot& snapshot);
std::vector<char> stats_window_snapshot_response(ServerData& data,
    const WindowSnapshotRequest& request, ann::StatsWindowAck counters);

struct StatsWindowSnapshotReply {
    std::string identity;
    std::vector<char> response;
};

// One encoder thread. Its methods run on the I/O thread, which alone sends the reply. The I/O
// thread finishes it before BEGIN, SNAPSHOT, STOP, RESET and shutdown to keep reply order.
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

// STOP: deactivates the window. Rejected unless request.ranks and `rank` match this server.
WindowStopAck stats_window_stop_local(ServerData& data,
    const WindowStopRequest& request, uint32_t rank);

// Adds each expansion's scored neighbors (plus 1 for a re-rank) to w(source). Ids >= N are ignored.
void stats_window_count_work(ServerData& data,
    const std::vector<std::pair<uint32_t, uint32_t>>& spans, const std::vector<uint8_t>& kinds,
    const std::vector<uint8_t>& reranked);

}  // namespace tide
