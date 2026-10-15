#include "stats_window.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <unistd.h>

namespace tide {

ann::StatsWindowBeginRequest stats_window_begin_request(uint64_t window_epoch, uint32_t nodes) {
    ann::StatsWindowBeginRequest request{};
    request.msg_type = ann::MessageType::STATS_WINDOW_BEGIN_REQUEST;
    request.version = kStatsWindowBeginVersion;
    request.nodes = nodes;
    request.window_epoch = window_epoch;
    return request;
}

void stats_window_allocate(ServerData& data, uint32_t static_nodes) {
    data.stats_window_active.store(false, std::memory_order_release);
    data.stats_window_delta.clear();
    data.stats_window_static_nodes = static_nodes;
    data.stats_window_w = std::make_unique<std::atomic<uint32_t>[]>(static_nodes);
    for (uint32_t i = 0; i < static_nodes; ++i) data.stats_window_w[i].store(0, std::memory_order_relaxed);
    data.stats_window_nodes.store(static_nodes, std::memory_order_release);
}

bool stats_window_begin_local(ServerData& data, const ann::StatsWindowBeginRequest& request) {
    if (request.msg_type != ann::MessageType::STATS_WINDOW_BEGIN_REQUEST ||
        request.version != kStatsWindowBeginVersion || request.pad[0] || request.pad[1] ||
        request.window_epoch == 0 || data.stats_window_w == nullptr) {
        return false;
    }
    // N: every id of the coordinator's membership. A controller phase starts
    // after a drain and refuses inserts while it runs, so every server knows
    // the same ids; a server that has not applied an insert's broadcast (N
    // above its n_base) cannot measure that id and refuses the window.
    const uint32_t nodes = request.nodes;
    if (nodes == 0 || nodes > n_base_load(data)) return false;
    std::lock_guard<std::mutex> lock(data.stats_window_mutex);
    if (request.window_epoch <= data.stats_window_epoch.load(std::memory_order_relaxed)) return false;
    const uint32_t static_nodes = data.stats_window_static_nodes;
    try {
        // Grown before anything changes: a failed allocation keeps the previous window.
        if (nodes > static_nodes && !data.stats_window_delta.ensure_capacity_for(nodes - static_nodes - 1))
            return false;
    } catch (...) {
        return false;
    }
    data.stats_window_active.store(false, std::memory_order_release);
    for (uint32_t i = 0; i < static_nodes; ++i) data.stats_window_w[i].store(0, std::memory_order_relaxed);
    data.stats_window_delta.zero(nodes > static_nodes ? nodes - static_nodes : 0);
    data.stats_window_nodes.store(nodes, std::memory_order_relaxed);
    data.stats_window_epoch.store(request.window_epoch, std::memory_order_relaxed);
    // Release: a search that sees the window active also sees N and the zeroed counters.
    data.stats_window_active.store(true, std::memory_order_release);
    return true;
}

bool stats_window_capture_owned(ServerData& data, uint32_t sections, const std::vector<uint32_t>& w_all,
                                StatsWindowCapturedSnapshot& out) {
    out.w.clear(); out.hist.clear(); out.nbwork.clear();
    const uint32_t owners = data.n_servers;
    const uint32_t nodes = data.stats_window_nodes.load(std::memory_order_acquire);
    if (!window_structure_sections_valid(sections) || !window_owners_valid(owners) ||
        data.my_partition >= owners || !nodes || w_all.size() != nodes) return false;
    const bool want_hist = sections & kWindowStructureOwnerHist;
    const bool want_nbwork = sections & kWindowStructureNbWork;
    const size_t membership_size = mem_size(data);
    const uint32_t n_base = n_base_load(data);
    const uint32_t n_local = n_local_load(data);
    out.owners = owners; out.nodes = nodes; out.sections = sections;
    out.owned_digest = kWindowOwnedDigestSeed; out.non_owned_w = 0;
    if (want_nbwork) out.nbwork.assign(nodes, 0);
    std::vector<uint32_t> counts(owners);
    for (uint32_t id = 0; id < nodes; ++id) {
        const bool owned = id < membership_size && mem_at(data, id) == data.my_partition;
        if (!owned) { out.non_owned_w += w_all[id]; continue; }
        out.owned_digest = window_owned_digest_add(out.owned_digest, id);
        out.w.push_back(w_all[id]);
        if (!want_hist && !want_nbwork) continue;
        std::fill(counts.begin(), counts.end(), 0u);
        // Checked lookup: never index an unallocated mapping chunk.
        const uint32_t local_idx = id < n_base ? g2l_checked(data, id) : UINT32_MAX;
        const uint64_t weight = w_all[id];
        if (valid_search_source(id, n_base, local_idx, n_local)) {
            // The search expansion's per-node lock, one row at a time; the
            // row is scanned in place instead of copied.
            std::unique_lock<std::mutex> row_lock;
            if (local_idx < data.node_locks.size())
                row_lock = std::unique_lock<std::mutex>(data.node_locks[local_idx].mtx);
            for (const uint32_t v : data.adj[local_idx]) {
                if (want_nbwork && v < nodes) out.nbwork[v] += weight;
                if (want_hist) {
                    // Every out-neighbour counts in its owner's column: the
                    // coordinator prices a row's MOVE as 4 + 4 x the histogram
                    // row's sum, so a neighbour this server cannot label (an
                    // id beyond its membership, or an entry never set) would
                    // under-count the row the MOVE ships. The
                    // capture fails instead; a controller phase starts after a
                    // drain, so every server knows every id. A corrupt label
                    // (>= K) never folds either.
                    const uint32_t owner = v < membership_size ? mem_at(data, v) : UINT32_MAX;
                    if (owner >= owners) return false;
                    ++counts[owner];
                }
            }
        }
        if (want_hist) {
            for (const uint32_t count : counts) {
                if (count > UINT16_MAX) return false;
                out.hist.push_back(static_cast<uint16_t>(count));
            }
        }
    }
    return true;
}

StatsWindowCapturedSnapshot stats_window_capture_snapshot(ServerData& data,
    const WindowSnapshotRequest& request, ann::StatsWindowAck ack) {
    ack.msg_type = ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE;
    ack.status = 1; ack.window_epoch = request.epoch; ack.total_w = 0; ack.reserved_total = 0;
    StatsWindowCapturedSnapshot snapshot;
    try {
        if (request.msg_type == ann::MessageType::STATS_WINDOW_SNAPSHOT_REQUEST &&
            request.version == kWindowSnapshotVersion && window_structure_sections_valid(request.sections) &&
            request.epoch != 0 && data.stats_window_w != nullptr &&
            request.nodes == data.stats_window_nodes.load(std::memory_order_acquire)) {
            bool cut = false;
            {
                std::lock_guard<std::mutex> lock(data.stats_window_mutex);
                if (data.stats_window_epoch.load(std::memory_order_relaxed) == request.epoch &&
                    data.stats_window_active.load(std::memory_order_relaxed)) {
                    data.stats_window_active.store(false, std::memory_order_release);
                    cut = true;
                }
            }
            if (cut) {
                // [0, N): the static table, then the delta table BEGIN grew to cover N.
                const uint32_t nodes = request.nodes;
                std::vector<uint32_t> w_all(nodes);
                uint64_t total = 0;
                bool complete = true;
                for (uint32_t i = 0; i < nodes; ++i) {
                    const std::atomic<uint32_t>* counter = stats_window_counter(data, i);
                    if (counter == nullptr) { complete = false; break; }
                    w_all[i] = counter->load(std::memory_order_relaxed);
                    total += w_all[i];
                }
                if (complete && stats_window_capture_owned(data, request.sections, w_all, snapshot)) {
                    ack.total_w = total;
                    ack.status = 0;
                }
            }
        }
    } catch (...) { ack.status = 1; }
    if (ack.status) snapshot = StatsWindowCapturedSnapshot{};
    snapshot.ack = ack;
    return snapshot;
}

static std::vector<char> snapshot_failure(ann::StatsWindowAck ack) {
    // Recognizably typed failure; never a successful 64-byte ACK.
    ack.msg_type = ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE;
    ack.status = 1;
    WindowSnapshotHeader header; header.ack = ack;
    std::vector<char> response(sizeof(header));
    std::memcpy(response.data(), &header, sizeof(header));
    return response;
}

std::vector<char> stats_window_encode_snapshot(const StatsWindowCapturedSnapshot& snapshot) {
    // Allocate the small typed failure before the larger payload/hash allocation.
    auto failure = snapshot_failure(snapshot.ack);
    if (snapshot.ack.status) return failure;
    try {
        auto response = window_snapshot_encode(snapshot.ack, snapshot.owners, snapshot.nodes, snapshot.sections,
                                               snapshot.owned_digest, snapshot.non_owned_w, snapshot.w,
                                               snapshot.hist, snapshot.nbwork);
        if (!response.empty()) return response;
    } catch (...) {}
    return failure;
}

std::vector<char> stats_window_snapshot_response(ServerData& data,
    const WindowSnapshotRequest& request, ann::StatsWindowAck ack) {
    return stats_window_encode_snapshot(stats_window_capture_snapshot(data, request, ack));
}

StatsWindowSnapshotTask::~StatsWindowSnapshotTask() {
    if (worker_.joinable()) worker_.join();
}

bool StatsWindowSnapshotTask::start(std::string identity, StatsWindowCapturedSnapshot snapshot,
                                   int wake_fd, Encoder encoder) {
    if (pending_) return false;
    reply_ = {};
    reply_.identity = std::move(identity);
    reply_.response = snapshot_failure(snapshot.ack);
    ready_.store(false, std::memory_order_relaxed);
    pending_ = true;
    try {
        worker_ = std::thread([this, snapshot = std::move(snapshot), wake_fd, encoder]() mutable {
            try {
                auto response = encoder(snapshot);
                if (!response.empty()) reply_.response = std::move(response);
            } catch (...) {} // Preserve preallocated, identity/epoch-exact failure.
            snapshot = {}; // Release large immutable inputs before publishing ready.
            ready_.store(true, std::memory_order_release);
            if (wake_fd >= 0) {
                const char byte = 1;
                while (::write(wake_fd, &byte, 1) < 0 && errno == EINTR) {}
            }
        });
    } catch (...) {
        // Thread creation failed: I/O loop consumes the same typed error slot.
        ready_.store(true, std::memory_order_release);
    }
    return true;
}

bool StatsWindowSnapshotTask::finish(StatsWindowSnapshotReply& reply, bool wait) {
    if (!pending_ || (!wait && !ready())) return false;
    if (worker_.joinable()) worker_.join();
    reply = std::move(reply_);
    pending_ = false;
    ready_.store(false, std::memory_order_relaxed);
    return true;
}

WindowStopAck stats_window_stop_local(ServerData& data,
    const WindowStopRequest& request, uint32_t rank) {
    std::lock_guard<std::mutex> lock(data.stats_window_mutex);
    WindowStopAck ack; ack.rank = rank; ack.requested_epoch = request.epoch;
    ack.actual_epoch = data.stats_window_epoch.load(std::memory_order_relaxed);
    ack.active = data.stats_window_active.load(std::memory_order_relaxed);
    // A partially delivered BEGIN can leave a previous epoch on a rank.
    // No future epoch is accepted; the receipt preserves the actual epoch.
    if (request.msg_type != ann::MessageType::STATS_WINDOW_STOP_REQUEST ||
        request.version != 1 || request.reserved || !window_owners_valid(data.n_servers) ||
        request.ranks != data.n_servers || rank >= data.n_servers ||
        ack.actual_epoch > request.epoch) return ack;
    data.stats_window_active.store(false, std::memory_order_release);
    ack.active = 0; ack.status = 0;
    return ack;
}

void stats_window_count_work(ServerData& data,
    const std::vector<std::pair<uint32_t, uint32_t>>& spans, const std::vector<uint8_t>& kinds,
    const std::vector<uint8_t>& reranked) {
    const uint32_t window_nodes = data.stats_window_nodes.load(std::memory_order_acquire);
    const size_t touched = kinds.size();
    for (size_t span = 0; span < spans.size(); ++span) {
        const uint32_t source = spans[span].first;
        const size_t begin_off = std::min<size_t>(spans[span].second, touched);
        const size_t end_off = std::min<size_t>((span + 1 < spans.size()) ? spans[span + 1].second : touched, touched);
        uint32_t work = span < reranked.size() && reranked[span] ? 1u : 0u;
        for (size_t i = begin_off; i < end_off; ++i) work += kinds[i] != kScoreMissing ? 1u : 0u;
        if (source < window_nodes && work) {
            // Null-checked: a delta chunk may be absent while a BEGIN replaces the window.
            if (std::atomic<uint32_t>* counter = stats_window_counter(data, source))
                counter->fetch_add(work, std::memory_order_relaxed);
        }
    }
}

}  // namespace tide
