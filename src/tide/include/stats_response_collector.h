#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "protocol/message_types.h"

namespace tide {

enum class StatsResponseDisposition {
    ACCEPTED,
    STALE_REQUEST,
    INVALID,
    DUPLICATE,
};

// One collector instance represents one GET_STATS round. It is deliberately
// independent of socket ownership: after daemon server threads start, only
// those threads touch ZMQ and this collector consumes their shared queue.
class StatsResponseCollector {
public:
    StatsResponseCollector(size_t server_count, uint64_t request_id)
        : request_id_(request_id), received_(server_count, false),
          snapshots_(server_count) {}

    StatsResponseDisposition accept(size_t server, const char* data, size_t size) {
        if (request_id_ == 0 || server >= snapshots_.size() ||
            size != sizeof(ann::StatsResponseHeader)) {
            return StatsResponseDisposition::INVALID;
        }
        ann::StatsResponseHeader response{};
        std::memcpy(&response, data, sizeof(response));
        if (response.msg_type != ann::MessageType::GET_STATS_RESPONSE ||
            response.partition_id != server) {
            return StatsResponseDisposition::INVALID;
        }
        if (response.request_id != request_id_) {
            return StatsResponseDisposition::STALE_REQUEST;
        }
        if (received_[server]) return StatsResponseDisposition::DUPLICATE;
        received_[server] = true;
        snapshots_[server] = response;
        ++count_;
        return StatsResponseDisposition::ACCEPTED;
    }

    bool complete() const { return count_ == snapshots_.size(); }
    bool received(size_t server) const {
        return server < received_.size() && received_[server];
    }
    size_t count() const { return count_; }
    uint64_t request_id() const { return request_id_; }
    const std::vector<ann::StatsResponseHeader>& snapshots() const {
        return snapshots_;
    }

private:
    uint64_t request_id_;
    std::vector<bool> received_;
    std::vector<ann::StatsResponseHeader> snapshots_;
    size_t count_ = 0;
};

}  // namespace tide
