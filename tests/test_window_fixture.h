// Snapshot frames for the controller tests (nbwork zero).
#pragma once

#include <cstdint>
#include <vector>

#include "protocol/message_types.h"
#include "window_snapshot_wire.h"

namespace window_test {

// Cumulative search counters a server reports in every stats-window ACK.
struct RankCounters {
    uint64_t requests = 0, distances = 0, distance_us = 0, total_us = 0;
};

inline ann::StatsWindowAck ack(ann::MessageType type, uint32_t rank, uint64_t epoch,
                               const RankCounters& c, uint64_t total_w = 0) {
    ann::StatsWindowAck a{};
    a.msg_type = type; a.status = 0; a.server_rank = rank; a.window_epoch = epoch;
    a.search_request_count = c.requests; a.search_distance_count = c.distances;
    a.search_distance_us = c.distance_us; a.search_total_us = c.total_us; a.total_w = total_w;
    return a;
}

// One rank's frame: `w_all` holds the work of every node (the frame keeps the
// owned ones), every owned id gets the histogram row `hist_row` (K entries,
// default one neighbor per owner), nbwork is zero.
inline std::vector<char> frame(uint32_t rank, uint64_t epoch, const RankCounters& c,
                               const std::vector<uint32_t>& w_all, const std::vector<uint32_t>& membership,
                               std::vector<uint16_t> hist_row = {},
                               uint32_t sections = tide::kWindowStructureOwnerHist | tide::kWindowStructureNbWork,
                               uint32_t owners = 4) {
    if (hist_row.empty()) hist_row.assign(owners, 1);
    uint64_t total = 0, digest = tide::kWindowOwnedDigestSeed;
    std::vector<uint32_t> w;
    std::vector<uint16_t> hist;
    for (uint32_t v = 0; v < membership.size(); ++v) {
        if (membership[v] != rank) continue;
        digest = tide::window_owned_digest_add(digest, v);
        w.push_back(w_all[v]);
        total += w_all[v];
        if (sections & tide::kWindowStructureOwnerHist) hist.insert(hist.end(), hist_row.begin(), hist_row.end());
    }
    std::vector<uint64_t> nbwork;
    if (sections & tide::kWindowStructureNbWork) nbwork.assign(membership.size(), 0);
    return tide::window_snapshot_encode(ack(ann::MessageType::STATS_WINDOW_HARVEST_ACK, rank, epoch, c, total),
        owners, static_cast<uint32_t>(membership.size()), sections, digest, 0, w, hist, nbwork);
}

}  // namespace window_test
