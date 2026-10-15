// Stats-window requests to every server. The replies come back through controller_offer.
#pragma once

#include <cstdint>

namespace tide {

// BEGIN: zero the counters of ids [0, nodes).
void coordinator_stats_window_send_begin(uint64_t window_epoch, uint32_t nodes);

// SNAPSHOT: close the window. Each server answers with one frame (sections: owner histogram, nbwork).
void coordinator_stats_window_send_harvest(uint64_t window_epoch, uint32_t nodes, uint32_t sections);

}  // namespace tide
