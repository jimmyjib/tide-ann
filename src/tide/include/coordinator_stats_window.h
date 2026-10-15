// Coordinator fanouts of the stats-window protocol, used by the runtime
// repair controller (window_controller_tick.cpp). Both only enqueue one request
// per server and return: mid-serving, the shared response queue interleaves
// the replies with live search responses, and the pipeline hands them back
// through controller_offer (controller_hooks.h).
#pragma once

#include <cstdint>

namespace tide {

// Opens window `window_epoch` over node ids [0, nodes) on every server: BEGIN
// zeroes the per-node counters; each server answers with a STATS_WINDOW_BEGIN_ACK
// (a server whose table holds another node count refuses).
void coordinator_stats_window_send_begin(uint64_t window_epoch, uint32_t nodes);

// Closes the window: each server answers with one snapshot wire v3 frame (the
// work of its owned ids plus the requested structure sections: the owner
// histogram, and the neighbour-work section under weight WNB only;
// window_repair_structure_sections).
void coordinator_stats_window_send_harvest(uint64_t window_epoch, uint32_t nodes, uint32_t sections);

}  // namespace tide
