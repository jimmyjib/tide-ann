// Pipeline -> controller hooks. Only the pipeline thread calls them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tide {

struct ControllerTickInputs {
    // Read only while dispatch is paused. PREPARE starts once all four are zero.
    size_t logical_in_flight = 0;
    int32_t transport_in_flight = 0;
    size_t queued_search = 0;
    size_t queued_responses = 0;
    // Counters of the current search command (recall: scored queries only).
    uint64_t completed_queries = 0;
    double recall_sum = 0.0;
    uint64_t recall_count = 0;
    // The counters restart with every command.
    uint64_t command_id = 0;
};

// True while a controller is active.
bool controller_running();
// False while dispatch is paused for a batch.
bool controller_dispatch_allowed();
// One step of the state machine. Never blocks on the network.
void controller_tick(const ControllerTickInputs& inputs);
// True when dispatch is paused and nothing is in flight (the command must end).
bool controller_dispatch_stalled(const ControllerTickInputs& inputs, std::string* reason);
// True when the reply was controller traffic (the caller then drops it).
bool controller_offer(uint32_t server, const void* data, size_t size);
// Same, moving a snapshot instead of copying it.
bool controller_offer_owned(uint32_t server, std::vector<char>& bytes);

}  // namespace tide
