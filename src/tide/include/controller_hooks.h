// The calls the serving pipeline makes into the runtime repair controller.
//
// The pipeline loop is the controller's only thread: while a controller is
// running, the loop calls controller_tick() once per iteration, checks
// controller_dispatch_allowed() before it enqueues new queries, and offers
// every server reply to controller_offer() before treating it as a search or
// maintenance reply. Nothing else in the pipeline depends on controller types.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tide {

struct ControllerTickInputs {
    // Read only while the controller holds its commit fence (dispatch paused):
    // the fence opens PREPARE once all four are zero.
    size_t logical_in_flight = 0;
    int32_t transport_in_flight = 0;
    size_t queued_search = 0;
    size_t queued_responses = 0;
    // Always filled. Counters of the current search command: completed
    // queries, and the running sum/count of per-query Recall@K for queries
    // scored against ground truth (zero when the command has none).
    uint64_t completed_queries = 0;
    double recall_sum = 0.0;
    uint64_t recall_count = 0;
    // Sequence number of the running search command. The counters above
    // restart with every command, so a window never spans two commands.
    uint64_t command_id = 0;
};

// True while a controller is active; only then does the pipeline build tick inputs.
bool controller_running();
// False while the controller holds its commit fence: the pipeline must not
// enqueue new work until the fence has committed or given up.
bool controller_dispatch_allowed();
// One step of the controller state machine; never blocks on the network.
void controller_tick(const ControllerTickInputs& inputs);
// True when the controller froze dispatch (an ownership transaction it could
// not confirm: accept_failed, commit_failed or prepare_rollback_failed with a
// failed rollback) and `inputs` show nothing in flight or queued. The serving
// command can then never complete: the pipeline ends it with an error so the
// admin recovery commands (migration-accept, migration-rollback, status, quit)
// can run. A confirmed migration-accept or migration-rollback lifts the freeze
// (window_controller_confirm_ownership) and controller-abort then ends the
// terminal controller. `reason` receives the controller's terminal reason.
bool controller_dispatch_stalled(const ControllerTickInputs& inputs, std::string* reason);
// Offers one server reply. Returns true when it was controller traffic (a
// stats-window ACK or snapshot, including a stale one after stop); the caller
// must then drop it. Search and maintenance replies return false untouched.
bool controller_offer(uint32_t server, const void* data, size_t size);
// Same for a reply the caller owns; a snapshot is moved to the controller's
// decode worker instead of being copied.
bool controller_offer_owned(uint32_t server, std::vector<char>& bytes);

}  // namespace tide
