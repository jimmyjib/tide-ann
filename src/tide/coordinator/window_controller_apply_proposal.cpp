#include "window_controller_runtime.h"
#include "window_controller_transitions.h"
#include "coordinator_types.h"
#include "membership_migration_coordinator.h"
#include "membership_migration_plan.h"

#include <filesystem>
#include <fstream>

namespace tide {
using namespace window_control_detail;

// Planning has finished. Admission, the plan file and the fence all stay
// serialized here on the pipeline owner; PREPARE is separately drain-gated.
void window_controller_apply_proposal(WindowControllerRuntime& runtime, std::vector<WindowBurstAction> actions,
                                      const std::string& propose_fields, Clock::time_point now) {
    const size_t proposed_count = actions.size();
    if (actions.empty()) {
        journal_event(runtime, "no_positive_action", propose_fields);
        enter_cooldown(runtime, now);
        return;
    }
    uint64_t kept_bytes = 0;
    if (!window_history_limit_actions(runtime, actions, coordinator_membership_ownership_epoch(), now,
                                      propose_fields, kept_bytes)) {
        if (!runtime.stopped_on_error) enter_cooldown(runtime, now);
        return;
    }
    const std::string burst_fields = propose_fields + ",\"proposed_actions\":" + std::to_string(proposed_count) +
        ",\"capped_actions\":" + std::to_string(actions.size());
    std::vector<char> plan_bytes;
    std::string error;
    if (!window_controller_skeleton_plan(actions, coordinator_membership_ownership_epoch(),
                                         runtime.config.ranks, &plan_bytes, &error)) {
        journal_event(runtime, "plan_failed", burst_fields + ",\"error\":" + json_quote(error));
        enter_cooldown(runtime, now);
        return;
    }
    const std::string plan_path = window_dump_dir(runtime) + "/plan.bin";
    {
        // Closed before the check: a small plan stays in the stream buffer until the close,
        // where a write error (a full disk) appears. A failed record is removed.
        std::ofstream plan_file(plan_path, std::ios::binary);
        plan_file.write(plan_bytes.data(), static_cast<std::streamsize>(plan_bytes.size()));
        plan_file.close();
        if (plan_file.fail()) {
            journal_event(runtime, "plan_write_failed", burst_fields + ",\"path\":" + json_quote(plan_path));
            std::error_code remove_error;
            std::filesystem::remove(plan_path, remove_error);
            enter_cooldown(runtime, now);
            return;
        }
    }
    // Fence PREPARE; the trigger window's measurement becomes the verdict baseline.
    // Its recall is the recall guard's reference: live recall over the trigger
    // window's queries, not the verify window's (see WindowRepairConfig::recall_guard).
    runtime.staged_verify_hop_per_query = runtime.measurement.hop_per_query;
    runtime.staged_verify_recall_percent = runtime.measurement.recall_percent;
    runtime.staged_verify_rank_cost = runtime.measurement.rank;
    runtime.pending_plan = std::move(plan_bytes);
    runtime.pending_plan_path = plan_path;
    runtime.pending_actions = std::move(actions);
    runtime.pending_bytes = kept_bytes;
    runtime.fence_started = now;
    runtime.fence_last_heartbeat = now;
    runtime.state = WindowControllerState::FENCE;
    journal_event(runtime, "fence_open", burst_fields);
}
}  // namespace tide
