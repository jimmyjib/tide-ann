#pragma once
#include "window_controller_runtime.h"
#include "window_control_worker.h"
#include "window_snapshot_wire.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace tide {
struct WindowControlGeneration {
    std::mutex publish_mutex;
    bool cancelled = false; // protected by publish_mutex
};
struct WindowSnapshotResult {
    uint32_t server = 0;
    size_t bytes = 0;
    bool valid = false, duplicate = false;
    std::string error;
    double work_us = 0;
    std::shared_ptr<const WindowRankFrame> frame;  // the decoded frame (valid only)
    // The frame header (also for an invalid frame, when it was readable).
    WindowSnapshotHeader header{};
};
// The snapshot structure sections of a harvest follow the repair weight:
// window_repair_structure_sections(config.repair) (window_repair_planner.h).
struct WindowObservationLoad {
    bool valid = false;
    WindowTables tables;
    WindowStructure structure;
    // Per-node MOVE bytes derived from the merged owner histogram
    // (window_controller_histogram_row_bytes, the vector bytes included).
    std::vector<uint64_t> row_bytes;
    std::string error;
};
// Frozen copies of everything the pure planner consumes; the owner installs
// the returned state.
struct WindowPlanningInput {
    uint32_t owners = 0;  // K, the controller's ranks
    std::vector<uint32_t> membership;
    std::vector<uint64_t> row_bytes;
    WindowTables tables;
    WindowStructure structure;
    WindowRepairConfig repair_config;
    WindowRepairPhase repair_phase = WindowRepairPhase::IDLE;
    uint32_t repair_window = 0;
    uint64_t window_queries = 0, total_bytes = 0;
    std::vector<double> repair_price, repair_pred;
    std::vector<uint32_t> repair_cooldown;
    // The action history's rejection memory at submission: empty when nothing
    // is cooled down or backed off (the planner is then unrestricted).
    WindowHistoryEligibility eligibility;
};
struct WindowPlanningResult {
    bool valid = false;
    std::vector<WindowBurstAction> actions;
    std::string error;
    double work_us = 0;
    WindowRepairPlan repair;
};
struct WindowNativeControl {
    // Worker is destroyed last, after generation invalidation in destructor.
    WindowControlWorker worker;
    std::shared_ptr<WindowControlGeneration> generation = std::make_shared<WindowControlGeneration>();
    std::vector<std::future<WindowSnapshotResult>> snapshots;
    std::vector<uint8_t> arrived;
    uint32_t arrival_count = 0;
    uint64_t epoch = 0, ownership_epoch = 0, command_id = 0;
    bool endpoints_frozen = false, loaded = false, planning_started = false;
    ControllerTickInputs endpoints;
    std::future<WindowObservationLoad> loading;
    std::future<WindowPlanningResult> planning;
    WindowObservationLoad observation;
    WindowPlanningResult proposal;
    std::shared_ptr<const WindowPlanningInput> planning_input;
    std::chrono::steady_clock::time_point planning_started_at, planning_deadline;
    ~WindowNativeControl();
};
// Worker queue capacity for K ranks: every rank's frame and one duplicate
// each (window_snapshot_offer_owned refuses more), the merge job and the
// planning job.
inline size_t window_control_queue_capacity(uint32_t ranks) { return 2 * size_t(ranks) + 2; }
void window_native_cancel(WindowControllerRuntime& runtime);
void window_native_arm(WindowControllerRuntime& runtime);
bool window_snapshot_offer_owned(WindowControllerRuntime& runtime, uint32_t server, std::vector<char> bytes);
void window_native_poll_snapshots(WindowControllerRuntime& runtime);
// Freezes the tick inputs at the first tick after every rank's frame arrived:
// the window's end boundary.
void window_native_freeze_endpoints(WindowControllerRuntime& runtime, const ControllerTickInputs& inputs);
// Merges the window's validated frames (runtime.snapshot_frames) in memory on
// the worker (window_merge_frames); true once merged.
bool window_native_load(WindowControllerRuntime& runtime, const std::vector<uint32_t>& membership);
// Submits window_repair_plan on frozen inputs. `window_queries` is the
// window's completed-query count; calibration normalises per query with it.
bool window_native_start_planning(WindowControllerRuntime& runtime, const std::vector<uint32_t>& membership,
    const WindowRankCostSnapshot& rank, uint64_t ownership_epoch, uint64_t command_id,
    uint64_t window_queries, std::string& error);
WindowPlanningResult window_native_plan(const WindowPlanningInput& input);
// The finished plan still applies: same window, ownership epoch, command and history.
bool window_native_plan_applicable(const WindowControllerRuntime& runtime, uint64_t ownership_epoch,
    const ControllerTickInputs& inputs);
} // namespace tide
