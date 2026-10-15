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
    std::shared_ptr<const WindowRankFrame> frame;  // valid frames only
    // frame header, also for an invalid frame when it was readable
    WindowSnapshotHeader header{};
};
// sections follow the weight: window_repair_structure_sections(config.repair)
struct WindowObservationLoad {
    bool valid = false;
    WindowTables tables;
    WindowStructure structure;
    // per-node MOVE bytes from the merged owner histogram, vector bytes included
    std::vector<uint64_t> row_bytes;
    std::string error;
};
// copies of the planner inputs (the owner installs the returned state)
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
    // rejection memory at submission, empty when nothing is restricted
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
    // destroyed last
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
// queue capacity: two frames per rank, the merge job and the planning job
inline size_t window_control_queue_capacity(uint32_t ranks) { return 2 * size_t(ranks) + 2; }
void window_native_cancel(WindowControllerRuntime& runtime);
void window_native_arm(WindowControllerRuntime& runtime);
bool window_snapshot_offer_owned(WindowControllerRuntime& runtime, uint32_t server, std::vector<char> bytes);
void window_native_poll_snapshots(WindowControllerRuntime& runtime);
// fixes the tick inputs once every rank's frame has arrived (the window's end)
void window_native_freeze_endpoints(WindowControllerRuntime& runtime, const ControllerTickInputs& inputs);
// merges the frames in memory on the worker, true once merged
bool window_native_load(WindowControllerRuntime& runtime, const std::vector<uint32_t>& membership);
// window_queries normalizes the calibration per query
bool window_native_start_planning(WindowControllerRuntime& runtime, const std::vector<uint32_t>& membership,
    const WindowRankCostSnapshot& rank, uint64_t ownership_epoch, uint64_t command_id,
    uint64_t window_queries, std::string& error);
WindowPlanningResult window_native_plan(const WindowPlanningInput& input);
// true while the plan still applies: same window, ownership epoch, command and history
bool window_native_plan_applicable(const WindowControllerRuntime& runtime, uint64_t ownership_epoch,
    const ControllerTickInputs& inputs);
} // namespace tide
