#include "window_native_control.h"
#include <mutex>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace tide {
WindowNativeControl::~WindowNativeControl() {
    std::lock_guard<std::mutex> lock(generation->publish_mutex);
    generation->cancelled = true;
}
void window_native_cancel(WindowControllerRuntime& r) {
    if (!r.native_control) return;
    auto& work = *r.native_control;
    { std::lock_guard<std::mutex> lock(work.generation->publish_mutex); work.generation->cancelled = true; }
    work.snapshots.clear(); work.loading = {}; work.planning = {}; work.planning_input.reset();
    work.endpoints_frozen = false; work.loaded = false; work.planning_started = false;
}
void window_native_arm(WindowControllerRuntime& r) {
    if (!r.native_control) r.native_control = std::make_shared<WindowNativeControl>();
    window_native_cancel(r);
    auto& work = *r.native_control;
    work.worker.set_capacity(window_control_queue_capacity(r.config.ranks));
    work.generation = std::make_shared<WindowControlGeneration>();
    work.epoch = r.window_epoch; work.arrived.assign(r.config.ranks, 0); work.arrival_count = 0;
    work.observation = {}; work.proposal = {};
}
void window_native_freeze_endpoints(WindowControllerRuntime& r, const ControllerTickInputs& inputs) {
    if (!r.native_control || r.exchange != WindowExchangeKind::HARVEST) return;
    auto& work = *r.native_control;
    if (!work.endpoints_frozen && work.epoch == r.window_epoch && work.arrival_count == r.config.ranks) {
        work.endpoints = inputs; work.endpoints_frozen = true;
        // The response queue has been drained for this owner iteration. This
        // is intentionally not claimed to be the network receive instant.
        r.journal << std::setprecision(17) << "{\"event\":\"snapshot_owner_boundary_frozen\",\"window_epoch\":"
            << r.window_epoch << ",\"cycle\":" << r.cycles << ",\"mono_ns\":" << window_journal_now_ns()
            << ",\"boundary\":\"first-owner-tick-after-all-rank-frame-arrivals\""
            << ",\"completed_queries\":" << inputs.completed_queries << ",\"recall_sum\":" << inputs.recall_sum
            << ",\"recall_count\":" << inputs.recall_count << ",\"command_id\":" << inputs.command_id << "}\n";
        r.journal.flush();
    }
}
bool window_native_load(WindowControllerRuntime& r, const std::vector<uint32_t>& membership) {
    auto& work = *r.native_control;
    if (work.loaded) return true;
    if (!work.loading.valid()) {
        // In-memory merge of the validated frames; no file round trip.
        work.loading = work.worker.submit([membership, frames=r.snapshot_frames, ranks=r.config.ranks,
                expected=r.harvest_per_rank_w, sections=window_repair_structure_sections(r.config.repair),
                vector_bytes=r.config.vector_bytes_per_row] {
            WindowObservationLoad result;
            result.valid = window_merge_frames(frames, membership, ranks, sections,
                                               &result.tables, &result.structure, &result.error);
            if (result.valid && result.tables.per_rank_work != expected) {
                result.valid = false; result.error = "snapshot and HARVEST work differ";
                result.structure.valid = false; result.structure.error = result.error;
            }
            if (result.valid)
                result.row_bytes = window_controller_histogram_row_bytes(result.structure.owner_hist, ranks,
                                                                         vector_bytes);
            if (!result.valid && result.structure.error.empty())
                result.structure.error = "observation invalid: " + result.error;
            return result;
        });
        if (!work.loading.valid()) { work.observation.error = "control queue full"; work.loaded = true; }
        return false;
    }
    if (!window_control_ready(work.loading)) return false;
    try { work.observation = work.loading.get(); }
    catch (const std::exception& e) { work.observation.error = e.what(); }
    work.loaded = true;
    const auto& structure = work.observation.structure;
    std::string quoted;
    for (const char c : structure.error) {
        if (c == '"' || c == '\\') quoted += '\\';
        quoted += c;
    }
    r.journal << "{\"event\":\"" << (structure.valid ? "structure_loaded" : "structure_unavailable")
        << "\",\"window_epoch\":" << r.window_epoch << ",\"cycle\":" << r.cycles
        << ",\"mono_ns\":" << window_journal_now_ns()
        << ",\"structure_sections\":" << window_repair_structure_sections(r.config.repair)
        << ",\"weight\":\"" << window_repair_weight_name(r.config.repair.weight) << '"'
        << ",\"owners\":" << r.config.ranks << ",\"merge\":\"in-memory\""
        << ",\"valid\":" << (structure.valid ? "true" : "false");
    if (structure.valid) {
        r.journal << ",\"cut_edges\":" << structure.cut_edges;
        // Only a WNB harvest carries the nbwork section (weight W: absent, not zero).
        if (structure.sections & kWindowStructureNbWork) r.journal << ",\"nb_work_total\":" << structure.nb_work_total;
    }
    else r.journal << ",\"error\":\"" << quoted << "\"";
    r.journal << "}\n";
    r.journal.flush();
    return true;
}
bool window_native_start_planning(WindowControllerRuntime& r, const std::vector<uint32_t>& membership,
    const WindowRankCostSnapshot& rank, uint64_t ownership_epoch, uint64_t command_id,
    uint64_t window_queries, std::string& error) {
    if (!r.native_control || !r.action_history || r.action_history->expected_epoch != ownership_epoch ||
        window_controller_has_unresolved_action(r)) {
        error = "planning requires a resolved history at the current ownership epoch"; return false;
    }
    auto& work = *r.native_control;
    const auto& structure = work.observation.structure;
    // Exactly the weight's sections: the owner histogram, plus nbwork under WNB only.
    const uint32_t sections = window_repair_structure_sections(r.config.repair);
    const bool want_nbwork = sections & kWindowStructureNbWork;
    if (!structure.valid || structure.sections != sections ||
        structure.owners != r.config.ranks ||
        structure.owner_hist.size() != membership.size() * r.config.ranks ||
        structure.nb_work.size() != (want_nbwork ? membership.size() : 0)) {
        error = "repair structure sections unavailable: " + structure.error; return false;
    }
    if (!window_queries) { error = "repair requires completed window queries"; return false; }
    // The window's ids are the phase's (no insert runs during a controller phase).
    if (membership.size() != r.nodes || r.row_bytes.size() != r.nodes || !r.total_bytes) {
        error = "planning requires the phase's node count and bound row bytes"; return false;
    }
    // The pure planner consumes frozen copies; the owner installs its result.
    window_controller_repair_freeze_prices(r, rank);
    if (!std::isfinite(r.repair_state.first_peak_us))
        r.repair_state.first_peak_us = window_controller_repair_peak_us(r, rank);
    WindowPlanningInput job;
    job.owners = r.config.ranks;
    job.structure = std::move(work.observation.structure);
    job.repair_config = r.config.repair; job.repair_phase = r.repair_state.phase;
    job.repair_window = r.repair_state.windows + 1;
    job.window_queries = window_queries; job.total_bytes = r.total_bytes;
    job.repair_price = r.repair_state.price; job.repair_pred = r.repair_state.pred;
    job.repair_cooldown = r.repair_state.cooldown_until;
    job.membership = membership; job.row_bytes = r.row_bytes;
    job.tables = std::move(work.observation.tables);
    const auto now = std::chrono::steady_clock::now();
    job.eligibility = window_history_eligibility(r, membership);
    work.ownership_epoch = ownership_epoch; work.command_id = command_id;
    work.planning_started_at = now; work.planning_deadline = now + std::chrono::seconds(30);
    work.planning_input = std::make_shared<const WindowPlanningInput>(std::move(job));
    work.planning = work.worker.submit([input=work.planning_input] { return window_native_plan(*input); });
    if (!work.planning.valid()) { error = "control planning queue full"; return false; }
    work.planning_started = true; return true;
}
bool window_native_plan_applicable(const WindowControllerRuntime& r, uint64_t ownership_epoch,
    const ControllerTickInputs& inputs) {
    if (!r.native_control) return false;
    const auto& work = *r.native_control;
    return work.planning_started && work.planning_input && work.epoch == r.window_epoch &&
        ownership_epoch == work.ownership_epoch && r.action_history &&
        r.action_history->expected_epoch == ownership_epoch && !r.pending_bytes &&
        inputs.command_id == work.command_id && !r.proposals_disabled && !r.stats_stop_pending &&
        !r.stopped_on_error && !window_controller_has_unresolved_action(r) &&
        (!r.config.proposal_query_limit || inputs.completed_queries < r.config.proposal_query_limit);
}
WindowPlanningResult window_native_plan(const WindowPlanningInput& in) {
    const auto start = std::chrono::steady_clock::now();
    WindowPlanningResult out;
    // Local-loop repair: calibrate, refine, balance on the frozen inputs.
    out.repair = window_repair_plan(in.repair_config, in.owners, in.repair_phase, in.repair_window, in.window_queries,
        in.total_bytes, in.membership, in.row_bytes, in.tables.work, in.structure.nb_work,
        in.structure.owner_hist, in.repair_price, in.repair_pred, in.repair_cooldown,
        in.eligibility.move_targets.empty() ? nullptr : &in.eligibility.move_targets);
    out.valid = out.repair.valid;
    out.error = out.repair.error;
    out.actions = out.repair.actions;
    out.work_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    return out;
}
} // namespace tide
