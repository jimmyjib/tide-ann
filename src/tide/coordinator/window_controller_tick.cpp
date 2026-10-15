#include "window_controller_runtime.h"
#include "window_native_control.h"
#include "window_controller_transitions.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include "coordinator_stats_window.h"
#include "coordinator_types.h"
#include "membership_migration_coordinator.h"
#include "membership_migration_plan.h"
namespace tide {
using namespace window_control_detail;

namespace {
template <class T> std::string json_array(const std::vector<T>& values) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    return out.str() + "]";
}

// Payload of the batch's MOVE: bytes, deadlines and timings (membership_migration_coordinator.h).
std::string fence_payload_fields() {
    const auto& p = coordinator_membership_migration_payload();
    return ",\"fence_payload_bytes\":" + std::to_string(p.total_bytes()) +
        ",\"refresh_request_bytes\":" + std::to_string(p.refresh_request_bytes) +
        ",\"refresh_response_bytes\":" + std::to_string(p.refresh_response_bytes) +
        ",\"prepare_request_bytes\":" + std::to_string(p.prepare_request_bytes) +
        ",\"vector_bytes\":" + std::to_string(p.vector_bytes) +
        ",\"prepare_vector_bytes\":" + std::to_string(p.prepare_vector_bytes) +
        migration_payload_by_server_fields(p) +
        ",\"refresh_deadline_s\":" + exact_number(p.refresh_deadline_s) +
        ",\"exchange_deadline_s\":" + exact_number(p.exchange_deadline_s) +
        ",\"refresh_s\":" + exact_number(p.refresh_s) + ",\"prepare_s\":" + exact_number(p.prepare_s);
}

// Per-rank costs of the window from the BEGIN/HARVEST ACK counters. False if unmeasurable.
bool measure_window(WindowControllerRuntime& r, const ControllerTickInputs& inputs) {
    WindowMeasurement m;
    const auto& b = r.window_begin_sums;
    const auto& e = r.exchange_counter_sums;
    m.per_rank_w = r.harvest_per_rank_w;
    m.total_w = r.harvest_total_w;
    m.skew = window_controller_skew(m.per_rank_w);
    m.requests = window_counter_delta(e.requests, b.requests);
    m.distances = window_counter_delta(e.distances, b.distances);
    m.distance_us = window_counter_delta(e.distance_us, b.distance_us);
    m.total_us = window_counter_delta(e.total_us, b.total_us);
    m.queries = inputs.command_id == r.window_begin_command
        ? window_counter_delta(inputs.completed_queries, r.window_begin_completed) : 0;
    m.hop_per_query = m.queries ? double(m.requests) / double(m.queries) : 0.0;
    m.recall_percent = window_controller_window_recall(r, inputs);
    const bool valid = window_controller_measurement_valid(r, inputs) &&
        window_controller_rank_snapshot(r, m.queries, &m.rank);
    std::ostringstream fields;
    fields << "\"skew\":" << exact_number(m.skew) << ",\"total_w\":" << m.total_w
           << ",\"per_rank_raw_w\":" << json_array(m.per_rank_w)
           << ",\"window_queries\":" << m.queries << ",\"min_window_queries\":" << r.config.min_window_queries
           << ",\"window_requests\":" << m.requests << ",\"window_distances\":" << m.distances
           << ",\"window_distance_us\":" << m.distance_us << ",\"window_total_us\":" << m.total_us
           << ",\"completed_begin\":" << r.window_begin_completed << ",\"completed_end\":" << inputs.completed_queries
           << ",\"command_id\":" << inputs.command_id
           << ",\"hop_per_query\":" << exact_number(m.hop_per_query)
           << ",\"recall_percent\":" << exact_number(m.recall_percent)
           << ",\"window_recall_count\":" << window_counter_delta(inputs.recall_count, r.window_begin_recall_count)
           << ",\"rank_counter_interval\":\"BEGIN_ACK_to_HARVEST_ACK\",\"rank_counter_deltas\":[";
    for (size_t rank = 0; rank < r.exchange_rank_counters.size() && rank < r.window_begin_rank_counters.size(); ++rank) {
        const auto& bb = r.window_begin_rank_counters[rank];
        const auto& ee = r.exchange_rank_counters[rank];
        fields << (rank ? "," : "") << "{\"rank\":" << rank
               << ",\"D\":" << window_counter_delta(ee.distances, bb.distances)
               << ",\"R\":" << window_counter_delta(ee.requests, bb.requests)
               << ",\"Td_us\":" << window_counter_delta(ee.distance_us, bb.distance_us)
               << ",\"Ttot_us\":" << window_counter_delta(ee.total_us, bb.total_us)
               << ",\"A\":" << window_counter_delta(ee.adc_tables, bb.adc_tables) << '}';
    }
    fields << "],\"rank_time_context\":" << window_rank_time_json(m.rank.rank_time);
    m.fields = fields.str();
    r.measurement = std::move(m);
    return valid;
}

// Installs the planner's weights and cooldowns and returns the proposal journal fields.
std::string repair_install_plan(WindowControllerRuntime& runtime, const WindowRepairPlan& plan) {
    // rejection memory ages by planning windows
    window_history_count_planning_window(runtime);
    auto& state = runtime.repair_state;
    const auto& config = runtime.config.repair;
    state.pred = plan.pred;
    state.cooldown_until = plan.cooldown_until;
    const uint32_t window = state.windows + 1;
    const std::string phase_before = window_repair_phase_name(state.phase);
    journal_event(runtime, "repair_calibrated", "\"window\":" + std::to_string(window) +
        ",\"phase\":\"" + phase_before + "\"," + window_repair_calibration_json(plan.calibration, state.price));
    const auto transition = window_repair_account_window(config, state, !plan.actions.empty(), plan.calibration.skew);
    uint64_t planned_bytes = 0;
    for (const auto& action : plan.actions) planned_bytes += action.adjacency_bytes;
    const std::string fields = ",\"repair_window\":" + std::to_string(window) +
        ",\"repair_phase\":\"" + window_repair_phase_name(state.phase) + "\"" +
        ",\"repair_planned\":" + (plan.planned ? "true" : "false") + window_repair_stats_json(plan.stats) +
        ",\"planned_actions\":" + std::to_string(plan.actions.size()) +
        ",\"planned_bytes\":" + std::to_string(planned_bytes) +
        ",\"cumulative_bytes\":" + std::to_string(state.cumulative_bytes) +
        ",\"quiet_windows\":" + std::to_string(state.quiet_windows) +
        ",\"bursts\":" + std::to_string(state.bursts) + ",\"reverts\":" + std::to_string(state.reverts);
    if (transition.reactivated)
        journal_event(runtime, "repair_reactivated", "\"window\":" + std::to_string(window) +
            ",\"skew\":" + exact_number(plan.calibration.skew));
    if (plan.planned) journal_event(runtime, "repair_burst_planned", fields.substr(1));
    else journal_event(runtime, "repair_calibration_only", fields.substr(1));
    if (transition.converged)
        journal_event(runtime, "repair_converged", "\"windows\":" + std::to_string(state.windows) +
            ",\"cumulative_bytes\":" + std::to_string(state.cumulative_bytes) +
            ",\"bursts\":" + std::to_string(state.bursts) + ",\"reverts\":" + std::to_string(state.reverts) +
            ",\"skew\":" + exact_number(plan.calibration.skew));
    return fields;
}

// Keep-or-revert check: the verify window's costs, hops and recall against the trigger window's.
bool repair_verify(WindowControllerRuntime& runtime, std::string* fields) {
    auto& state = runtime.repair_state;
    const auto& config = runtime.config.repair;
    const auto& measured = runtime.measurement;
    WindowRepairVerdict repair;
    std::vector<double> baseline_cost, measured_cost;
    const bool priced = window_controller_rank_costs(runtime.verify_rank_cost, runtime.verify_rank_cost, &baseline_cost) &&
        window_controller_rank_costs(runtime.verify_rank_cost, measured.rank, &measured_cost);
    if (priced)
        repair = window_repair_verdict(config, runtime.config.ranks, baseline_cost, measured_cost,
                                       runtime.verify_hop_per_query,
                                       measured.hop_per_query, measured.recall_percent, runtime.verify_recall_percent);
    if (!repair.accepted) ++state.reverts;
    const double peak_us = window_controller_repair_peak_us(runtime, measured.rank);
    const bool tripped = window_repair_guard_tripped(config, state, peak_us);
    const std::string verdict_fields = "\"accepted\":" + std::string(repair.accepted ? "true" : "false") +
        ",\"priced\":" + (priced ? "true" : "false") +
        ",\"peak_ratio\":" + exact_number(repair.peak_ratio) + ",\"total_ratio\":" + exact_number(repair.total_ratio) +
        ",\"hop_ratio\":" + exact_number(repair.hop_ratio) +
        ",\"recall_guard\":" + (config.recall_guard ? exact_number(config.recall_guard_pp) : std::string("\"off\"")) +
        ",\"recall\":" + exact_number(measured.recall_percent) +
        ",\"reference\":" + exact_number(runtime.verify_recall_percent) +
        ",\"slack\":" + exact_number(config.slack) + ",\"hop_slack\":" + exact_number(config.hop_slack) +
        ",\"verified_peak_us\":" + exact_number(peak_us) + ",\"first_peak_us\":" + exact_number(state.first_peak_us);
    journal_event(runtime, "repair_verdict", verdict_fields);
    if (tripped && state.phase != WindowRepairPhase::ABORTED) {
        state.phase = WindowRepairPhase::ABORTED;
        journal_event(runtime, "repair_aborted", "\"reason\":\"cumulative-regression\",\"first_peak_us\":" +
            exact_number(state.first_peak_us) + ",\"guard\":" + exact_number(config.guard) +
            ",\"converge_windows\":" + std::to_string(config.converge_windows) +
            ",\"verified_peak_us\":" + exact_number(peak_us));
    }
    if (fields) *fields = ",\"repair_verdict\":{" + verdict_fields + '}';
    return repair.accepted;
}

bool accept_burst(WindowControllerRuntime& runtime, const std::string& fields, Clock::time_point now) {
    if (!window_history_resolved(runtime, true, coordinator_membership_ownership_epoch(), now)) {
        stop_on_error(runtime, "history_accept_failed", "");
        return false;
    }
    runtime.verify_pending = false;
    runtime.verify_invalid_windows = 0;
    runtime.last_committed_actions.clear();
    journal_event(runtime, "verify_accepted", fields);
    return true;
}

// Reverts a batch with its inverse MOVE. The history reserved its bytes when the batch was allowed.
void revert_burst(WindowControllerRuntime& runtime, const std::string& fields, Clock::time_point now) {
    std::vector<WindowBurstAction> inverse = window_controller_inverse_actions(runtime.last_committed_actions);
    runtime.reverting = true;
    uint64_t inverse_bytes = 0;
    for (const WindowBurstAction& action : inverse) inverse_bytes += action.adjacency_bytes;
    std::vector<char> plan_bytes;
    std::string error;
    if (inverse.empty() || !window_controller_skeleton_plan(inverse, coordinator_membership_ownership_epoch(),
                                                            runtime.config.ranks, &plan_bytes, &error)) {
        stop_on_error(runtime, "revert_plan_failed", fields.substr(1) + ",\"error\":" + json_quote(error));
        return;
    }
    // Revert from the in-memory plan (revert-plan.bin is only a record).
    const std::string dump_dir = window_dump_dir(runtime);
    std::error_code dir_error;
    std::filesystem::create_directories(dump_dir, dir_error);
    std::string plan_path = dump_dir + "/revert-plan.bin";
    {
        std::ofstream plan_file(plan_path, std::ios::binary);
        plan_file.write(plan_bytes.data(), static_cast<std::streamsize>(plan_bytes.size()));
        plan_file.close();
        if (plan_file.fail()) {
            journal_event(runtime, "revert_plan_record_failed",
                          "\"path\":" + json_quote(plan_path) +
                          (dir_error ? ",\"error\":" + json_quote(dir_error.message()) : std::string()) +
                          ",\"plan_bytes\":" + std::to_string(plan_bytes.size()) + ",\"plan_in_memory\":true");
            std::error_code remove_error;
            std::filesystem::remove(plan_path, remove_error);
            plan_path.clear();
        }
    }
    journal_event(runtime, "verify_rejected", fields.substr(1));
    runtime.pending_actions = std::move(inverse);
    runtime.pending_bytes = inverse_bytes;
    runtime.pending_plan = std::move(plan_bytes);
    runtime.pending_plan_path = plan_path;
    runtime.fence_started = now;
    runtime.fence_last_heartbeat = now;
    runtime.state = WindowControllerState::FENCE;
    journal_event(runtime, "revert_fence_open", "\"actions\":" + std::to_string(runtime.pending_actions.size()));
}

void defer(WindowControllerRuntime& runtime, const char* event, const std::string& fields, Clock::time_point now) {
    journal_event(runtime, event, fields);
    enter_cooldown(runtime, now);
}

// verify_invalid_windows and verify_invalid_limit while a batch waits for its check
std::string verify_invalid_fields(const WindowControllerRuntime& runtime) {
    if (!runtime.verify_pending) return "";
    return ",\"verify_invalid_windows\":" + std::to_string(runtime.verify_invalid_windows) +
        ",\"verify_invalid_limit\":" + std::to_string(runtime.config.repair.verify_invalid_limit);
}

// A window that cannot check the pending moves counts toward verify_invalid_limit. At the limit
// they are reverted. Returns true when the revert started.
bool verify_window_lost(WindowControllerRuntime& runtime, const char* reason, Clock::time_point now) {
    if (!runtime.verify_pending || runtime.reverting) return false;
    ++runtime.verify_invalid_windows;
    const auto& config = runtime.config.repair;
    if (runtime.verify_invalid_windows < config.verify_invalid_limit) return false;
    window_native_cancel(runtime);
    runtime.exchange = WindowExchangeKind::NONE;
    ++runtime.repair_state.reverts;
    const std::string verdict_fields = std::string("\"accepted\":false,\"priced\":false") +
        ",\"reason\":\"verify-invalid-limit\",\"last_invalid\":" + json_quote(reason) +
        ",\"verify_invalid_windows\":" + std::to_string(runtime.verify_invalid_windows) +
        ",\"verify_invalid_limit\":" + std::to_string(config.verify_invalid_limit) +
        ",\"peak_ratio\":null,\"total_ratio\":null,\"hop_ratio\":null" +
        ",\"recall_guard\":" + (config.recall_guard ? exact_number(config.recall_guard_pp) : std::string("\"off\"")) +
        ",\"recall\":null,\"reference\":" + exact_number(runtime.verify_recall_percent);
    journal_event(runtime, "repair_verdict", verdict_fields);
    revert_burst(runtime, ",\"repair_verdict\":{" + verdict_fields + '}', now);
    return true;
}

// Check a pending batch, then plan the next one.
void harvested(WindowControllerRuntime& runtime, const ControllerTickInputs& inputs, Clock::time_point now) {
    auto& work = *runtime.native_control;
    if (!measure_window(runtime, inputs)) {
        if (verify_window_lost(runtime, "measurement-invalid", now)) return;
        // pending actions are kept and the next window measures again
        defer(runtime, runtime.verify_pending ? "verify_deferred_measurement" : "measurement_unavailable",
              runtime.measurement.fields + verify_invalid_fields(runtime), now);
        return;
    }
    journal_event(runtime, "harvest_measured", runtime.measurement.fields);
    // The check needs only the measurement, so it runs before the planning inputs are checked.
    if (runtime.verify_pending) {
        runtime.verify_invalid_windows = 0;
        std::string verdict;
        if (!repair_verify(runtime, &verdict)) {
            revert_burst(runtime, verdict, now);
            return;
        }
        if (!accept_burst(runtime, verdict.substr(1), now)) return;
    }
    if (!work.observation.valid) {
        // the next window observes again
        defer(runtime, "observation_unavailable",
              runtime.measurement.fields + ",\"error\":" + json_quote(work.observation.error), now);
        return;
    }
    // row bytes come from the first owner histogram of the phase, and later ones must match
    std::string row_error;
    if (!window_controller_bind_row_bytes(runtime, std::move(work.observation.row_bytes), &row_error)) {
        defer(runtime, "observation_unavailable",
              runtime.measurement.fields + ",\"error\":" + json_quote(row_error), now);
        return;
    }
    if (runtime.proposals_disabled) {
        defer(runtime, "observation_only", runtime.measurement.fields, now);
        return;
    }
    if (work.observation.tables.per_rank_work != runtime.measurement.per_rank_w) {
        stop_on_error(runtime, "window_work_totals_failed", "\"error\":\"dump and HARVEST work differ\"");
        return;
    }
    if (runtime.config.repair.recall_guard && !std::isfinite(runtime.measurement.recall_percent)) {
        // an unscored window cannot start a batch under the recall guard
        defer(runtime, "recall_guard_unscored", "\"window_queries\":" + std::to_string(runtime.measurement.queries), now);
        return;
    }
    std::string error;
    if (!window_native_start_planning(runtime, g_daemon.membership, runtime.measurement.rank,
            coordinator_membership_ownership_epoch(), inputs.command_id, runtime.measurement.queries, error)) {
        defer(runtime, "propose_failed", "\"error\":" + json_quote(error), now);
        return;
    }
    runtime.state = WindowControllerState::PLAN_WAIT;
    const auto& eligibility = work.planning_input->eligibility;
    journal_event(runtime, "planning_submitted",
        "\"history_planning_window\":" + std::to_string(eligibility.planning_window) +
        ",\"history_cooldown_nodes\":" + std::to_string(eligibility.cooldown_nodes) +
        ",\"history_backoff_directions\":" + std::to_string(eligibility.backoff_directions));
}
}  // namespace

void controller_tick(const ControllerTickInputs& inputs) { window_controller_tick(inputs); }

void window_controller_tick(const ControllerTickInputs& live_inputs) {
    WindowControllerRuntime* runtime = window_controller_active();
    if (runtime == nullptr) return;
    if (runtime->stopped_on_error || runtime->stats_stop_pending) {
        window_native_cancel(*runtime);
        return;
    }
    const Clock::time_point now = Clock::now();
    window_native_freeze_endpoints(*runtime, live_inputs);
    window_native_poll_snapshots(*runtime);
    const auto* native = runtime->native_control.get();
    const bool frozen = native && native->endpoints_frozen && runtime->state == WindowControllerState::HARVEST_WAIT;
    const ControllerTickInputs& inputs = frozen ? native->endpoints : live_inputs;
    if (!runtime->proposals_disabled && runtime->config.proposal_query_limit > 0 &&
        live_inputs.completed_queries >= runtime->config.proposal_query_limit) {
        runtime->proposals_disabled = true;
        journal_event(*runtime, "proposal_limit_reached",
                      "\"completed_queries\":" + std::to_string(live_inputs.completed_queries));
    }

    switch (runtime->state) {
    case WindowControllerState::RECOVERY_FAILED:
        return;
    case WindowControllerState::COOLDOWN:
        if (now < runtime->cooldown_until) return;
        runtime->state = WindowControllerState::IDLE;
        return;

    case WindowControllerState::IDLE: {
        if (now < runtime->cooldown_until) return;
        runtime->window_epoch += 1;
        // Record the command before BEGIN goes out. ACKs from a later command must not open a window.
        runtime->window_begin_command = inputs.command_id;
        window_controller_arm_exchange(*runtime, WindowExchangeKind::BEGIN);
        runtime->exchange_deadline = now + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(window_controller_exchange_deadline_sec(runtime->config.ranks)));
        coordinator_stats_window_send_begin(runtime->window_epoch, runtime->nodes);
        runtime->state = WindowControllerState::BEGIN_WAIT;
        return;
    }

    case WindowControllerState::BEGIN_WAIT: {
        if (window_controller_exchange_complete(*runtime)) {
            runtime->exchange = WindowExchangeKind::NONE;
            if (inputs.command_id != runtime->window_begin_command) {
                // BEGIN went out in an earlier search command.
                const std::string fields = "\"reason\":\"search-command-changed-before-begin-ack\""
                    ",\"begin_command_id\":" + std::to_string(runtime->window_begin_command) +
                    ",\"command_id\":" + std::to_string(inputs.command_id);
                journal_event(*runtime, "window_abandoned", fields + verify_invalid_fields(*runtime));
                if (!verify_window_lost(*runtime, "window-abandoned", now)) enter_cooldown(*runtime, now);
                return;
            }
            runtime->window_begin_sums = runtime->exchange_counter_sums;
            runtime->window_begin_rank_counters = runtime->exchange_rank_counters;
            runtime->window_begin_completed = inputs.completed_queries;
            runtime->window_begin_recall_sum = inputs.recall_sum;
            runtime->window_begin_recall_count = inputs.recall_count;
            runtime->window_started = now;
            runtime->state = WindowControllerState::WINDOW_OPEN;
            journal_event(*runtime, "window_open", "\"command_id\":" + std::to_string(inputs.command_id) +
                ",\"completed_begin\":" + std::to_string(inputs.completed_queries));
            return;
        }
        if (runtime->exchange_rejected || now >= runtime->exchange_deadline) {
            journal_event(*runtime, "window_begin_failed",
                          "\"acks\":" + std::to_string(runtime->exchange_ack_count) +
                          ",\"rejected\":" + (runtime->exchange_rejected ? "true" : "false") +
                          verify_invalid_fields(*runtime));
            runtime->exchange = WindowExchangeKind::NONE;
            if (!verify_window_lost(*runtime, "window-begin-failed", now)) enter_cooldown(*runtime, now);
        }
        return;
    }

    case WindowControllerState::WINDOW_OPEN: {
        if (inputs.command_id != runtime->window_begin_command) {
            // counters restart with every search command, so this window cannot be measured
            journal_event(*runtime, "window_abandoned",
                          "\"reason\":\"search-command-changed\"" + verify_invalid_fields(*runtime));
            if (!verify_window_lost(*runtime, "window-abandoned", now)) enter_cooldown(*runtime, now);
            return;
        }
        if (!window_controller_observation_ready(*runtime, inputs, seconds_since(runtime->window_started, now)))
            return;
        runtime->cycles += 1;
        std::error_code ec;
        std::filesystem::create_directories(window_dump_dir(*runtime), ec);
        if (ec) {
            journal_event(*runtime, "dump_dir_failed", "\"error\":" + json_quote(ec.message()) +
                          verify_invalid_fields(*runtime));
            if (!verify_window_lost(*runtime, "dump-dir-failed", now)) enter_cooldown(*runtime, now);
            return;
        }
        window_controller_arm_exchange(*runtime, WindowExchangeKind::HARVEST);
        const double deadline_s = window_controller_exchange_deadline_sec(runtime->config.ranks);
        runtime->exchange_deadline = now + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(deadline_s));
        coordinator_stats_window_send_harvest(runtime->window_epoch, runtime->nodes,
                                              window_repair_structure_sections(runtime->config.repair));
        runtime->state = WindowControllerState::HARVEST_WAIT;
        // the deadline runs from armed_mono_ns to structure_loaded
        journal_event(*runtime, "harvest_requested",
            "\"armed_mono_ns\":" + std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count()) +
            ",\"deadline_s\":" + exact_number(deadline_s) + ",\"ranks\":" + std::to_string(runtime->config.ranks) +
            ",\"structure_sections\":" + std::to_string(window_repair_structure_sections(runtime->config.repair)));
        return;
    }

    case WindowControllerState::HARVEST_WAIT: {
        if (!window_controller_exchange_complete(*runtime)) {
            if (runtime->exchange_rejected || now >= runtime->exchange_deadline) {
                runtime->exchange = WindowExchangeKind::NONE;
                journal_event(*runtime, "harvest_failed", "\"acks\":" + std::to_string(runtime->exchange_ack_count) +
                              ",\"rejected\":" + (runtime->exchange_rejected ? "true" : "false") +
                              verify_invalid_fields(*runtime));
                if (!verify_window_lost(*runtime, "harvest-failed", now)) enter_cooldown(*runtime, now);
            }
            return;
        }
        if (now >= runtime->exchange_deadline) {
            runtime->exchange = WindowExchangeKind::NONE;
            journal_event(*runtime, "harvest_failed",
                          "\"reason\":\"decode-or-load-deadline\"" + verify_invalid_fields(*runtime));
            if (!verify_window_lost(*runtime, "harvest-deadline", now)) enter_cooldown(*runtime, now);
            return;
        }
        // every queued duplicate must be valid too
        for (auto& job : runtime->native_control->snapshots) if (job.valid()) return;
        if (!runtime->native_control->endpoints_frozen || !window_native_load(*runtime, g_daemon.membership)) return;
        runtime->exchange = WindowExchangeKind::NONE;
        if (live_inputs.command_id != inputs.command_id) {
            journal_event(*runtime, "harvest_failed",
                          "\"reason\":\"measurement-command-changed\"" + verify_invalid_fields(*runtime));
            if (!verify_window_lost(*runtime, "measurement-command-changed", now)) enter_cooldown(*runtime, now);
            return;
        }
        harvested(*runtime, inputs, now);
        return;
    }

    case WindowControllerState::PLAN_WAIT: {
        auto& work = *runtime->native_control;
        if (!window_native_plan_applicable(*runtime, coordinator_membership_ownership_epoch(), live_inputs)) {
            defer(*runtime, "planning_result_discarded", "\"reason\":\"owner-state-changed\"", now);
            return;
        }
        if (now >= work.planning_deadline) {
            defer(*runtime, "planning_result_discarded", "\"reason\":\"deadline\"", now);
            return;
        }
        if (!window_control_ready(work.planning)) return;
        try { work.proposal = work.planning.get(); }
        catch (const std::exception& e) { work.proposal.valid = false; work.proposal.error = e.what(); }
        auto& result = work.proposal;
        const std::string planning_fields = ",\"planning_us\":" + exact_number(result.work_us) +
            ",\"planning_wall_us\":" + exact_number(std::chrono::duration<double, std::micro>(
                Clock::now() - work.planning_started_at).count());
        if (!result.valid) {
            defer(*runtime, "propose_failed", runtime->measurement.fields + planning_fields +
                  ",\"error\":" + json_quote(result.error), now);
            return;
        }
        const std::string repair_fields = repair_install_plan(*runtime, result.repair);
        window_controller_apply_proposal(*runtime, std::move(result.actions),
                                         runtime->measurement.fields + planning_fields + repair_fields, now);
        return;
    }

    case WindowControllerState::FENCE: {
        if (runtime->proposals_disabled && !runtime->reverting) {
            journal_event(*runtime, "proposal_cancelled_limit", "");
            enter_cooldown(*runtime, now);
            return;
        }
        if (!window_controller_fence_drained(inputs)) {
            const double waited_s = seconds_since(runtime->fence_started, now);
            if (seconds_since(runtime->fence_last_heartbeat, now) >= kFenceHeartbeatSec) {
                runtime->fence_last_heartbeat = now;
                journal_event(*runtime, "fence_waiting",
                              "\"waited_s\":" + std::to_string(waited_s) +
                              ",\"logical_in_flight\":" + std::to_string(inputs.logical_in_flight) +
                              ",\"transport_in_flight\":" + std::to_string(inputs.transport_in_flight) +
                              ",\"queued_search\":" + std::to_string(inputs.queued_search) +
                              ",\"queued_responses\":" + std::to_string(inputs.queued_responses));
            }
            if (waited_s >= kFenceDrainDeadlineSec) {
                // the inverse MOVE is still owed
                journal_event(*runtime, "fence_drain_timeout", "\"waited_s\":" + std::to_string(waited_s) +
                              ",\"reverting\":" + (runtime->reverting ? "true" : "false"));
                if (runtime->reverting) stop_on_error(*runtime, "revert_fence_timeout", "");
                else enter_cooldown(*runtime, now);
            }
            return;
        }
        if (!runtime->reverting) {
            const auto* work = runtime->native_control.get();
            if (!work || coordinator_membership_ownership_epoch() != work->ownership_epoch || !runtime->action_history ||
                runtime->action_history->expected_epoch != work->ownership_epoch ||
                live_inputs.command_id != work->command_id || runtime->verify_pending) {
                journal_event(*runtime, "proposal_cancelled_stale", "");
                enter_cooldown(*runtime, now);
                return;
            }
            for (const auto& action : runtime->pending_actions) {
                if (action.node >= g_daemon.membership.size() || g_daemon.membership[action.node] != action.source_owner) {
                    journal_event(*runtime, "proposal_cancelled_stale", "\"reason\":\"membership-changed\"");
                    enter_cooldown(*runtime, now);
                    return;
                }
            }
        }
        const double fence_wait_s = seconds_since(runtime->fence_started, now);
        const std::string plan_path = runtime->pending_plan_path;
        const Clock::time_point prepare_start = Clock::now();
        if (!coordinator_membership_migration_prepare_controller(runtime->pending_plan)) {
            const bool rollback_ok = coordinator_membership_migration_rollback();
            journal_event(*runtime, "prepare_failed",
                          "\"fence_wait_s\":" + std::to_string(fence_wait_s) +
                          ",\"reverting\":" + (runtime->reverting ? "true" : "false") +
                          ",\"path\":" + json_quote(plan_path) + fence_payload_fields());
            if (runtime->reverting || !rollback_ok) {
                stop_on_error(*runtime, runtime->reverting ? "revert_prepare_failed" : "prepare_rollback_failed",
                              "\"rollback_ok\":" + std::string(rollback_ok ? "true" : "false"), !rollback_ok);
            } else {
                window_history_cancel(*runtime, now, true);
                enter_cooldown(*runtime, now);
            }
            return;
        }
        const std::string fence_fields =
            "\"fence_wait_s\":" + std::to_string(fence_wait_s) +
            ",\"reverting\":" + (runtime->reverting ? "true" : "false") +
            ",\"prepare_us\":" + std::to_string(static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - prepare_start).count())) +
            ",\"actions\":" + std::to_string(runtime->pending_actions.size()) + fence_payload_fields();
        if (!coordinator_membership_migration_commit()) {
            const bool rollback_ok = coordinator_membership_migration_rollback();
            stop_on_error(*runtime, "commit_failed",
                          fence_fields + ",\"rollback_ok\":" + (rollback_ok ? "true" : "false"), !rollback_ok);
            return;
        }
        if (!coordinator_membership_migration_accept()) {
            stop_on_error(*runtime, "accept_failed", fence_fields, true);
            return;
        }
        // pause_s: from opening the pause (drain wait included) to ACCEPT
        const std::string committed_fields =
            fence_fields + ",\"burst_bytes\":" + std::to_string(runtime->pending_bytes) +
            ",\"pause_s\":" + std::to_string(seconds_since(runtime->fence_started, Clock::now())) +
            ",\"migration_s\":" + exact_number(std::chrono::duration<double>(Clock::now() - prepare_start).count()) +
            ",\"epoch_to\":" + std::to_string(coordinator_membership_ownership_epoch());
        if (runtime->reverting) {
            runtime->moved_bytes += runtime->pending_bytes;
            if (!window_history_resolved(*runtime, false, coordinator_membership_ownership_epoch(), now)) {
                stop_on_error(*runtime, "history_reverse_failed", "");
                return;
            }
            runtime->reverts += 1;
            runtime->verify_pending = false;
            runtime->verify_invalid_windows = 0;
            runtime->last_committed_actions.clear();
            enter_cooldown(*runtime, now);
            journal_event(*runtime, "reverted", committed_fields);
            return;
        }
        if (!window_history_committed(*runtime, coordinator_membership_ownership_epoch(), now)) {
            stop_on_error(*runtime, "history_commit_failed", "");
            return;
        }
        runtime->moved_bytes += runtime->pending_bytes;
        runtime->commits += 1;
        runtime->actions_committed += static_cast<uint32_t>(runtime->pending_actions.size());
        ++runtime->repair_state.bursts;
        runtime->repair_state.cumulative_bytes += runtime->pending_bytes;
        runtime->last_committed_actions = runtime->pending_actions;
        runtime->verify_hop_per_query = runtime->staged_verify_hop_per_query;
        runtime->verify_recall_percent = runtime->staged_verify_recall_percent;
        runtime->verify_rank_cost = runtime->staged_verify_rank_cost;
        runtime->verify_pending = true;
        runtime->verify_invalid_windows = 0;
        enter_cooldown(*runtime, now);
        journal_event(*runtime, "committed", committed_fields + window_repair_state_json(runtime->repair_state));
        return;
    }
    }
}

}  // namespace tide
