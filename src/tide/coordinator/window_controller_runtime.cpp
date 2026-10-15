#include "window_controller_runtime.h"
#include "window_snapshot_transport.h"
#include "window_native_control.h"
#include "window_controller_transitions.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>

#include "coordinator_stats_window.h"
#include "membership_migration_plan.h"
#include "sha256.h"
#include "protocol/message_types.h"

namespace tide {

namespace {
std::unique_ptr<WindowControllerRuntime> g_controller;
// kept across controller-stop and controller-start
std::shared_ptr<WindowActionHistory> g_action_history;
// servers reject a window epoch not above their last one
uint64_t g_window_epoch_floor = 0;

bool unit(double value) { return value >= 0.0 && value <= 1.0; }

bool config_valid(const WindowControllerConfig& c) {
    const auto& r = c.repair;
    return window_owners_valid(c.ranks) && !c.journal_dir.empty() &&
        std::isfinite(c.window_seconds) && c.window_seconds > 0.0 &&
        std::isfinite(c.cooldown_seconds) && c.cooldown_seconds >= 0.0 &&
        r.enabled && r.ema > 0.0 && r.ema <= 1.0 &&
        (r.weight == WindowRepairWeight::W || r.weight == WindowRepairWeight::WNB) &&
        r.cumulative_cap > 0.0 && r.cumulative_cap <= 1.0 &&
        unit(r.refine_budget) && unit(r.balance_budget) && unit(r.tau) &&
        std::isfinite(r.eps) && r.eps >= 0.0 &&
        std::isfinite(r.skew_trigger) && r.skew_trigger >= 1.0 &&
        std::isfinite(r.slack) && r.slack >= 0.0 &&
        std::isfinite(r.hop_slack) && r.hop_slack >= 0.0 &&
        (!r.recall_guard || (std::isfinite(r.recall_guard_pp) && r.recall_guard_pp >= 0.0)) &&
        std::isfinite(r.guard) && r.guard >= 0.0 && r.converge_windows > 0 && r.verify_invalid_limit > 0;
}

double frozen_rank_time_cost(const WindowRankCostSnapshot& s, const WindowRankTimeContext& prices,
                             std::vector<double>* by_rank = nullptr) {
    const size_t ranks = prices.ranks.size();
    if (!window_rank_time_valid(prices) || s.distances_per_query.size() != ranks ||
        s.requests_per_query.size() != ranks)
        return std::numeric_limits<double>::quiet_NaN();
    if (by_rank) by_rank->assign(ranks, 0.0);
    double peak = 0;
    for (size_t rank = 0; rank < ranks; ++rank) {
        const double d = s.distances_per_query[rank], r = s.requests_per_query[rank];
        const auto& p = prices.ranks[rank];
        const double cost = p.c_d_us * d + p.c_r_us * r;
        if (!std::isfinite(d) || d < 0 || !std::isfinite(r) || r < 0 || !std::isfinite(cost))
            return std::numeric_limits<double>::quiet_NaN();
        if (by_rank) (*by_rank)[rank] = cost;
        peak = std::max(peak, cost);
    }
    return peak;
}
}  // namespace

WindowControllerRuntime* window_controller_start(
    const WindowControllerConfig& config, size_t node_count, std::string* error) {
    if (g_controller != nullptr) {
        if (error != nullptr) *error = "controller already active";
        return nullptr;
    }
    if (node_count == 0 || node_count > UINT32_MAX || !config_valid(config)) {
        if (error != nullptr)
            *error = window_owners_valid(config.ranks) ? "invalid controller config"
                                                       : "invalid controller config: the server count must be 2..64";
        return nullptr;
    }
    auto runtime = std::make_unique<WindowControllerRuntime>();
    runtime->config = config;
    runtime->nodes = static_cast<uint32_t>(node_count);
    runtime->journal.open(config.journal_dir + "/controller-journal.jsonl", std::ios::app);
    if (!runtime->journal.good()) {
        if (error != nullptr) *error = "cannot open controller journal";
        return nullptr;
    }
    runtime->cooldown_until = std::chrono::steady_clock::now();
    std::string history_error;
    if (!window_history_bind(*runtime, g_action_history, config.start_ownership_epoch, history_error)) {
        if (error) *error = history_error;
        return nullptr;
    }
    // servers reject a window epoch not above their last one
    runtime->window_epoch = std::max(runtime->window_epoch, g_window_epoch_floor);
    runtime->stats_off_confirmed = true;  // no window open yet
    g_controller = std::move(runtime);
    const auto& history = *g_controller->action_history;
    window_control_detail::journal_event(*g_controller, "controller_started",
        "\"nodes\":" + std::to_string(node_count) + ",\"ranks\":" + std::to_string(config.ranks) +
        ",\"phase\":" + std::to_string(history.phases) +
        ",\"vector_bytes_per_row\":" + std::to_string(config.vector_bytes_per_row) +
        ",\"row_bytes_unit\":\"" + (config.vector_bytes_per_row ? "adjacency+vector" : "adjacency") + "\"" +
        ",\"carried_pred_nodes\":" + std::to_string(g_controller->repair_state.pred.size()) +
        ",\"carried_repair_windows\":" + std::to_string(g_controller->repair_state.windows) +
        ",\"window_seconds\":" + window_control_detail::exact_number(config.window_seconds) +
        ",\"cooldown_seconds\":" + window_control_detail::exact_number(config.cooldown_seconds) +
        ",\"min_window_queries\":" + std::to_string(config.min_window_queries) +
        ",\"proposal_query_limit\":" + std::to_string(config.proposal_query_limit) +
        ",\"structure_sections\":" + std::to_string(window_repair_structure_sections(config.repair)) +
        window_repair_config_json(config.repair));
    return g_controller.get();
}

std::vector<uint64_t> window_controller_histogram_row_bytes(const std::vector<uint32_t>& owner_hist,
                                                            uint32_t owners, uint32_t vector_bytes) {
    std::vector<uint64_t> bytes;
    if (!owners || owner_hist.size() % owners) return bytes;
    const size_t nodes = owner_hist.size() / owners;
    bytes.resize(nodes);
    for (size_t v = 0; v < nodes; ++v) {
        uint64_t degree = 0;
        for (uint32_t p = 0; p < owners; ++p) degree += owner_hist[v * owners + p];
        bytes[v] = 4 + 4 * degree + vector_bytes;
    }
    return bytes;
}

bool window_controller_bind_row_bytes(WindowControllerRuntime& r, std::vector<uint64_t> row_bytes,
                                      std::string* error) {
    if (row_bytes.size() != r.nodes) {
        if (error) *error = "row bytes for " + std::to_string(row_bytes.size()) + " nodes, the window has " +
                            std::to_string(r.nodes);
        return false;
    }
    if (!r.row_bytes.empty()) {
        if (row_bytes == r.row_bytes) return true;
        size_t differing = 0, first = row_bytes.size();
        for (size_t v = 0; v < row_bytes.size(); ++v)
            if (row_bytes[v] != r.row_bytes[v]) { if (first == row_bytes.size()) first = v; ++differing; }
        window_control_detail::journal_event(r, "row_bytes_mismatch",
            "\"differing_nodes\":" + std::to_string(differing) + ",\"first_node\":" + std::to_string(first) +
            ",\"bound\":" + std::to_string(r.row_bytes[first]) + ",\"harvest\":" + std::to_string(row_bytes[first]));
        if (error) *error = "row bytes changed within the phase at " + std::to_string(differing) + " nodes";
        return false;
    }
    uint64_t total = 0, adjacency_total = 0;
    std::vector<uint32_t> adjacency(row_bytes.size());
    for (size_t v = 0; v < row_bytes.size(); ++v) {
        total += row_bytes[v];
        const uint64_t row = row_bytes[v] - r.config.vector_bytes_per_row;
        adjacency_total += row;
        adjacency[v] = static_cast<uint32_t>(row);
    }
    // sha256 of the adjacency row bytes (4 + 4 x degree) as little-endian uint32
    const std::string digest = sha256_hex(reinterpret_cast<const uint8_t*>(adjacency.data()),
                                          adjacency.size() * sizeof(uint32_t));
    r.row_bytes = std::move(row_bytes);
    r.total_bytes = total;
    window_history_bind_phase_bytes(r);
    const auto& h = *r.action_history;
    window_control_detail::journal_event(r, "row_bytes_bound",
        "\"nodes\":" + std::to_string(r.nodes) + ",\"total_bytes\":" + std::to_string(total) +
        ",\"adjacency_bytes\":" + std::to_string(adjacency_total) +
        ",\"vector_bytes_per_row\":" + std::to_string(r.config.vector_bytes_per_row) +
        ",\"row_bytes_unit\":\"" + (r.config.vector_bytes_per_row ? "adjacency+vector" : "adjacency") + "\"" +
        ",\"adjacency_row_bytes_sha256\":\"" + digest + "\"" +
        ",\"phase\":" + std::to_string(h.phases) + ",\"total_cap_bytes\":" + std::to_string(h.total_cap) +
        ",\"move_rate_cap_bytes\":" + std::to_string(h.move_rate_cap));
    return true;
}

bool window_controller_has_unresolved_action(const WindowControllerRuntime& runtime) {
    return (runtime.action_history && !runtime.action_history->pending.empty()) ||
           runtime.verify_pending || runtime.reverting || runtime.ownership_ambiguous ||
           !runtime.pending_actions.empty() || !runtime.last_committed_actions.empty() ||
           runtime.stopped_on_error;
}

bool window_controller_stop() {
    if (!g_controller) return true;
    auto& runtime = *g_controller;
    if (!runtime.stats_off_confirmed || window_controller_has_unresolved_action(runtime)) {
        window_control_detail::journal_event(runtime, "stop_refused",
            std::string("\"stats_off_confirmed\":") + (runtime.stats_off_confirmed ? "true" : "false") +
            ",\"unresolved_action\":" + (window_controller_has_unresolved_action(runtime) ? "true" : "false"));
        return false;
    }
    window_native_cancel(runtime);
    window_history_save(runtime);
    g_window_epoch_floor = runtime.window_epoch;
    window_control_detail::journal_event(runtime, "controller_stopped", "");
    g_controller.reset();
    return true;
}

bool window_controller_abort_allowed(const WindowControllerRuntime& runtime, std::string* refusal) {
    std::string why;
    if (!runtime.stopped_on_error)
        why = "the controller is not terminal: controller-stop ends it once its batch is resolved";
    else if (runtime.ownership_ambiguous)
        why = "ownership is ambiguous after " + runtime.terminal_reason +
              ": complete migration-accept (or migration-rollback if ACCEPT was never sent) first";
    if (refusal) *refusal = why;
    return why.empty();
}

bool window_controller_abort(std::string* refusal) {
    if (!g_controller) return true;
    auto& runtime = *g_controller;
    std::string why;
    if (window_controller_abort_allowed(runtime, &why) && !runtime.stats_off_confirmed)
        why = "the servers' stats windows are not confirmed closed";
    if (!why.empty()) {
        window_control_detail::journal_event(runtime, "abort_refused",
            "\"terminal_reason\":" + window_control_detail::json_quote(runtime.terminal_reason) +
            ",\"reason\":" + window_control_detail::json_quote(why));
        if (refusal) *refusal = why;
        return false;
    }
    window_native_cancel(runtime);
    const auto& h = runtime.action_history;
    window_control_detail::journal_event(runtime, "controller_aborted",
        "\"terminal_reason\":" + window_control_detail::json_quote(runtime.terminal_reason) +
        ",\"verify_pending\":" + (runtime.verify_pending ? "true" : "false") +
        ",\"reverting\":" + (runtime.reverting ? "true" : "false") +
        ",\"history_dropped\":true,\"history_expected_epoch\":" + std::to_string(h ? h->expected_epoch : 0) +
        ",\"history_pending_actions\":" + std::to_string(h ? h->pending.size() : 0));
    g_window_epoch_floor = runtime.window_epoch;
    g_action_history.reset();
    g_controller.reset();
    if (refusal) refusal->clear();
    return true;
}

void window_controller_confirm_ownership(uint64_t ownership_epoch, const char* command) {
    if (!g_controller || !g_controller->ownership_ambiguous) return;
    g_controller->ownership_ambiguous = false;
    window_control_detail::journal_event(*g_controller, "ownership_confirmed",
        "\"ownership_epoch\":" + std::to_string(ownership_epoch) +
        ",\"command\":" + window_control_detail::json_quote(command ? command : "") +
        ",\"terminal_reason\":" + window_control_detail::json_quote(g_controller->terminal_reason));
}

WindowControllerRuntime* window_controller_active() { return g_controller.get(); }

bool window_controller_dispatch_frozen() {
    const WindowControllerRuntime* runtime = window_controller_active();
    return runtime != nullptr &&
           (runtime->state == WindowControllerState::FENCE || runtime->ownership_ambiguous);
}

bool window_controller_reset_history() {
    if (g_controller) return false;
    // reset undid the moves the history describes, but the window-epoch floor stays
    g_action_history.reset();
    return true;
}

bool controller_running() { return g_controller != nullptr; }
bool controller_dispatch_allowed() { return !window_controller_dispatch_frozen(); }
bool controller_dispatch_stalled(const ControllerTickInputs& inputs, std::string* reason) {
    const WindowControllerRuntime* runtime = window_controller_active();
    if (!runtime || !runtime->ownership_ambiguous || !window_controller_fence_drained(inputs)) return false;
    if (reason) *reason = runtime->terminal_reason;
    return true;
}
bool controller_offer(uint32_t server, const void* data, size_t size) {
    WindowControllerRuntime* runtime = window_controller_active();
    if (window_snapshot_is_response(data, size))
        return runtime ? window_snapshot_offer(*runtime, server, data, size) : true;
    return runtime ? window_controller_offer_ack(*runtime, server, data, size)
                   : window_controller_is_ack(data, size);
}
bool controller_offer_owned(uint32_t server, std::vector<char>& bytes) {
    if (!window_snapshot_is_response(bytes.data(), bytes.size()))
        return controller_offer(server, bytes.data(), bytes.size());
    if (auto* runtime = window_controller_active())
        window_snapshot_offer_owned(*runtime, server, std::move(bytes));
    return true;
}

bool window_controller_observation_ready(const WindowControllerRuntime& runtime,
    const ControllerTickInputs& inputs, double elapsed_seconds) {
    return elapsed_seconds >= runtime.config.window_seconds &&
        inputs.command_id == runtime.window_begin_command &&
        inputs.completed_queries >= runtime.window_begin_completed &&
        inputs.completed_queries - runtime.window_begin_completed >= runtime.config.min_window_queries;
}

bool window_controller_rank_snapshot(const WindowControllerRuntime& runtime, uint64_t queries,
                                     WindowRankCostSnapshot* out) {
    if (!out) return false;
    *out = {};
    out->rank_time.Q = queries;
    if (!queries || !window_owners_valid(runtime.config.ranks) ||
        runtime.exchange_ack_count != runtime.config.ranks ||
        runtime.exchange_acked.size() != runtime.config.ranks ||
        std::find(runtime.exchange_acked.begin(), runtime.exchange_acked.end(), 0) != runtime.exchange_acked.end() ||
        runtime.exchange_rank_counters.size() != runtime.config.ranks ||
        runtime.window_begin_rank_counters.size() != runtime.config.ranks) return false;
    out->rank_time.ranks.assign(runtime.config.ranks, {});
    for (uint32_t r = 0; r < runtime.config.ranks; ++r) {
        const auto& b = runtime.window_begin_rank_counters[r];
        const auto& e = runtime.exchange_rank_counters[r];
        if (e.requests < b.requests || e.distances < b.distances ||
            e.distance_us < b.distance_us || e.total_us < b.total_us) {
            out->rank_time.reason = "rank-" + std::to_string(r) + "-counter-regression";
            return false;
        }
        out->distances_per_query.push_back(double(e.distances - b.distances) / queries);
        out->requests_per_query.push_back(double(e.requests - b.requests) / queries);
        out->rank_time.ranks[r] = {e.distances - b.distances, e.requests - b.requests,
                                   e.distance_us - b.distance_us, e.total_us - b.total_us};
    }
    window_rank_time_seal(&out->rank_time);
    return window_rank_time_valid(out->rank_time);
}

bool window_controller_measurement_valid(const WindowControllerRuntime& runtime,
                                         const ControllerTickInputs& inputs) {
    const auto& b = runtime.window_begin_sums;
    const auto& e = runtime.exchange_counter_sums;
    if (inputs.command_id != runtime.window_begin_command ||
        inputs.completed_queries <= runtime.window_begin_completed) return false;
    const uint64_t queries = inputs.completed_queries - runtime.window_begin_completed;
    WindowRankCostSnapshot ignored;
    return queries >= runtime.config.min_window_queries &&
        inputs.recall_count >= runtime.window_begin_recall_count &&
        std::isfinite(inputs.recall_sum) && inputs.recall_sum >= runtime.window_begin_recall_sum &&
        e.requests > b.requests && e.distances > b.distances &&
        e.distance_us >= b.distance_us && e.total_us >= b.total_us &&
        e.total_us - b.total_us >= e.distance_us - b.distance_us &&
        runtime.harvest_total_w > 0 && window_controller_rank_snapshot(runtime, queries, &ignored);
}

double window_controller_window_recall(const WindowControllerRuntime& runtime,
                                       const ControllerTickInputs& inputs) {
    if (inputs.command_id != runtime.window_begin_command ||
        inputs.completed_queries <= runtime.window_begin_completed ||
        inputs.recall_count < runtime.window_begin_recall_count)
        return std::numeric_limits<double>::quiet_NaN();
    const uint64_t queries = inputs.completed_queries - runtime.window_begin_completed;
    const uint64_t scored = inputs.recall_count - runtime.window_begin_recall_count;
    const double sum = inputs.recall_sum - runtime.window_begin_recall_sum;
    if (scored != queries || !std::isfinite(sum) || sum < 0.0 || sum > double(queries))
        return std::numeric_limits<double>::quiet_NaN();
    return 100.0 * sum / double(queries);
}

double window_controller_skew(const std::vector<uint64_t>& per_rank_w) {
    if (per_rank_w.empty()) return 0.0;
    uint64_t sum = 0, peak = 0;
    for (uint64_t value : per_rank_w) {
        sum += value;
        peak = std::max(peak, value);
    }
    if (sum == 0) return 0.0;
    return static_cast<double>(peak) / (static_cast<double>(sum) / static_cast<double>(per_rank_w.size()));
}

bool window_controller_skeleton_plan(const std::vector<WindowBurstAction>& actions, uint64_t epoch_from,
    uint32_t n_servers, std::vector<char>* out, std::string* error) {
    if (out == nullptr || actions.empty()) {
        if (error != nullptr) *error = "no actions to plan";
        return false;
    }
    MembershipMigrationPlan plan;
    plan.n_servers = n_servers;
    plan.mode = kMigrationPlanModeMove;
    plan.epoch_from = epoch_from;
    plan.epoch_to = epoch_from + 1;
    for (const WindowBurstAction& action : actions) {
        MembershipMove move;
        move.global_id = action.node;
        move.from_partition = action.source_owner;
        move.to_partition = action.target_owner;
        move.adjacency_digest = migration_row_digest(move.adjacency);
        plan.moves.push_back(std::move(move));
    }
    std::string serialize_error;
    if (!serialize_membership_migration_plan(plan, *out, serialize_error)) {
        if (error != nullptr) *error = serialize_error;
        return false;
    }
    return true;
}

std::vector<WindowBurstAction> window_controller_inverse_actions(const std::vector<WindowBurstAction>& actions) {
    std::vector<WindowBurstAction> inverse = actions;
    for (WindowBurstAction& action : inverse) std::swap(action.source_owner, action.target_owner);
    return inverse;
}

void window_controller_arm_exchange(WindowControllerRuntime& runtime, WindowExchangeKind kind) {
    window_native_arm(runtime);
    runtime.exchange = kind;
    runtime.exchange_acked.assign(runtime.config.ranks, 0);
    runtime.exchange_ack_count = 0;
    runtime.exchange_rejected = false;
    runtime.exchange_counter_sums = WindowCounterSums{};
    runtime.exchange_rank_counters.assign(runtime.config.ranks, {});
    if (kind == WindowExchangeKind::BEGIN) runtime.stats_off_confirmed = false;
    if (kind == WindowExchangeKind::HARVEST) {
        runtime.snapshot_digests.assign(runtime.config.ranks, {});
        runtime.snapshot_frames.assign(runtime.config.ranks, nullptr);
        runtime.harvest_per_rank_w.assign(runtime.config.ranks, 0);
        runtime.harvest_total_w = 0;
    }
}

bool window_controller_rank_costs(const WindowRankCostSnapshot& prices, const WindowRankCostSnapshot& measured,
                                  std::vector<double>* costs) {
    if (!costs || !window_rank_time_valid(prices.rank_time)) return false;
    std::vector<double> by_rank;
    if (!std::isfinite(frozen_rank_time_cost(measured, prices.rank_time, &by_rank))) return false;
    *costs = std::move(by_rank);
    return true;
}

double window_controller_exchange_deadline_sec(uint32_t ranks) {
    return window_control_detail::kExchangeDeadlineSec * std::max(1.0, double(ranks) / 4.0);
}

void window_controller_repair_freeze_prices(WindowControllerRuntime& runtime,
                                            const WindowRankCostSnapshot& snapshot) {
    auto& state = runtime.repair_state;
    if (state.price_frozen) return;
    const size_t ranks = runtime.config.ranks;
    state.price.assign(ranks, 1.0); state.price_r.assign(ranks, 0.0);
    if (window_rank_time_valid(snapshot.rank_time)) {
        // One pooled price for all ranks.
        uint64_t D = 0, R = 0, Td = 0, Tr = 0;
        for (const auto& row : snapshot.rank_time.ranks) {
            D += row.D; R += row.R; Td += row.Td_us; Tr += row.Ttot_us - row.Td_us;
        }
        if (D && Td) state.price.assign(ranks, double(Td) / double(D));
        if (R && Tr) state.price_r.assign(ranks, double(Tr) / double(R));
    }
    state.price_frozen = true;
}

double window_controller_repair_peak_us(const WindowControllerRuntime& runtime,
                                        const WindowRankCostSnapshot& snapshot) {
    const auto& state = runtime.repair_state;
    const size_t ranks = state.price.size();
    if (!state.price_frozen || !ranks || state.price_r.size() != ranks ||
        snapshot.distances_per_query.size() != ranks || snapshot.requests_per_query.size() != ranks)
        return std::numeric_limits<double>::quiet_NaN();
    double peak = 0.0;
    for (size_t rank = 0; rank < ranks; ++rank) {
        const double d = snapshot.distances_per_query[rank], r = snapshot.requests_per_query[rank];
        const double cost = state.price[rank] * d + state.price_r[rank] * r;
        if (!std::isfinite(d) || d < 0 || !std::isfinite(r) || r < 0 || !std::isfinite(cost))
            return std::numeric_limits<double>::quiet_NaN();
        peak = std::max(peak, cost);
    }
    return peak;
}

bool window_controller_is_ack(const void* data, size_t size) {
    if (!data || size != sizeof(ann::StatsWindowAck)) return false;
    ann::StatsWindowAck ack{};
    std::memcpy(&ack, data, sizeof(ack));
    return ack.msg_type == ann::MessageType::STATS_WINDOW_BEGIN_ACK ||
           ack.msg_type == ann::MessageType::STATS_WINDOW_HARVEST_ACK;
}

bool window_controller_offer_ack(WindowControllerRuntime& runtime, uint32_t server,
                                 const void* data, size_t size) {
    if (!window_controller_is_ack(data, size)) return false;
    ann::StatsWindowAck ack{};
    std::memcpy(&ack, data, sizeof(ack));
    const bool is_begin = ack.msg_type == ann::MessageType::STATS_WINDOW_BEGIN_ACK;
    const bool is_harvest = ack.msg_type == ann::MessageType::STATS_WINDOW_HARVEST_ACK;
    // Every stats-window ACK ends here. Only those of the pending exchange change state.
    const bool matches = (runtime.exchange == WindowExchangeKind::BEGIN && is_begin) ||
                         (runtime.exchange == WindowExchangeKind::HARVEST && is_harvest);
    if (!matches || ack.window_epoch != runtime.window_epoch) return true;
    if (ack.status != 0 || server >= runtime.config.ranks || ack.server_rank != server) {
        runtime.exchange_rejected = true;
        return true;
    }
    if (runtime.exchange_acked[server]) return true;
    // a HARVEST counts only with its snapshot
    if (is_harvest && (runtime.snapshot_digests.size() != runtime.config.ranks ||
                       runtime.snapshot_digests[server].empty())) {
        runtime.exchange_rejected = true;
        return true;
    }
    runtime.exchange_acked[server] = 1;
    runtime.exchange_ack_count += 1;
    runtime.exchange_counter_sums.requests += ack.search_request_count;
    runtime.exchange_counter_sums.distances += ack.search_distance_count;
    runtime.exchange_counter_sums.distance_us += ack.search_distance_us;
    runtime.exchange_counter_sums.total_us += ack.search_total_us;
    runtime.exchange_counter_sums.adc_tables += ack.search_adc_table_count;
    if (runtime.exchange_rank_counters.size() == runtime.config.ranks)
        runtime.exchange_rank_counters[server] = {ack.search_request_count, ack.search_distance_count,
                                                  ack.search_distance_us, ack.search_total_us,
                                                  ack.search_adc_table_count};
    if (is_harvest) {
        runtime.harvest_per_rank_w[server] = ack.total_w;
        runtime.harvest_total_w += ack.total_w;
    }
    return true;
}

std::string window_controller_status_json(const WindowControllerRuntime& r) {
    using window_control_detail::exact_number;
    std::ostringstream out;
    out << "\"state\":" << static_cast<int>(r.state) << ",\"ranks\":" << r.config.ranks
        << ",\"window_epoch\":" << r.window_epoch << ",\"cycles\":" << r.cycles
        << ",\"commits\":" << r.commits << ",\"reverts\":" << r.reverts
        << ",\"actions_committed\":" << r.actions_committed
        << ",\"moved_bytes\":" << r.moved_bytes << ",\"total_bytes\":" << r.total_bytes
        << ",\"verify_pending\":" << (r.verify_pending ? "true" : "false")
        << ",\"verify_invalid_windows\":" << r.verify_invalid_windows
        << ",\"unresolved_action\":" << (window_controller_has_unresolved_action(r) ? "true" : "false")
        << ",\"proposals_disabled\":" << (r.proposals_disabled ? "true" : "false")
        << ",\"stopped_on_error\":" << (r.stopped_on_error ? "true" : "false")
        << ",\"ownership_ambiguous\":" << (r.ownership_ambiguous ? "true" : "false")
        << ",\"dispatch_frozen\":" << (window_controller_dispatch_frozen() ? "true" : "false")
        << ",\"stats_off_confirmed\":" << (r.stats_off_confirmed ? "true" : "false")
        << ",\"stats_stop_pending\":" << (r.stats_stop_pending ? "true" : "false")
        << ",\"terminal_reason\":" << window_control_detail::json_quote(r.terminal_reason)
        << ",\"window_seconds\":" << exact_number(r.config.window_seconds)
        << ",\"cooldown_seconds\":" << exact_number(r.config.cooldown_seconds)
        << ",\"min_window_queries\":" << r.config.min_window_queries
        << ",\"proposal_query_limit\":" << r.config.proposal_query_limit
        << window_repair_config_json(r.config.repair) << window_repair_state_json(r.repair_state)
        << window_history_json(r, std::chrono::steady_clock::now());
    return out.str();
}

}  // namespace tide
