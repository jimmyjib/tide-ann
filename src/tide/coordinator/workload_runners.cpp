#include "coordinator_types.h"
#include "pipeline_types.h"
#include "protocol/broadcast_messages.h"
#include "npy_reader.h"
#include "npy_writer.h"

#include <iostream>
#include <sstream>
#include <random>
#include <algorithm>
#include <set>
#include <atomic>
#include <iomanip>

using namespace std::chrono;

namespace tide {

// --debug: per-query CSV logs in the working directory, or a warning if it is not writable.
static void enable_debug_logs(const std::string& log_path) {
    enable_partition_debug(g_daemon.n_servers, log_path, &g_daemon.membership);
    if (!g_debug.log_file.is_open())
        daemon_output_line("Warning: --debug: cannot write " + log_path +
                           "; per-query logs are off for this command");
}

// --debug: top-K ids of every search as debug_results.npy. Returns the path, empty on failure.
static std::string dump_debug_results(const UnifiedContext& ctx) {
    std::vector<uint32_t> all_results;
    uint32_t result_rows = 0;
    const uint32_t K = ctx.K;
    for (const auto& item : ctx.workloads) {
        if (item.type == tide::WorkloadType::SEARCH && item.done) {
            for (uint32_t i = 0; i < K; i++)
                all_results.push_back(i < item.result_neighbors.size() ? item.result_neighbors[i] : UINT32_MAX);
            result_rows++;
        }
    }
    const std::string dump_path = "debug_results.npy";
    try {
        ann::write_npy_uint32_2d(dump_path, all_results, result_rows, K);
    } catch (const std::exception& e) {
        daemon_output_line(std::string("Warning: --debug: cannot write ") + dump_path + ": " + e.what());
        return "";
    }
    daemon_output_line("Debug results dumped: " + dump_path + " (" + std::to_string(result_rows) + " x " +
                       std::to_string(K) + ")");
    return dump_path;
}

static void fnv1a64_append_u32(uint64_t& hash, uint32_t value) {
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    // little-endian on every host
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        hash ^= static_cast<uint8_t>((value >> shift) & 0xffU);
        hash *= kFnvPrime;
    }
}

// FNV-1a over the ranked top-K ids in query order. Missing slots are UINT32_MAX.
static void populate_search_result_digest(const UnifiedContext& ctx,
                                          ann::WorkloadResult& result) {
    std::vector<const WorkloadItem*> rows;
    rows.reserve(ctx.searches_completed);
    for (const auto& item : ctx.workloads) {
        if (item.type == WorkloadType::SEARCH && item.done) rows.push_back(&item);
    }
    if (rows.empty()) return;

    auto ranked_value = [&](const WorkloadItem& item, uint32_t rank) {
        return rank < item.result_neighbors.size()
            ? item.result_neighbors[rank]
            : UINT32_MAX;
    };
    std::sort(rows.begin(), rows.end(), [&](const WorkloadItem* lhs,
                                            const WorkloadItem* rhs) {
        if (lhs->query_index != rhs->query_index) {
            return lhs->query_index < rhs->query_index;
        }
        for (uint32_t rank = 0; rank < ctx.K; ++rank) {
            uint32_t left = ranked_value(*lhs, rank);
            uint32_t right = ranked_value(*rhs, rank);
            if (left != right) return left < right;
        }
        return false;
    });

    uint64_t hash = 14695981039346656037ULL;
    fnv1a64_append_u32(hash, 0x42544b31U);  // format tag "BTK1": a new tag changes every digest
    fnv1a64_append_u32(hash, ctx.K);
    fnv1a64_append_u32(hash, static_cast<uint32_t>(rows.size()));
    uint64_t neighbor_count = 0;
    for (const WorkloadItem* item : rows) {
        fnv1a64_append_u32(hash, item->query_index);
        neighbor_count += std::min<size_t>(ctx.K, item->result_neighbors.size());
        for (uint32_t rank = 0; rank < ctx.K; ++rank) {
            fnv1a64_append_u32(hash, ranked_value(*item, rank));
        }
    }

    std::ostringstream digest;
    digest << std::hex << std::setfill('0') << std::setw(16) << hash;
    result.search_result_digest_algorithm =
        "fnv1a64-sorted-query-index-ranked-topk-v1";
    result.search_result_digest = digest.str();
    result.search_result_digest_queries = static_cast<uint32_t>(rows.size());
    result.search_result_digest_neighbors = neighbor_count;
}

static void copy_network_baseline(ann::WorkloadResult& result) {
    if (!g_daemon.baseline.measured) return;

    result.network_baseline.measured = true;
    result.network_baseline.ping_count = g_daemon.baseline.ping_count;

    result.network_baseline.servers.resize(g_daemon.baseline.servers.size());
    for (size_t i = 0; i < g_daemon.baseline.servers.size(); i++) {
        result.network_baseline.servers[i].id = g_daemon.baseline.servers[i].id;
        result.network_baseline.servers[i].count = g_daemon.baseline.servers[i].count;
        result.network_baseline.servers[i].avg_us = g_daemon.baseline.servers[i].avg_us;
        result.network_baseline.servers[i].min_us = g_daemon.baseline.servers[i].min_us;
        result.network_baseline.servers[i].p50_us = g_daemon.baseline.servers[i].p50_us;
        result.network_baseline.servers[i].p95_us = g_daemon.baseline.servers[i].p95_us;
        result.network_baseline.servers[i].p99_us = g_daemon.baseline.servers[i].p99_us;
        result.network_baseline.servers[i].max_us = g_daemon.baseline.servers[i].max_us;
    }

    result.network_baseline.total_pings = g_daemon.baseline.total_pings;
    result.network_baseline.overall_avg_us = g_daemon.baseline.overall_avg_us;
    result.network_baseline.overall_min_us = g_daemon.baseline.overall_min_us;
    result.network_baseline.overall_p50_us = g_daemon.baseline.overall_p50_us;
    result.network_baseline.overall_p95_us = g_daemon.baseline.overall_p95_us;
    result.network_baseline.overall_p99_us = g_daemon.baseline.overall_p99_us;
    result.network_baseline.overall_max_us = g_daemon.baseline.overall_max_us;

    result.network_baseline.direct.avg_us = g_daemon.baseline.direct.avg_us;
    result.network_baseline.direct.min_us = g_daemon.baseline.direct.min_us;
    result.network_baseline.direct.p50_us = g_daemon.baseline.direct.p50_us;
    result.network_baseline.direct.p95_us = g_daemon.baseline.direct.p95_us;
    result.network_baseline.direct.p99_us = g_daemon.baseline.direct.p99_us;
    result.network_baseline.direct.max_us = g_daemon.baseline.direct.max_us;
}

// An early end reports an error and no partial result. Inserts may be partly applied.
static bool report_command_ended_early(const UnifiedContext& ctx, const char* action, uint32_t queries = 0) {
    if (ctx.aborted_reason.empty()) return false;
    const bool inserts = std::string(action) != "search";
    daemon_output_line(std::string("Error: ") + action + " command ended early: " + ctx.aborted_reason +
                       (inserts ? " (the servers may hold part of these inserts: reset before serving again)" : ""));
    std::string quoted;
    for (const char c : ctx.aborted_reason) {
        if (c == '"' || c == '\\') quoted += '\\';
        quoted += c;
    }
    daemon_output_line("===JSON_RESULT===");
    daemon_output_line(std::string("{\"status\": \"error\", \"action\": \"") + action + "\", \"error\": \"" + quoted +
                       "\", \"completed_queries\": " + std::to_string(ctx.searches_completed) +
                       ", \"completed_inserts\": " + std::to_string(ctx.inserts_completed) +
                       ", \"queries\": " + std::to_string(queries) + "}");
    daemon_output_line("===JSON_END===");
    return true;
}

static void copy_observability_metadata(ann::WorkloadResult& result, bool diagnostic_mode = false) {
    result.diagnostic_mode = diagnostic_mode;
}

void run_search_workload(uint32_t n_queries, const std::string& gt_path) {
    run_search_workload_ex(n_queries, gt_path, "", 0, false, 0);
}

// recall --self-ids: rank of each query's own node, and whether its distance is 0.
static constexpr size_t kSelfQueryFailureLimit = 100000;
static void populate_self_query(const UnifiedContext& ctx, ann::WorkloadResult& result) {
    if (ctx.self_ids.empty()) return;
    result.self_query = true;
    for (const auto& item : ctx.workloads) {
        if (item.type != WorkloadType::SEARCH || !item.done) continue;
        ++result.self_query_count;
        const uint32_t expected = item.query_index < ctx.self_ids.size() ? ctx.self_ids[item.query_index] : UINT32_MAX;
        const size_t k = std::min<size_t>(ctx.K, item.result_neighbors.size());
        int32_t rank = -1;
        for (size_t r = 0; r < k; ++r) {
            if (item.result_neighbors[r] == expected) { rank = static_cast<int32_t>(r); break; }
        }
        const float distance = rank >= 0 && static_cast<size_t>(rank) < item.result_distances.size()
            ? item.result_distances[static_cast<size_t>(rank)] : -1.0f;
        const bool at_zero = rank >= 0 && distance == 0.0f;
        if (at_zero) {
            ++result.self_query_found_at_zero;
            if (rank == 0) ++result.self_query_top1_at_zero;
            continue;
        }
        if (rank >= 0) ++result.self_query_found_nonzero; else ++result.self_query_not_found;
        if (result.self_query_failures.size() < kSelfQueryFailureLimit)
            result.self_query_failures.push_back({expected, rank, distance});
        else
            result.self_query_failures_truncated = true;
    }
    std::sort(result.self_query_failures.begin(), result.self_query_failures.end(),
              [](const auto& a, const auto& b) { return a.id < b.id; });
}

void run_search_workload_ex(uint32_t n_queries, const std::string& gt_path,
                            const std::string& query_file, uint32_t query_offset,
                            bool self_recall, uint32_t self_recall_base_id,
                            const std::vector<uint32_t>& self_ids, uint32_t search_l) {
    if (!self_ids.empty() && (query_file.empty() || n_queries != self_ids.size())) {
        daemon_output_line("Error: --self-ids needs --query-file and --queries equal to its " +
                           std::to_string(self_ids.size()) + " ids");
        return;
    }
    std::vector<int32_t> groundtruth;
    uint32_t gt_rows = 0, gt_cols = 0;

    if (!self_recall && !gt_path.empty()) {
        // a bad file rejects the command and leaves the coordinator running
        std::vector<int32_t> gt_data;
        try {
            gt_data = ann::read_npy_groundtruth(gt_path, gt_rows, gt_cols);
        } catch (const std::exception& e) {
            daemon_output_line(std::string("Error: cannot read the ground truth: ") + e.what());
            return;
        }
        groundtruth.reserve(gt_data.size());
        for (auto v : gt_data) groundtruth.push_back(static_cast<int32_t>(v));
        daemon_output_line("Groundtruth loaded: " + std::to_string(gt_rows) + " x " + std::to_string(gt_cols));
    }

    // serving time starts here, after the ground truth is loaded
    const uint64_t serving_begin_ns = ann::ServingAccounting::now_ns();
    // alt_queries lives for the whole run because ctx.queries_data points into it
    std::vector<float> alt_queries;
    uint32_t alt_q_count = 0;
    if (!query_file.empty()) {
        uint32_t file_rows = 0, file_cols = 0;
        std::vector<float> all_q;
        try {
            all_q = ann::read_npy_float(query_file, file_rows, file_cols);
        } catch (const std::exception& e) {
            daemon_output_line(std::string("Error: cannot read the query file: ") + e.what());
            return;
        }
        if (file_cols != g_daemon.dim) {
            daemon_output_line("Error: query-file dim " + std::to_string(file_cols) +
                              " != server dim " + std::to_string(g_daemon.dim));
            return;
        }
        if (query_offset >= file_rows) {
            daemon_output_line("Error: query-offset " + std::to_string(query_offset) +
                              " >= file rows " + std::to_string(file_rows));
            return;
        }
        // Load at most the rows this command uses. n_queries is not clamped (queries repeat).
        alt_q_count = std::min(n_queries, file_rows - query_offset);
        if (!self_ids.empty() && (alt_q_count != self_ids.size() || n_queries != self_ids.size())) {
            daemon_output_line("Error: --self-ids names " + std::to_string(self_ids.size()) +
                               " ids, the query file supplies " + std::to_string(alt_q_count) + " rows");
            return;
        }
        size_t off = static_cast<size_t>(query_offset) * file_cols;
        size_t end = off + static_cast<size_t>(alt_q_count) * file_cols;
        alt_queries.assign(all_q.begin() + off, all_q.begin() + end);
        daemon_output_line("Loaded " + std::to_string(alt_q_count) + " queries from " +
                          query_file + " (offset=" + std::to_string(query_offset) + ")");
    }

    // The ground truth must cover row i % <loaded queries> and K ids for every query.
    if (!groundtruth.empty()) {
        const uint32_t loaded = alt_queries.empty() ? g_daemon.n_queries : alt_q_count;
        const uint32_t needed_rows = std::min(n_queries, loaded);
        if (gt_rows < needed_rows || gt_cols < g_daemon.K) {
            daemon_output_line("Error: ground truth " + std::to_string(gt_rows) + " x " + std::to_string(gt_cols) +
                               " does not cover the " + std::to_string(needed_rows) + " queries x K=" +
                               std::to_string(g_daemon.K) + " this command scores");
            return;
        }
    }

    // n_queries may exceed the loaded queries, which then repeat
    daemon_output_line("Running search workload: " + std::to_string(n_queries) + " queries (available unique=" + std::to_string(g_daemon.n_queries) + ")");

    UnifiedContext unified_ctx;
    unified_ctx.serving.enabled = true;
    unified_ctx.serving.begin_ns = serving_begin_ns;
    unified_ctx.serving.declared_queries = n_queries;
    unified_ctx.n_searches = n_queries;
    unified_ctx.n_available_queries = alt_queries.empty()
        ? g_daemon.n_queries : alt_q_count;
    unified_ctx.n_inserts = 0;
    unified_ctx.total_workloads = n_queries;
    unified_ctx.insert_ratio = 0;
    unified_ctx.self_recall = self_recall || !self_ids.empty();
    unified_ctx.self_recall_base_id = unified_ctx.self_recall
        ? (self_recall_base_id ? self_recall_base_id : g_daemon.n_base)
        : 0;
    unified_ctx.self_ids = self_ids;

    unified_ctx.L = search_l ? search_l : g_daemon.L;
    unified_ctx.K = g_daemon.K;
    unified_ctx.W = g_daemon.W;
    unified_ctx.effective_batch_size = g_daemon.batch_size;
    unified_ctx.dispatch_window = g_daemon.dispatch_window;
    unified_ctx.batch_per_server = g_daemon.batch_size;
    unified_ctx.p2p_mode = g_daemon.p2p_mode;
    unified_ctx.debug_mode = g_daemon.debug_mode;
    unified_ctx.debug_recall = g_daemon.debug_mode;
    unified_ctx.no_backlink = g_daemon.no_backlink;
    unified_ctx.no_broadcast = g_daemon.no_broadcast;
    unified_ctx.insert_search_only = false;
    unified_ctx.no_merge = g_daemon.no_merge;
    unified_ctx.trace_routes = g_daemon.trace_routes;

    unified_ctx.insert_R = g_daemon.insert_R;
    unified_ctx.insert_L = g_daemon.insert_L;
    unified_ctx.insert_W = g_daemon.insert_W;
    unified_ctx.insert_alpha = g_daemon.insert_alpha;
    unified_ctx.insert_start_id = 0;

    unified_ctx.dim = g_daemon.dim;
    unified_ctx.n_servers = g_daemon.n_servers;
    unified_ctx.n_base = g_daemon.n_base + g_daemon.total_inserted;
    unified_ctx.insert_threshold_id = 0;  // inserted nodes stay in the results
    unified_ctx.queries_data = alt_queries.empty()
        ? g_daemon.queries_data.data() : alt_queries.data();
    unified_ctx.insert_vectors = nullptr;
    unified_ctx.membership = g_daemon.membership.data();
    unified_ctx.membership_vec = &g_daemon.membership;
    unified_ctx.groundtruth = groundtruth.empty() ? nullptr : groundtruth.data();
    unified_ctx.gt_cols = gt_cols;

    unified_ctx.sockets = &g_daemon.sockets;
    unified_ctx.poll_items = &g_daemon.poll_items;

    unified_ctx.head_index = &g_daemon.head_index;

    unified_ctx.pending_mgr = &g_daemon.pending_mgr;

    std::vector<uint32_t> server_node_counts = get_partition_node_counts();

    std::string debug_log_path = generate_debug_log_path("q", n_queries);
    if (g_daemon.debug_mode) {
        enable_debug_logs(debug_log_path);
    } else {
        disable_partition_debug();
    }
    generate_mixed_workloads(unified_ctx);
    run_unified_workload(unified_ctx);
    print_partition_debug_stats();
    if (report_command_ended_early(unified_ctx, "search", n_queries)) return;

    std::string debug_npy_path;
    if (g_daemon.debug_mode) debug_npy_path = dump_debug_results(unified_ctx);

    uint64_t total_shared_locks = 0, total_unique_locks = 0;
    uint64_t total_shared_wait = 0, total_unique_wait = 0;
    for (size_t i = 0; i < g_daemon.n_servers; i++) {
        auto& lm = unified_ctx.server_lock_metrics[i];
        total_shared_locks += lm.shared_lock_count;
        total_unique_locks += lm.unique_lock_count;
        total_shared_wait += lm.shared_lock_wait_us;
        total_unique_wait += lm.unique_lock_wait_us;
    }
    double avg_shared_wait_total = total_shared_locks > 0 ?
        static_cast<double>(total_shared_wait) / total_shared_locks : 0.0;
    double avg_unique_wait_total = total_unique_locks > 0 ?
        static_cast<double>(total_unique_wait) / total_unique_locks : 0.0;

    ann::WorkloadResult result;
    result.timestamp = ann::get_timestamp();
    result.dataset = g_daemon.data_dir;
    result.groundtruth_file = gt_path;
    result.serving = unified_ctx.serving;
    result.mode = "search";
    result.p2p_mode = g_daemon.p2p_mode;
    copy_observability_metadata(result);
    result.queries = n_queries;
    result.search_count = n_queries;
    result.k = g_daemon.K;
    result.L = unified_ctx.L;
    result.batch_size = g_daemon.batch_size;
    result.debug_npy_path = debug_npy_path;

    result.recall = unified_ctx.avg_recall * 100;
    result.qps = unified_ctx.search_qps;
    result.avg_hops = unified_ctx.avg_hops;
    result.avg_query_latency_ms = unified_ctx.avg_latency_ms;
    result.avg_hop_latency_ms = unified_ctx.avg_hop_latency_ms;
    result.total_time_ms = unified_ctx.elapsed_ms;

    result.search_wall_ms = unified_ctx.search_wall_ms;
    result.insert_wall_ms = unified_ctx.insert_wall_ms;
    result.total_with_metrics_ms = unified_ctx.elapsed_with_metrics_ms;
    result.metric_thread_join_ms = unified_ctx.metric_thread_join_ms;
    if (unified_ctx.elapsed_with_metrics_ms > 0) {
        result.search_with_metrics_qps =
            unified_ctx.searches_completed / (unified_ctx.elapsed_with_metrics_ms / 1000.0);
    }
    if (unified_ctx.search_wall_ms > 0) {
        result.search_only_qps = unified_ctx.searches_completed / (unified_ctx.search_wall_ms / 1000.0);
    }
    if (unified_ctx.insert_wall_ms > 0) {
        result.insert_only_tps = unified_ctx.inserts_completed / (unified_ctx.insert_wall_ms / 1000.0);
    }

    result.insert_count = g_daemon.total_inserted;

    result.workloads_generated = unified_ctx.workloads.size();
    result.search_phase_completed = unified_ctx.insert_search_phase_completed;
    result.insert_rpc_completed = unified_ctx.insert_rpc_phase_completed;
    result.backlink_phase_completed = unified_ctx.backlink_phase_completed;
    result.broadcast_phase_completed = unified_ctx.broadcast_phase_completed;

    result.search_phase_ms = unified_ctx.total_search_phase_us / 1000.0;
    result.insert_rpc_ms = unified_ctx.total_insert_rpc_us / 1000.0;
    result.backlink_ms = unified_ctx.total_backlink_us / 1000.0;
    result.broadcast_ms = unified_ctx.total_broadcast_us / 1000.0;

    result.beam_total_entries = unified_ctx.beam_total_entries;
    result.beam_inserted_entries = unified_ctx.beam_inserted_entries;
    result.result_total_entries = unified_ctx.result_total_entries;
    result.result_inserted_entries = unified_ctx.result_inserted_entries;

    result.server_total_nodes = server_node_counts;

    result.total_shared_locks = total_shared_locks;
    result.total_unique_locks = total_unique_locks;
    result.avg_unique_lock_wait_us = avg_unique_wait_total;
    result.avg_shared_lock_wait_us = avg_shared_wait_total;
    for (size_t i = 0; i < g_daemon.n_servers; i++) {
        auto& lm = unified_ctx.server_lock_metrics[i];
        result.per_server_lock_stats.emplace_back(
            lm.shared_lock_count, lm.unique_lock_count,
            lm.avg_shared_wait_us(), lm.avg_unique_wait_us()
        );
    }

    populate_detailed_metrics(unified_ctx, result);

    populate_debug_metrics(result);

    query_server_lock_metrics(result);

    copy_network_baseline(result);

    // computed after the timed part
    if (g_daemon.result_digest) populate_search_result_digest(unified_ctx, result);
    populate_self_query(unified_ctx, result);

    daemon_output_line("===JSON_RESULT===");
    daemon_output_line(ann::to_json(result));
    daemon_output_line("===JSON_END===");
}

// Rows [start_id, start_id + count) of an insert file, with count clipped to the file.
static bool load_insert_vectors(const std::string& insert_file, uint32_t start_id, uint32_t& count,
                                std::vector<float>& vectors, std::string& error) {
    if (insert_file.empty()) {
        error = "inserts need an insert file (insert ... --file PATH, mixed ... --insert-file PATH)";
        return false;
    }
    uint32_t file_rows = 0, file_cols = 0;
    std::vector<float> all_vectors;
    try {
        all_vectors = ann::read_npy_float(insert_file, file_rows, file_cols);
    } catch (const std::exception& e) {
        error = std::string("cannot read the insert file: ") + e.what();
        return false;
    }
    if (file_cols != g_daemon.dim) {
        error = "insert file dimension " + std::to_string(file_cols) + " != index dimension " +
                std::to_string(g_daemon.dim);
        return false;
    }
    if (start_id >= file_rows) {
        error = "insert start id " + std::to_string(start_id) + " is beyond the file's " +
                std::to_string(file_rows) + " rows";
        return false;
    }
    count = std::min(count, file_rows - start_id);
    const size_t start_offset = static_cast<size_t>(start_id) * file_cols;
    const size_t end_offset = start_offset + static_cast<size_t>(count) * file_cols;
    vectors.assign(all_vectors.begin() + start_offset, all_vectors.begin() + end_offset);
    return true;
}

void run_insert_workload(uint32_t n_insert, const std::string& insert_file,
                         uint32_t start_id) {
    daemon_output_line("Running insert workload: " + std::to_string(n_insert) + " vectors");

    std::vector<float> insert_vectors;
    {
        std::string error;
        if (!load_insert_vectors(insert_file, start_id, n_insert, insert_vectors, error)) {
            daemon_output_line("Error: " + error);
            return;
        }
        daemon_output_line("Loaded " + std::to_string(n_insert) + " vectors from " + insert_file);
    }

    UnifiedContext unified_ctx;
    unified_ctx.n_searches = 0;
    unified_ctx.n_inserts = n_insert;
    unified_ctx.total_workloads = n_insert;
    unified_ctx.insert_ratio = 100;

    unified_ctx.L = g_daemon.L;
    unified_ctx.K = g_daemon.K;
    unified_ctx.W = g_daemon.W;
    unified_ctx.effective_batch_size = g_daemon.batch_size;
    unified_ctx.dispatch_window = g_daemon.dispatch_window;
    unified_ctx.batch_per_server = g_daemon.batch_size;
    unified_ctx.p2p_mode = g_daemon.p2p_mode;
    unified_ctx.debug_mode = g_daemon.debug_mode;
    unified_ctx.debug_recall = false;
    unified_ctx.no_backlink = g_daemon.no_backlink;
    unified_ctx.no_broadcast = g_daemon.no_broadcast;
    unified_ctx.insert_search_only = g_daemon.insert_search_only;
    unified_ctx.no_merge = g_daemon.no_merge;
    unified_ctx.trace_routes = g_daemon.trace_routes;

    unified_ctx.insert_R = g_daemon.insert_R;
    unified_ctx.insert_L = g_daemon.insert_L;
    unified_ctx.insert_W = g_daemon.insert_W;
    unified_ctx.insert_alpha = g_daemon.insert_alpha;
    unified_ctx.insert_start_id = g_daemon.total_inserted;

    unified_ctx.dim = g_daemon.dim;
    unified_ctx.n_servers = g_daemon.n_servers;
    unified_ctx.n_base = g_daemon.n_base + g_daemon.total_inserted;
    unified_ctx.insert_threshold_id = g_daemon.n_base;
    unified_ctx.queries_data = nullptr;
    unified_ctx.insert_vectors = insert_vectors.data();
    unified_ctx.membership = g_daemon.membership.data();
    unified_ctx.membership_vec = &g_daemon.membership;
    unified_ctx.groundtruth = nullptr;
    unified_ctx.gt_cols = 0;

    unified_ctx.sockets = &g_daemon.sockets;
    unified_ctx.poll_items = &g_daemon.poll_items;

    unified_ctx.head_index = &g_daemon.head_index;

    unified_ctx.pending_mgr = &g_daemon.pending_mgr;

    std::string debug_log_path = generate_debug_log_path("insert", n_insert);
    if (g_daemon.debug_mode) {
        enable_debug_logs(debug_log_path);
    } else {
        disable_partition_debug();
    }
    // reserve() may reallocate: refresh unified_ctx.membership after it.
    {
        size_t expected_max = static_cast<size_t>(g_daemon.n_base) +
                              g_daemon.total_inserted + n_insert + 1;
        if (g_daemon.membership.capacity() < expected_max) {
            g_daemon.membership.reserve(expected_max);
            unified_ctx.membership = g_daemon.membership.data();
        }
    }
    generate_mixed_workloads(unified_ctx);
    run_unified_workload(unified_ctx);
    print_partition_debug_stats();

    // --search-only leaves the graph unchanged
    if (!g_daemon.insert_search_only) {
        g_daemon.total_inserted += n_insert;
        g_daemon.all_inserted_vectors.insert(g_daemon.all_inserted_vectors.end(),
                                             insert_vectors.begin(), insert_vectors.end());
    }
    if (report_command_ended_early(unified_ctx, "insert")) return;

    uint64_t total_shared_locks = 0, total_unique_locks = 0;
    uint64_t total_shared_wait = 0, total_unique_wait = 0;
    for (size_t i = 0; i < g_daemon.n_servers; i++) {
        auto& lm = unified_ctx.server_lock_metrics[i];
        total_shared_locks += lm.shared_lock_count;
        total_unique_locks += lm.unique_lock_count;
        total_shared_wait += lm.shared_lock_wait_us;
        total_unique_wait += lm.unique_lock_wait_us;
    }
    double avg_shared_wait_total = total_shared_locks > 0 ?
        static_cast<double>(total_shared_wait) / total_shared_locks : 0.0;
    double avg_unique_wait_total = total_unique_locks > 0 ?
        static_cast<double>(total_unique_wait) / total_unique_locks : 0.0;

    ann::WorkloadResult result;
    result.timestamp = ann::get_timestamp();
    result.dataset = g_daemon.data_dir;
    result.mode = "insert";
    result.p2p_mode = g_daemon.p2p_mode;
    copy_observability_metadata(result, g_daemon.insert_search_only);
    result.queries = 0;
    result.search_count = 0;
    result.k = g_daemon.K;
    result.L = g_daemon.L;
    result.batch_size = g_daemon.batch_size;

    result.insert_count = n_insert;
    result.insert_throughput = unified_ctx.insert_throughput;
    result.avg_backlinks = n_insert > 0 ? static_cast<double>(unified_ctx.total_backlinks_sent) / n_insert : 0;
    result.total_backlinks = unified_ctx.total_backlinks_sent;
    result.total_broadcasts = unified_ctx.total_broadcasts_acked;
    result.total_time_ms = unified_ctx.elapsed_ms;

    result.search_wall_ms = unified_ctx.search_wall_ms;
    result.insert_wall_ms = unified_ctx.insert_wall_ms;
    result.total_with_metrics_ms = unified_ctx.elapsed_with_metrics_ms;
    result.metric_thread_join_ms = unified_ctx.metric_thread_join_ms;
    if (unified_ctx.elapsed_with_metrics_ms > 0) {
        result.search_with_metrics_qps =
            unified_ctx.searches_completed / (unified_ctx.elapsed_with_metrics_ms / 1000.0);
    }
    if (unified_ctx.search_wall_ms > 0) {
        result.search_only_qps = unified_ctx.searches_completed / (unified_ctx.search_wall_ms / 1000.0);
    }
    if (unified_ctx.insert_wall_ms > 0) {
        result.insert_only_tps = unified_ctx.inserts_completed / (unified_ctx.insert_wall_ms / 1000.0);
    }

    result.workloads_generated = unified_ctx.workloads.size();
    result.search_phase_completed = unified_ctx.insert_search_phase_completed;
    result.insert_rpc_completed = unified_ctx.insert_rpc_phase_completed;
    result.backlink_phase_completed = unified_ctx.backlink_phase_completed;
    result.broadcast_phase_completed = unified_ctx.broadcast_phase_completed;

    result.search_phase_ms = unified_ctx.total_search_phase_us / 1000.0;
    result.insert_rpc_ms = unified_ctx.total_insert_rpc_us / 1000.0;
    result.backlink_ms = unified_ctx.total_backlink_us / 1000.0;
    result.broadcast_ms = unified_ctx.total_broadcast_us / 1000.0;

    // per-insert averages
    if (n_insert > 0) {
        result.per_insert_search_ms = result.search_phase_ms / n_insert;
        result.per_insert_rpc_ms = result.insert_rpc_ms / n_insert;
        result.per_insert_backlink_ms = result.backlink_ms / n_insert;
        result.per_insert_broadcast_ms = result.broadcast_ms / n_insert;
        result.per_insert_total_ms = result.per_insert_search_ms + result.per_insert_rpc_ms +
                                     result.per_insert_backlink_ms + result.per_insert_broadcast_ms;
    }

    result.total_shared_locks = total_shared_locks;
    result.total_unique_locks = total_unique_locks;
    result.avg_unique_lock_wait_us = avg_unique_wait_total;
    result.avg_shared_lock_wait_us = avg_shared_wait_total;
    for (size_t i = 0; i < g_daemon.n_servers; i++) {
        auto& lm = unified_ctx.server_lock_metrics[i];
        result.per_server_lock_stats.emplace_back(
            lm.shared_lock_count, lm.unique_lock_count,
            lm.avg_shared_wait_us(), lm.avg_unique_wait_us()
        );
    }

    result.server_total_nodes = get_partition_node_counts();

    // lock metrics first: populate_detailed_metrics reads them
    query_server_lock_metrics(result);

    populate_detailed_metrics(unified_ctx, result);

    populate_debug_metrics(result);

    // per-insert lock wait
    if (n_insert > 0) {
        result.per_insert_alloc_lock_wait_ms = result.avg_insert_alloc_lock_wait_us / 1000.0;
        result.per_insert_edge_lock_wait_ms = result.avg_insert_edge_lock_wait_us / 1000.0;
        double avg_backlinks = unified_ctx.total_backlinks_sent > 0 ?
            static_cast<double>(unified_ctx.total_backlinks_sent) / n_insert : 0.0;
        result.per_insert_backlink_lock_wait_ms = (result.avg_backlink_lock_wait_us * avg_backlinks) / 1000.0;
        result.per_insert_broadcast_lock_wait_ms = (result.avg_broadcast_lock_wait_us * g_daemon.n_servers) / 1000.0;
        result.per_insert_total_lock_wait_ms = result.per_insert_alloc_lock_wait_ms +
                                               result.per_insert_edge_lock_wait_ms +
                                               result.per_insert_backlink_lock_wait_ms +
                                               result.per_insert_broadcast_lock_wait_ms;

        // per-request INSERT timing from the servers
        result.per_insert_rpc_queue_wait_ms = (unified_ctx.total_insert_queue_wait_us / n_insert) / 1000.0;
        result.per_insert_rpc_deserialize_ms = (unified_ctx.total_insert_deserialize_us / n_insert) / 1000.0;
        result.per_insert_rpc_alloc_lock_ms = (unified_ctx.total_insert_alloc_lock_us / n_insert) / 1000.0;
        result.per_insert_rpc_edge_lock_ms = (unified_ctx.total_insert_edge_lock_us / n_insert) / 1000.0;
        result.per_insert_rpc_prune_ms = (unified_ctx.total_insert_prune_us / n_insert) / 1000.0;
        result.per_insert_rpc_alloc_memory_ms = (unified_ctx.total_insert_alloc_memory_us / n_insert) / 1000.0;
        result.resize_full_base.count = unified_ctx.resize_full_base.count;
        result.resize_full_base.total_ms = unified_ctx.resize_full_base.time_us / 1000.0;
        result.resize_membership.count = unified_ctx.resize_membership.count;
        result.resize_membership.total_ms = unified_ctx.resize_membership.time_us / 1000.0;
        result.resize_adj.count = unified_ctx.resize_adj.count;
        result.resize_adj.total_ms = unified_ctx.resize_adj.time_us / 1000.0;
        result.resize_global_to_local.count = unified_ctx.resize_global_to_local.count;
        result.resize_global_to_local.total_ms = unified_ctx.resize_global_to_local.time_us / 1000.0;
        result.num_servers = static_cast<int>(g_daemon.n_servers);
        for (int s = 0; s < result.num_servers && s < ann::WorkloadResult::MAX_SERVERS; s++) {
            result.resize_per_server[s].full_base.count = unified_ctx.resize_per_server[s].full_base.count;
            result.resize_per_server[s].full_base.total_ms = unified_ctx.resize_per_server[s].full_base.time_us / 1000.0;
            result.resize_per_server[s].membership.count = unified_ctx.resize_per_server[s].membership.count;
            result.resize_per_server[s].membership.total_ms = unified_ctx.resize_per_server[s].membership.time_us / 1000.0;
            result.resize_per_server[s].adj.count = unified_ctx.resize_per_server[s].adj.count;
            result.resize_per_server[s].adj.total_ms = unified_ctx.resize_per_server[s].adj.time_us / 1000.0;
            result.resize_per_server[s].global_to_local.count = unified_ctx.resize_per_server[s].global_to_local.count;
            result.resize_per_server[s].global_to_local.total_ms = unified_ctx.resize_per_server[s].global_to_local.time_us / 1000.0;
        }
        result.per_insert_rpc_edge_memory_ms = (unified_ctx.total_insert_edge_memory_us / n_insert) / 1000.0;
        result.per_insert_rpc_local_backlink_prune_ms = (unified_ctx.total_insert_local_backlink_prune_us / n_insert) / 1000.0;
        result.per_insert_rpc_network_rtt_ms = (unified_ctx.total_insert_network_rtt_us / n_insert) / 1000.0;
    }

    copy_network_baseline(result);

    daemon_output_line("===JSON_RESULT===");
    daemon_output_line(ann::to_json(result));
    daemon_output_line("===JSON_END===");
}

void run_mixed_workload(uint32_t n_queries, uint32_t insert_ratio, const std::string& gt_path,
                        const std::string& insert_gt_path,
                        const std::string& insert_file, uint32_t insert_start_id,
                        double inserted_query_ratio) {
    uint32_t n_insert = (n_queries * insert_ratio) / 100;
    uint32_t n_search = n_queries - n_insert;
    // at least one of each
    if (insert_ratio > 0 && insert_ratio < 100) {
        if (n_insert == 0) n_insert = 1;
        if (n_search == 0) n_search = 1;
    }

    daemon_output_line("Running mixed workload: " + std::to_string(n_search) + " searches + "
                       + std::to_string(n_insert) + " inserts (ratio=" + std::to_string(insert_ratio) + "%)");

    std::vector<float> insert_vectors;
    if (n_insert > 0) {
        std::string error;
        if (!load_insert_vectors(insert_file, insert_start_id, n_insert, insert_vectors, error)) {
            daemon_output_line("Error: " + error);
            return;
        }
        daemon_output_line("Loaded " + std::to_string(n_insert) + " vectors from " + insert_file
                           + " (start_id=" + std::to_string(insert_start_id) + ")");
    }

    // searches cycle through the loaded query.npy rows
    if (n_search > 0 && g_daemon.n_queries == 0) {
        daemon_output_line("Error: mixed needs queries: the coordinator loaded no query.npy rows");
        return;
    }

    // The search ground truth must cover every search, as for recall.
    uint32_t gt_rows = 0, gt_cols = 0;
    std::vector<int32_t> groundtruth;
    if (n_search > 0 && !gt_path.empty()) {
        try {
            auto gt_data = ann::read_npy_groundtruth(gt_path, gt_rows, gt_cols);
            groundtruth.reserve(gt_data.size());
            for (auto v : gt_data) groundtruth.push_back(static_cast<int32_t>(v));
            daemon_output_line("Loaded groundtruth: " + std::to_string(gt_rows) + " x " + std::to_string(gt_cols));
        } catch (const std::exception& e) {
            daemon_output_line(std::string("Error: cannot read the ground truth: ") + e.what());
            return;
        }
        const uint32_t needed_rows = std::min(n_search, g_daemon.n_queries);
        if (gt_rows < needed_rows || gt_cols < g_daemon.K) {
            daemon_output_line("Error: ground truth " + std::to_string(gt_rows) + " x " + std::to_string(gt_cols) +
                               " does not cover the " + std::to_string(needed_rows) + " queries x K=" +
                               std::to_string(g_daemon.K) + " this command scores");
            return;
        }
    }

    // every row of the insert ground truth needs K ids
    std::vector<int32_t> insert_groundtruth;
    uint32_t insert_gt_rows = 0, insert_gt_cols = 0;
    bool enable_inserted_query = false;
    if (!insert_gt_path.empty()) {
        try {
            auto gt_data = ann::read_npy_groundtruth(insert_gt_path, insert_gt_rows, insert_gt_cols);
            insert_groundtruth.reserve(gt_data.size());
            for (auto v : gt_data) insert_groundtruth.push_back(static_cast<int32_t>(v));
            enable_inserted_query = true;
            daemon_output_line("Loaded insert GT: " + std::to_string(insert_gt_rows) + " x " + std::to_string(insert_gt_cols));
        } catch (const std::exception& e) {
            daemon_output_line(std::string("Error: cannot read the insert ground truth: ") + e.what());
            return;
        }
        if (insert_gt_cols < g_daemon.K) {
            daemon_output_line("Error: insert ground truth " + std::to_string(insert_gt_rows) + " x " +
                               std::to_string(insert_gt_cols) + " has fewer than K=" + std::to_string(g_daemon.K) +
                               " ids per row");
            return;
        }
    }

    UnifiedContext unified_ctx;
    unified_ctx.n_searches = n_search;
    // searches beyond the loaded queries cycle through them
    unified_ctx.n_available_queries = g_daemon.n_queries;
    unified_ctx.n_inserts = n_insert;
    unified_ctx.total_workloads = n_search + n_insert;
    unified_ctx.insert_ratio = insert_ratio;

    unified_ctx.L = g_daemon.L;
    unified_ctx.K = g_daemon.K;
    unified_ctx.W = g_daemon.W;
    unified_ctx.effective_batch_size = g_daemon.batch_size;
    unified_ctx.dispatch_window = g_daemon.dispatch_window;
    unified_ctx.batch_per_server = g_daemon.batch_size;
    unified_ctx.p2p_mode = g_daemon.p2p_mode;
    unified_ctx.debug_mode = g_daemon.debug_mode;
    unified_ctx.debug_recall = (g_daemon.debug_mode && n_search > 0 && !groundtruth.empty());
    unified_ctx.no_backlink = g_daemon.no_backlink;
    unified_ctx.no_broadcast = g_daemon.no_broadcast;
    unified_ctx.insert_search_only = false;
    unified_ctx.no_merge = g_daemon.no_merge;

    unified_ctx.insert_R = g_daemon.insert_R;
    unified_ctx.insert_L = g_daemon.insert_L;
    unified_ctx.insert_W = g_daemon.insert_W;
    unified_ctx.insert_alpha = g_daemon.insert_alpha;
    unified_ctx.insert_start_id = g_daemon.total_inserted;

    unified_ctx.dim = g_daemon.dim;
    unified_ctx.n_servers = g_daemon.n_servers;
    unified_ctx.n_base = g_daemon.n_base + g_daemon.total_inserted;
    unified_ctx.insert_threshold_id = g_daemon.n_base;
    unified_ctx.queries_data = n_search > 0 ? g_daemon.queries_data.data() : nullptr;
    unified_ctx.insert_vectors = n_insert > 0 ? insert_vectors.data() : nullptr;
    unified_ctx.membership = g_daemon.membership.data();
    unified_ctx.membership_vec = &g_daemon.membership;
    unified_ctx.groundtruth = groundtruth.empty() ? nullptr : groundtruth.data();
    unified_ctx.gt_cols = gt_cols;

    unified_ctx.insert_groundtruth = std::move(insert_groundtruth);
    unified_ctx.insert_gt_rows = insert_gt_rows;
    unified_ctx.insert_gt_cols = insert_gt_cols;
    unified_ctx.enable_inserted_query = enable_inserted_query;
    unified_ctx.inserted_query_ratio = inserted_query_ratio;

    unified_ctx.sockets = &g_daemon.sockets;
    unified_ctx.poll_items = &g_daemon.poll_items;

    unified_ctx.head_index = &g_daemon.head_index;

    unified_ctx.pending_mgr = &g_daemon.pending_mgr;

    std::string debug_log_path = generate_debug_log_path("mixed" + std::to_string(insert_ratio) + "pct_q", n_queries);
    if (g_daemon.debug_mode) {
        enable_debug_logs(debug_log_path);
    } else {
        disable_partition_debug();
    }
    // reserve() may reallocate: refresh unified_ctx.membership after it.
    if (n_insert > 0) {
        size_t expected_max = static_cast<size_t>(g_daemon.n_base) +
                              g_daemon.total_inserted + n_insert + 1;
        if (g_daemon.membership.capacity() < expected_max) {
            g_daemon.membership.reserve(expected_max);
            unified_ctx.membership = g_daemon.membership.data();
        }
    }
    generate_mixed_workloads(unified_ctx);
    run_unified_workload(unified_ctx);
    print_partition_debug_stats();

    std::string debug_npy_path;
    if (g_daemon.debug_mode && n_search > 0) debug_npy_path = dump_debug_results(unified_ctx);

    if (n_insert > 0) {
        g_daemon.total_inserted += n_insert;
        g_daemon.all_inserted_vectors.insert(g_daemon.all_inserted_vectors.end(),
                                             insert_vectors.begin(), insert_vectors.end());
    }
    if (report_command_ended_early(unified_ctx, "mixed")) return;

    uint64_t total_shared_locks = 0, total_unique_locks = 0;
    uint64_t total_shared_wait = 0, total_unique_wait = 0;
    for (size_t i = 0; i < g_daemon.n_servers; i++) {
        auto& lm = unified_ctx.server_lock_metrics[i];
        total_shared_locks += lm.shared_lock_count;
        total_unique_locks += lm.unique_lock_count;
        total_shared_wait += lm.shared_lock_wait_us;
        total_unique_wait += lm.unique_lock_wait_us;
    }
    double avg_shared_wait_total = total_shared_locks > 0 ?
        static_cast<double>(total_shared_wait) / total_shared_locks : 0.0;
    double avg_unique_wait_total = total_unique_locks > 0 ?
        static_cast<double>(total_unique_wait) / total_unique_locks : 0.0;

    ann::WorkloadResult result;
    result.timestamp = ann::get_timestamp();
    result.dataset = g_daemon.data_dir;
    result.groundtruth_file = gt_path;
    result.mode = "mixed";
    result.p2p_mode = g_daemon.p2p_mode;
    copy_observability_metadata(result);
    result.queries = n_queries;
    result.search_count = n_search;
    result.k = g_daemon.K;
    result.L = g_daemon.L;
    result.batch_size = g_daemon.batch_size;
    result.debug_npy_path = debug_npy_path;

    if (n_search > 0) {
        result.recall = unified_ctx.avg_recall * 100;
        result.qps = unified_ctx.search_qps;
        result.avg_hops = unified_ctx.avg_hops;
        result.avg_query_latency_ms = unified_ctx.avg_latency_ms;
        result.avg_hop_latency_ms = unified_ctx.avg_hop_latency_ms;
    }

    result.insert_count = n_insert;
    if (n_insert > 0) {
        result.insert_throughput = unified_ctx.insert_throughput;
        result.avg_backlinks = static_cast<double>(unified_ctx.total_backlinks_sent) / n_insert;
        result.total_backlinks = unified_ctx.total_backlinks_sent;
        result.total_broadcasts = unified_ctx.total_broadcasts_acked;
    }
    result.total_time_ms = unified_ctx.elapsed_ms;

    result.search_wall_ms = unified_ctx.search_wall_ms;
    result.insert_wall_ms = unified_ctx.insert_wall_ms;
    result.total_with_metrics_ms = unified_ctx.elapsed_with_metrics_ms;
    result.metric_thread_join_ms = unified_ctx.metric_thread_join_ms;
    if (unified_ctx.elapsed_with_metrics_ms > 0) {
        result.search_with_metrics_qps =
            unified_ctx.searches_completed / (unified_ctx.elapsed_with_metrics_ms / 1000.0);
    }
    if (unified_ctx.search_wall_ms > 0) {
        result.search_only_qps = unified_ctx.searches_completed / (unified_ctx.search_wall_ms / 1000.0);
    }
    if (unified_ctx.insert_wall_ms > 0) {
        result.insert_only_tps = unified_ctx.inserts_completed / (unified_ctx.insert_wall_ms / 1000.0);
    }

    result.workloads_generated = unified_ctx.workloads.size();
    result.search_phase_completed = unified_ctx.insert_search_phase_completed;
    result.insert_rpc_completed = unified_ctx.insert_rpc_phase_completed;
    result.backlink_phase_completed = unified_ctx.backlink_phase_completed;
    result.broadcast_phase_completed = unified_ctx.broadcast_phase_completed;

    result.search_phase_ms = unified_ctx.total_search_phase_us / 1000.0;
    result.insert_rpc_ms = unified_ctx.total_insert_rpc_us / 1000.0;
    result.backlink_ms = unified_ctx.total_backlink_us / 1000.0;
    result.broadcast_ms = unified_ctx.total_broadcast_us / 1000.0;

    // per-insert averages
    if (n_insert > 0) {
        result.per_insert_search_ms = result.search_phase_ms / n_insert;
        result.per_insert_rpc_ms = result.insert_rpc_ms / n_insert;
        result.per_insert_backlink_ms = result.backlink_ms / n_insert;
        result.per_insert_broadcast_ms = result.broadcast_ms / n_insert;
        result.per_insert_total_ms = result.per_insert_search_ms + result.per_insert_rpc_ms +
                                     result.per_insert_backlink_ms + result.per_insert_broadcast_ms;
    }

    result.total_shared_locks = total_shared_locks;
    result.total_unique_locks = total_unique_locks;
    result.avg_unique_lock_wait_us = avg_unique_wait_total;
    result.avg_shared_lock_wait_us = avg_shared_wait_total;
    for (size_t i = 0; i < g_daemon.n_servers; i++) {
        auto& lm = unified_ctx.server_lock_metrics[i];
        result.per_server_lock_stats.emplace_back(
            lm.shared_lock_count, lm.unique_lock_count,
            lm.avg_shared_wait_us(), lm.avg_unique_wait_us()
        );
    }

    result.server_total_nodes = get_partition_node_counts();

    // lock metrics first: populate_detailed_metrics reads them
    query_server_lock_metrics(result);

    populate_detailed_metrics(unified_ctx, result);

    populate_debug_metrics(result);

    // per-insert lock wait
    if (n_insert > 0) {
        result.per_insert_alloc_lock_wait_ms = result.avg_insert_alloc_lock_wait_us / 1000.0;
        result.per_insert_edge_lock_wait_ms = result.avg_insert_edge_lock_wait_us / 1000.0;
        double avg_backlinks = unified_ctx.total_backlinks_sent > 0 ?
            static_cast<double>(unified_ctx.total_backlinks_sent) / n_insert : 0.0;
        result.per_insert_backlink_lock_wait_ms = (result.avg_backlink_lock_wait_us * avg_backlinks) / 1000.0;
        result.per_insert_broadcast_lock_wait_ms = (result.avg_broadcast_lock_wait_us * g_daemon.n_servers) / 1000.0;
        result.per_insert_total_lock_wait_ms = result.per_insert_alloc_lock_wait_ms +
                                               result.per_insert_edge_lock_wait_ms +
                                               result.per_insert_backlink_lock_wait_ms +
                                               result.per_insert_broadcast_lock_wait_ms;

        // per-request INSERT timing from the servers
        result.per_insert_rpc_queue_wait_ms = (unified_ctx.total_insert_queue_wait_us / n_insert) / 1000.0;
        result.per_insert_rpc_deserialize_ms = (unified_ctx.total_insert_deserialize_us / n_insert) / 1000.0;
        result.per_insert_rpc_alloc_lock_ms = (unified_ctx.total_insert_alloc_lock_us / n_insert) / 1000.0;
        result.per_insert_rpc_edge_lock_ms = (unified_ctx.total_insert_edge_lock_us / n_insert) / 1000.0;
        result.per_insert_rpc_prune_ms = (unified_ctx.total_insert_prune_us / n_insert) / 1000.0;
        result.per_insert_rpc_alloc_memory_ms = (unified_ctx.total_insert_alloc_memory_us / n_insert) / 1000.0;
        result.resize_full_base.count = unified_ctx.resize_full_base.count;
        result.resize_full_base.total_ms = unified_ctx.resize_full_base.time_us / 1000.0;
        result.resize_membership.count = unified_ctx.resize_membership.count;
        result.resize_membership.total_ms = unified_ctx.resize_membership.time_us / 1000.0;
        result.resize_adj.count = unified_ctx.resize_adj.count;
        result.resize_adj.total_ms = unified_ctx.resize_adj.time_us / 1000.0;
        result.resize_global_to_local.count = unified_ctx.resize_global_to_local.count;
        result.resize_global_to_local.total_ms = unified_ctx.resize_global_to_local.time_us / 1000.0;
        result.num_servers = static_cast<int>(g_daemon.n_servers);
        for (int s = 0; s < result.num_servers && s < ann::WorkloadResult::MAX_SERVERS; s++) {
            result.resize_per_server[s].full_base.count = unified_ctx.resize_per_server[s].full_base.count;
            result.resize_per_server[s].full_base.total_ms = unified_ctx.resize_per_server[s].full_base.time_us / 1000.0;
            result.resize_per_server[s].membership.count = unified_ctx.resize_per_server[s].membership.count;
            result.resize_per_server[s].membership.total_ms = unified_ctx.resize_per_server[s].membership.time_us / 1000.0;
            result.resize_per_server[s].adj.count = unified_ctx.resize_per_server[s].adj.count;
            result.resize_per_server[s].adj.total_ms = unified_ctx.resize_per_server[s].adj.time_us / 1000.0;
            result.resize_per_server[s].global_to_local.count = unified_ctx.resize_per_server[s].global_to_local.count;
            result.resize_per_server[s].global_to_local.total_ms = unified_ctx.resize_per_server[s].global_to_local.time_us / 1000.0;
        }
        result.per_insert_rpc_edge_memory_ms = (unified_ctx.total_insert_edge_memory_us / n_insert) / 1000.0;
        result.per_insert_rpc_local_backlink_prune_ms = (unified_ctx.total_insert_local_backlink_prune_us / n_insert) / 1000.0;
        result.per_insert_rpc_network_rtt_ms = (unified_ctx.total_insert_network_rtt_us / n_insert) / 1000.0;
    }

    if (unified_ctx.inserted_queries_completed > 0) {
        result.inserted_query.count = unified_ctx.inserted_queries_completed;
        result.inserted_query.recall =
            (unified_ctx.inserted_recall_sum / unified_ctx.inserted_queries_completed) * 100;
        result.inserted_query.baseline_count = unified_ctx.baseline_queries_completed;
    }

    copy_network_baseline(result);

    // computed after the timed part, over the searches
    if (g_daemon.result_digest) populate_search_result_digest(unified_ctx, result);

    daemon_output_line("===JSON_RESULT===");
    daemon_output_line(ann::to_json(result));
    daemon_output_line("===JSON_END===");
}

} // namespace tide
