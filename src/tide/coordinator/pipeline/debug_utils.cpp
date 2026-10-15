/**
 * Debug utilities of the unified pipeline.
 *
 * Contains:
 * - Crash handlers for debugging
 * - Memory usage tracking
 * - Partition debug logging
 */

#include "pipeline_types.h"
#include "metrics/experiment_result.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include <cstring>

// Signal handling for crash debugging
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

namespace tide {

// Global state for crash reporting
std::atomic<uint32_t> g_last_completed{0};
std::atomic<uint32_t> g_total_workloads{0};
std::atomic<uint32_t> g_in_flight_search{0};
std::atomic<uint32_t> g_in_flight_insert{0};
std::atomic<size_t> g_peak_memory_kb{0};

// Global debug state
DebugState g_debug;


void crash_handler(int sig) {
    const char* sig_name = (sig == SIGSEGV) ? "SIGSEGV" :
                           (sig == SIGABRT) ? "SIGABRT" :
                           (sig == SIGBUS) ? "SIGBUS" :
                           (sig == SIGFPE) ? "SIGFPE" : "UNKNOWN";

    fprintf(stderr, "\n=== CRASH: Signal %d (%s) ===\n", sig, sig_name);
    fprintf(stderr, "Progress: %u/%u completed\n",
            g_last_completed.load(), g_total_workloads.load());
    fprintf(stderr, "In-flight: Search=%u, Insert=%u\n",
            g_in_flight_search.load(), g_in_flight_insert.load());

    // Get backtrace
    void* array[50];
    int size = backtrace(array, 50);
    fprintf(stderr, "Backtrace (%d frames):\n", size);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    fflush(stderr);

    signal(sig, SIG_DFL);
    raise(sig);
}

void install_crash_handlers() {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
}

// Memory usage tracking (Linux-specific)
size_t get_memory_usage_kb() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;

    char line[128];
    size_t vm_rss = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &vm_rss);
            break;
        }
    }
    fclose(file);
    return vm_rss;  // in KB
}

// Helper: the owner of a node ID, load-time or inserted
uint32_t get_node_partition(const UnifiedContext& ctx, uint32_t node_id) {
    if (ctx.membership_vec && node_id < ctx.membership_vec->size()) {
        return owner_label_checked((*ctx.membership_vec)[node_id], ctx.n_servers, "node partition");
    } else if (ctx.membership && node_id < ctx.n_base) {
        return owner_label_checked(ctx.membership[node_id], ctx.n_servers, "node partition");
    }
    return 0;
}

std::vector<uint32_t> owner_labels(const UnifiedContext& ctx, const std::vector<uint32_t>& ids) {
    std::vector<uint32_t> labels;
    labels.reserve(ids.size());
    for (const uint32_t id : ids) labels.push_back(get_node_partition(ctx, id));
    return labels;
}

std::vector<uint32_t> owner_labels(const UnifiedContext& ctx, const std::vector<ann::BeamEntry>& beam) {
    std::vector<uint32_t> labels;
    labels.reserve(beam.size());
    for (const auto& entry : beam) labels.push_back(get_node_partition(ctx, entry.node_id));
    return labels;
}

std::string debug_log_header(uint32_t n_servers) {
    std::ostringstream header;
    header << "type,id,entry_partition,final_partition,hops,";
    for (uint32_t i = 0; i < 10; i++) header << "nb" << i << "_id,nb" << i << "_part,nb" << i << "_dist,";
    for (uint32_t p = 0; p < n_servers; p++) header << "topK_p" << p << ",";
    for (uint32_t i = 0; i < 10; i++) header << "gt" << i << "_id,";
    header << "recall_hits\n";
    return header.str();
}

std::string debug_hop_header(uint32_t n_servers) {
    std::ostringstream header;
    header << "type,id,hop,cur_part,next_part,beam_size,";
    for (uint32_t p = 0; p < n_servers; p++) header << "beam_p" << p << ",";
    header << "top1_id,top1_part,top1_dist\n";
    return header.str();
}

void open_debug_log(const std::string& path, uint32_t n_servers) {
    g_debug.log_file.open(path);
    g_debug.n_servers = n_servers;
    if (g_debug.log_file.is_open()) {
        g_debug.log_file << debug_log_header(n_servers);
        g_debug.log_file.flush();
    }

    std::string hop_path = path.substr(0, path.rfind('.')) + "_hops.csv";
    g_debug.hop_file.open(hop_path);
    if (g_debug.hop_file.is_open()) {
        g_debug.hop_file << debug_hop_header(n_servers);
        g_debug.hop_file.flush();
        std::cout << "Hop log opened: " << hop_path << "\n";
    }
}

void close_debug_log() {
    if (g_debug.log_file.is_open()) g_debug.log_file.close();
    if (g_debug.hop_file.is_open()) g_debug.hop_file.close();
}

// =============================================================================
// log_hop_state - one line per hop; labels[i] is the owner of beam[i]
// =============================================================================
void log_hop_state(const char* type, uint32_t id, uint32_t hop_num,
                   uint32_t current_partition, uint32_t next_partition,
                   const std::vector<ann::BeamEntry>& beam, const std::vector<uint32_t>& labels) {
    if (!g_debug.hop_file.is_open()) return;

    std::lock_guard<std::mutex> lock(g_debug.log_mutex);

    // Count beam partition distribution
    std::vector<uint32_t> beam_part(g_debug.n_servers, 0);
    for (size_t i = 0; i < beam.size() && i < labels.size(); i++) {
        if (labels[i] < g_debug.n_servers) beam_part[labels[i]]++;
    }

    std::ostringstream oss;
    oss << type << "," << id << "," << hop_num << ","
        << current_partition << "," << next_partition << ","
        << beam.size() << ",";

    for (uint32_t p = 0; p < g_debug.n_servers; p++) {
        oss << beam_part[p] << ",";
    }

    if (!beam.empty()) {
        uint32_t top1_part = labels.empty() ? 0 : labels[0];
        oss << beam[0].node_id << "," << top1_part << ","
            << std::fixed << std::setprecision(4) << beam[0].distance;
    } else {
        oss << "-1,-1,-1";
    }
    oss << "\n";

    g_debug.hop_file << oss.str();

    static uint32_t hop_log_count = 0;
    if (++hop_log_count % 5000 == 0) {
        g_debug.hop_file.flush();
    }
}

// =============================================================================
// log_search_result - one line per search; labels[i] is the owner of neighbors[i]
// =============================================================================
void log_search_result(const char* type, uint32_t id, uint32_t entry_part, uint32_t final_part,
                       uint32_t hops, const std::vector<uint32_t>& neighbors, const std::vector<uint32_t>& labels,
                       const std::vector<float>& distances, uint32_t K,
                       const int32_t* gt_row, uint32_t gt_k) {
    if (!g_debug.log_file.is_open()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_debug.log_mutex);

    std::ostringstream oss;
    oss << type << "," << id << "," << entry_part << "," << final_part << "," << hops << ",";

    // Top-10 neighbors with partition and distance
    for (uint32_t i = 0; i < 10; i++) {
        if (i < neighbors.size()) {
            uint32_t nb_part = i < labels.size() ? labels[i] : 0;
            float dist = (i < distances.size()) ? distances[i] : -1.0f;
            oss << neighbors[i] << "," << nb_part << "," << std::fixed << std::setprecision(4) << dist << ",";
        } else {
            oss << "-1,-1,-1,";
        }
    }

    // Top-K partition distribution
    std::vector<uint32_t> part_count(g_debug.n_servers, 0);
    for (uint32_t i = 0; i < std::min((uint32_t)neighbors.size(), K) && i < labels.size(); i++) {
        if (labels[i] < g_debug.n_servers) {
            part_count[labels[i]]++;
        }
    }
    for (uint32_t p = 0; p < g_debug.n_servers; p++) {
        oss << part_count[p] << ",";
    }

    // GT columns and recall calculation
    uint32_t recall_hits = 0;
    std::unordered_set<int32_t> gt_set;

    if (gt_row && gt_k > 0) {
        uint32_t gt_limit = std::min(gt_k, K);
        for (uint32_t i = 0; i < gt_limit; i++) {
            gt_set.insert(gt_row[i]);
        }
        for (uint32_t i = 0; i < std::min((uint32_t)neighbors.size(), K); i++) {
            if (gt_set.count(static_cast<int32_t>(neighbors[i]))) {
                recall_hits++;
            }
        }
    }

    for (uint32_t i = 0; i < 10; i++) {
        if (gt_row && i < gt_k) {
            oss << gt_row[i] << ",";
        } else {
            oss << "-1,";
        }
    }
    oss << recall_hits << "\n";

    g_debug.log_file << oss.str();

    static uint32_t log_count = 0;
    if (++log_count % 1000 == 0) {
        g_debug.log_file.flush();
    }
}

void enable_partition_debug(uint32_t n_servers, const std::string& log_path,
                            const std::vector<uint32_t>* membership_vec) {
    g_debug.enabled = true;
    g_debug.n_servers = n_servers;
    g_debug.entry_partition_count.assign(n_servers, 0);
    g_debug.final_partition_count.assign(n_servers, 0);
    g_debug.transition_matrix.assign(n_servers, std::vector<uint32_t>(n_servers, 0));
    g_debug.topK_partition_total.assign(n_servers, 0);
    g_debug.insert_hop_distribution.assign(20, 0);
    g_debug.insert_total_hops = 0;
    g_debug.insert_count = 0;
    g_debug.query_result_partition.assign(n_servers, 0);
    g_debug.query_count = 0;
    g_debug.query_total_hops = 0;

    if (!log_path.empty() && membership_vec != nullptr) {
        open_debug_log(log_path, n_servers);
        if (g_debug.log_file.is_open()) std::cout << "Debug log opened: " << log_path << "\n";
    }
}

void disable_partition_debug() {
    close_debug_log();
    g_debug.enabled = false;
    g_debug.entry_partition_count.clear();
    g_debug.final_partition_count.clear();
    g_debug.transition_matrix.clear();
    g_debug.topK_partition_total.clear();
    g_debug.insert_count = 0;
    g_debug.insert_total_hops = 0;
    g_debug.insert_hop_distribution.clear();
    g_debug.insert_L = 0;
    g_debug.insert_K = 0;
    g_debug.insert_W = 0;
    g_debug.query_count = 0;
    g_debug.query_total_hops = 0;
    g_debug.query_result_partition.clear();
    g_debug.query_L = 0;
    g_debug.query_K = 0;
    g_debug.n_servers = 4;
}

void print_partition_debug_stats() {
    if (!g_debug.enabled) return;

    std::cout << "\n=== INSERT PARTITION DEBUG ===\n";

    uint64_t total_entry = 0;
    for (auto c : g_debug.entry_partition_count) total_entry += c;
    std::cout << "Entry partition distribution (from head index):\n";
    for (size_t i = 0; i < g_debug.entry_partition_count.size(); i++) {
        double pct = total_entry > 0 ? 100.0 * g_debug.entry_partition_count[i] / total_entry : 0;
        std::cout << "  P" << i << ": " << g_debug.entry_partition_count[i]
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }

    uint64_t total_final = 0;
    for (auto c : g_debug.final_partition_count) total_final += c;
    std::cout << "Final partition distribution (from search result):\n";
    for (size_t i = 0; i < g_debug.final_partition_count.size(); i++) {
        double pct = total_final > 0 ? 100.0 * g_debug.final_partition_count[i] / total_final : 0;
        std::cout << "  P" << i << ": " << g_debug.final_partition_count[i]
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }

    uint64_t topK_total = 0;
    for (auto c : g_debug.topK_partition_total) topK_total += c;
    std::cout << "\nTop-K neighbors partition distribution (all " << topK_total << " neighbors):\n";
    for (size_t i = 0; i < g_debug.topK_partition_total.size(); i++) {
        double pct = topK_total > 0 ? 100.0 * g_debug.topK_partition_total[i] / topK_total : 0;
        std::cout << "  P" << i << ": " << g_debug.topK_partition_total[i]
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }

    std::cout << "\nTransition matrix (Entry -> Final):\n";
    std::cout << "        ";
    for (size_t j = 0; j < g_debug.transition_matrix.size(); j++) {
        std::cout << "  Final" << j;
    }
    std::cout << "\n";
    for (size_t i = 0; i < g_debug.transition_matrix.size(); i++) {
        uint64_t row_total = 0;
        for (auto c : g_debug.transition_matrix[i]) row_total += c;
        std::cout << "Entry" << i << ":";
        for (size_t j = 0; j < g_debug.transition_matrix[i].size(); j++) {
            double pct = row_total > 0 ? 100.0 * g_debug.transition_matrix[i][j] / row_total : 0;
            std::cout << " " << std::setw(6) << std::fixed << std::setprecision(1) << pct << "%";
        }
        std::cout << "  (n=" << row_total << ")\n";
    }

    if (g_debug.query_count > 0) {
        std::cout << "\nQuery search result[0] partition distribution (" << g_debug.query_count << " queries):\n";
        for (size_t i = 0; i < g_debug.query_result_partition.size(); i++) {
            double pct = 100.0 * g_debug.query_result_partition[i] / g_debug.query_count;
            std::cout << "  P" << i << ": " << g_debug.query_result_partition[i]
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
        }
        double avg_hops = (double)g_debug.query_total_hops / g_debug.query_count;
        std::cout << "  Avg hops: " << std::fixed << std::setprecision(2) << avg_hops << "\n";
    }

    std::cout << "\nSearch parameters:\n";
    std::cout << "  Insert search: L=" << g_debug.insert_L
              << ", K=" << g_debug.insert_K
              << ", W=" << g_debug.insert_W << "\n";
    std::cout << "  Query search:  L=" << g_debug.query_L
              << ", K=" << g_debug.query_K << "\n";

    if (g_debug.insert_count > 0) {
        std::cout << "\nInsert search hop distribution:\n";
        double avg_hops = (double)g_debug.insert_total_hops / g_debug.insert_count;
        std::cout << "  Total inserts: " << g_debug.insert_count << "\n";
        std::cout << "  Total hops: " << g_debug.insert_total_hops << "\n";
        std::cout << "  Avg hops per insert: " << std::fixed << std::setprecision(2) << avg_hops << "\n";
        std::cout << "  Hop distribution:\n";
        for (size_t i = 0; i < g_debug.insert_hop_distribution.size() && g_debug.insert_hop_distribution[i] > 0; i++) {
            double pct = 100.0 * g_debug.insert_hop_distribution[i] / g_debug.insert_count;
            std::cout << "    " << i << " hops: " << g_debug.insert_hop_distribution[i]
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
        }
    }

    std::cout << "=== END PARTITION DEBUG ===\n\n";

    close_debug_log();
}

void populate_debug_metrics(ann::ExperimentResult& result) {
    if (!g_debug.enabled) return;

    for (size_t i = 0; i < g_debug.entry_partition_count.size(); i++) {
        if (g_debug.entry_partition_count[i] > 0) {
            result.partition_analysis.entry_partition[i] = g_debug.entry_partition_count[i];
        }
    }

    for (size_t i = 0; i < g_debug.final_partition_count.size(); i++) {
        if (g_debug.final_partition_count[i] > 0) {
            result.partition_analysis.final_partition[i] = g_debug.final_partition_count[i];
        }
    }

    if (!g_debug.transition_matrix.empty()) {
        result.partition_analysis.transitions = g_debug.transition_matrix;
    }
}

} // namespace tide
