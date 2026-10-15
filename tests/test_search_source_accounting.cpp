// Search sources, owner labels and backlink edge accounting on the real server path.
#include "server_types.h"
#include "stats_window.h"
#include "npy_writer.h"
#include "owner_label.h"
#include "protocol/broadcast_messages.h"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { std::cerr << "CHECK failed: " #expr "\n"; return 1; } } while (0)

namespace tide {
std::shared_mutex g_resize_mutex;
std::mutex g_alloc_mutex;
LockMetrics g_lock_metrics;
GraphMetrics g_graph_metrics;
std::string g_data_dir;
int g_rank = 0;
uint32_t g_start_node = 0;
}

namespace {

struct SearchRun {
    ann::StateStatus status = ann::StateStatus::ERROR;
    std::vector<uint32_t> neighbors;
    tide::ProcessMetrics metrics;
};

SearchRun run_once(tide::ServerData& data) {
    const float query[1] = {1.0f};
    std::vector<ann::BeamEntry> beam{{0, 0.0f, 0, {0, 0, 0}}};
    SearchRun run;
    uint32_t next_partition = 0;
    std::vector<float> distances;
    std::vector<bool> seen(1, false);
    std::vector<uint32_t> seen_nodes;
    tide::process_state(
        data, query, 1, /*L=*/4, /*K=*/2, /*W=*/1, beam, run.status,
        next_partition, run.neighbors, distances, run.metrics, seen, seen_nodes,
        /*query_id=*/0, /*hop_count=*/0);
    return run;
}

// One backlink batch for the new node 3 with one edge of every outcome: to 0
// (local, applied), to 1 (local, already holds 3), to 2 (owned elsewhere: no
// local row) and to 7 (beyond this server's n_base: an id it has not learned).
int backlink_balance() {
    tide::ServerData data;
    data.dim = 1;
    data.my_partition = 0;
    data.n_servers = 4;
    data.base_static_capacity = 8;
    data.full_base.assign(8, 0.0f);
    data.membership = {0, 0, 1, 0, 0, 0, 0, 0};
    tide::n_base_set(data, 3);
    CHECK(data.global_to_local.ensure_capacity_for(7, UINT32_MAX));
    CHECK(data.adj.ensure_capacity_for(1));
    CHECK(data.local_to_global.ensure_capacity_for(1, UINT32_MAX));
    data.global_to_local[0] = 0; data.local_to_global[0] = 0; data.adj[0] = {1};
    data.global_to_local[1] = 1; data.local_to_global[1] = 1; data.adj[1] = {0, 3};
    tide::n_local_set(data, 2);
    tide::ensure_node_locks(data, 2);
    tide::g_graph_metrics.reset();

    const float vector[1] = {3.0f};
    std::vector<tide::WorkItem> batch(1);
    batch[0].request = ann::serialize_batch_backlink_request(
        /*batch_id*/ 9, /*from_id*/ 3, /*from_partition*/ 1, /*dim*/ 1, /*R*/ 0, /*alpha*/ 1.2f,
        {0, 1, 2, 7}, vector);
    batch[0].msg_type = ann::MessageType::BATCH_BACKLINK_REQUEST;
    tide::TimingMetrics metrics;
    const auto responses = tide::batch_process_backlinks(data, batch, metrics);
    CHECK(responses.size() == 1);

    const auto& m = tide::g_graph_metrics;
    CHECK(m.backlink_edges_requested == 4);
    CHECK(m.backlink_edges_applied == 1);
    CHECK(m.backlink_skip_already_exists == 1);
    CHECK(m.backlink_skip_not_local == 1);
    CHECK(m.backlink_skip_out_of_range == 1);
    CHECK(m.backlink_edges_requested == m.backlink_edges_applied + m.backlink_skip_already_exists +
                                           m.backlink_skip_not_local + m.backlink_skip_out_of_range);
    CHECK((data.adj[0] == std::vector<uint32_t>{1, 3}));
    CHECK((data.adj[1] == std::vector<uint32_t>{0, 3}));
    return 0;
}

// Owner labels at load: membership.npy with a label >= K (and a rank >= K) is
// rejected before any graph file is read, and the routing helpers never fold.
int owner_labels_checked() {
    char pattern[] = "/tmp/server-owner-labels-XXXXXX";
    const char* made = mkdtemp(pattern);
    CHECK(made != nullptr);
    const std::string root = made;
    ann::write_npy_float(root + "/base.npy", std::vector<float>(8, 0.0f), 8, 1);
    ann::write_npy_uint32(root + "/membership.npy", {0, 1, 2, 3, 4, 0, 1, 2});
    std::string what;
    {
        tide::ServerData data;
        data.n_servers = 4; data.my_partition = 0;
        try { tide::load_server_data(data, root, 0); } catch (const std::runtime_error& e) { what = e.what(); }
        CHECK(what.find("node 4 has owner label 4, not below the server count 4") != std::string::npos);
    }
    ann::write_npy_uint32(root + "/membership.npy", {0, 1, 2, 3, 0, 0, 1, 2});
    {
        tide::ServerData data;
        data.n_servers = 4; data.my_partition = 4;
        what.clear();
        try { tide::load_server_data(data, root, 4); } catch (const std::runtime_error& e) { what = e.what(); }
        CHECK(what.find("rank 4 is not below the server count 4") != std::string::npos);
    }
    {
        // Eight servers accept labels 4..7 (they would have been folded or rejected at K = 4). The
        // load then stops only at the missing partition directory, after the label check.
        tide::ServerData data;
        data.n_servers = 8; data.my_partition = 0;
        ann::write_npy_uint32(root + "/membership.npy", {0, 1, 2, 3, 4, 5, 6, 7});
        what.clear();
        try { tide::load_server_data(data, root, 0); } catch (const std::exception& e) { what = e.what(); }
        CHECK(what.find("owner label") == std::string::npos && data.membership.size() == 8);
    }
    std::filesystem::remove_all(root);
    std::string error;
    CHECK(tide::owner_labels_valid({0, 1, 7}, 8, &error) && error.empty());
    CHECK(!tide::owner_labels_valid({0, 1, 8}, 8, &error) && error.find("node 2 has owner label 8") != std::string::npos);
    CHECK(!tide::owner_labels_valid({0}, 0, &error));
    CHECK(tide::owner_label_checked(7, 8, "test") == 7);
    // The search path stops on a corrupt label (4 of K = 4) instead of folding it onto rank 0.
    std::cout.flush();
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        tide::ServerData data;
        data.dim = 1; data.my_partition = 0; data.n_servers = 4; data.base_static_capacity = 2;
        data.full_base = {1.0f, 2.0f};
        data.membership = {4, 0};
        tide::n_base_set(data, 2);
        tide::n_local_set(data, 0);
        if (!data.global_to_local.ensure_capacity_for(1, UINT32_MAX)) _exit(3);
        run_once(data);
        _exit(0);  // reached only if the label was folded
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    return 0;
}

// PQ mode: the same two owned nodes plus node 2, owned by
// rank 1 and held here only as a code, and node 3, inserted on rank 1 and not
// yet known here by any code. Expanding node 0 re-ranks it exactly and scores
// its neighbors 1 (held: exact), 2 (code: ADC) and 3 (missing: FLT_MAX). The
// counters split the work by kind. The distance count and the window work W(0)
// count the evaluations: the re-rank, the exact and the ADC distance (3), not
// the missing code. One ADC table was built for the hop.
int pq_accounting() {
    tide::ServerData data;
    data.own_only = true;
    data.dim = 2; data.my_partition = 0; data.n_servers = 4; data.base_static_capacity = 3;
    data.membership = {0, 0, 1};
    tide::n_base_set(data, 3);
    // One chunk of two dimensions (M = 1): centroid k = (k, 0).
    data.pq_table.dim = 2; data.pq_table.M = 1; data.pq_table.sub_dim = 2;
    data.pq_table.centroids.assign(2 * ann::PQTable::K, 0.0f);
    for (uint32_t k = 0; k < ann::PQTable::K; ++k) data.pq_table.centroid_ptr_mut(0, k)[0] = static_cast<float>(k);
    data.pq_table.build_sym_table();
    data.pq_codes_static = {1, 2, 5};
    data.pq_codes_delta.init(1);
    data.full_own.init(2);
    CHECK(data.global_to_local.ensure_capacity_for(2, UINT32_MAX));
    CHECK(data.adj.ensure_capacity_for(1));
    CHECK(data.local_to_global.ensure_capacity_for(1, UINT32_MAX));
    const float v0[2] = {1.0f, 0.0f}, v1[2] = {2.0f, 0.5f};
    tide::store_own_vector(data, 0, v0);
    tide::store_own_vector(data, 1, v1);
    data.global_to_local[0] = 0; data.local_to_global[0] = 0; data.adj[0] = {1, 2};
    data.global_to_local[1] = 1; data.local_to_global[1] = 1; data.adj[1] = {};
    tide::n_local_set(data, 2);
    tide::ensure_node_locks(data, 2);
    // Node 3: inserted on rank 1 (membership known here), no code stored yet.
    tide::server_ensure_mem_capacity(data, 3);
    tide::mem_set(data, 3, 1);
    tide::n_base_bump(data, 4);
    data.adj[0] = {1, 2, 3};
    // Window statistics active over the three load-time nodes (the static
    // table, while inserted ids would count in the delta table a BEGIN grows).
    tide::stats_window_allocate(data, 3);
    data.stats_window_active.store(true);

    const float query[2] = {4.0f, 0.0f};
    std::vector<ann::BeamEntry> beam{{0, 0.0f, 0, {0, 0, 0}}};
    ann::StateStatus status = ann::StateStatus::ERROR;
    uint32_t next = 0;
    std::vector<uint32_t> neighbors;
    std::vector<float> distances;
    tide::ProcessMetrics m;
    std::vector<bool> seen(4, false);
    std::vector<uint32_t> seen_nodes;
    tide::process_state(data, query, 2, /*L=*/4, /*K=*/3, /*W=*/1, beam, status, next, neighbors, distances, m,
                          seen, seen_nodes, 0, 0);
    // Node 2 is rank 1's: the search hops there with the code's ADC estimate.
    CHECK(status == ann::StateStatus::NEED_HOP && next == 1);
    CHECK(m.neighbors_evaluated == 3);
    CHECK(m.exact_distances == 1 && m.adc_distances == 1 && m.missing_codes == 1 && m.rerank_distances == 1);
    // Node 0 (the coordinator's entry, no exact flag) was re-ranked. Node 1 was
    // scored exactly and is not. Node 2 keeps its estimate (W = 1: never expanded).
    CHECK(m.distance_computations == 3);
    CHECK(data.stats_window_w[0].load() == 3 && data.stats_window_w[1].load() == 0 &&
          data.stats_window_w[2].load() == 0);
    CHECK(m.adc_tables == 1);
    CHECK(beam.size() == 4 && beam[0].node_id == 2 && !ann::beam_entry_exact(beam[0]) && beam[0].distance == 1.0f);
    CHECK(beam[1].node_id == 1 && ann::beam_entry_exact(beam[1]) && beam[1].distance == 4.25f);
    CHECK(beam[2].node_id == 0 && ann::beam_entry_exact(beam[2]) && beam[2].distance == 9.0f);
    CHECK(beam[3].node_id == 3 && !ann::beam_entry_exact(beam[3]));
    // A hop that only re-ranks (the entry's neighbors were all seen) still
    // counts the re-rank as distance work, timed, and builds no ADC table.
    std::vector<ann::BeamEntry> again{{1, 7.0f, 0, {0, 0, 0}}};
    std::vector<bool> all_seen(4, true);
    std::vector<uint32_t> seen_again;
    tide::ProcessMetrics r;
    tide::process_state(data, query, 2, 4, 3, 1, again, status, next, neighbors, distances, r, all_seen,
                          seen_again, 0, 0);
    CHECK(status == ann::StateStatus::DONE && r.rerank_distances == 1 && r.distance_computations == 1 &&
          r.neighbors_evaluated == 0 && r.adc_tables == 0 && neighbors == std::vector<uint32_t>{1} &&
          distances.size() == 1 && distances[0] == std::sqrt(4.25f));
    CHECK(data.stats_window_w[1].load() == 1);
    return 0;
}

}  // namespace

int main() {
    if (backlink_balance() != 0) return 1;
    if (owner_labels_checked() != 0) return 1;
    if (pq_accounting() != 0) return 1;
    // Two global nodes, both owned by partition 0 of four.
    tide::ServerData data;
    data.dim = 1;
    data.my_partition = 0;
    data.n_servers = 4;
    data.base_static_capacity = 2;
    data.full_base = {1.0f, 2.0f};
    data.membership = {0, 0};
    tide::n_base_set(data, 2);
    tide::n_local_set(data, 0);
    CHECK(data.global_to_local.ensure_capacity_for(1, UINT32_MAX));

    // Node 0 is owned but has no global-to-local row: the search marks it
    // expanded without reading adjacency or computing any distance.
    auto invalid = run_once(data);
    CHECK(invalid.status == ann::StateStatus::DONE);
    CHECK(invalid.metrics.local_nodes_expanded == 1);
    CHECK(invalid.metrics.neighbors_evaluated == 0);
    CHECK(invalid.metrics.distance_computations == 0);
    CHECK(invalid.neighbors == std::vector<uint32_t>{0});

    // Give node 0 a valid local row whose adjacency holds itself (already seen)
    // and node 1 (unseen): exactly one distance is computed, and node 1 enters
    // the result. Node 1 has no local row, so it is not expanded in turn.
    data.global_to_local[0] = 0;
    CHECK(data.adj.ensure_capacity_for(0));
    data.adj[0] = {0, 1};
    tide::n_local_set(data, 1);
    tide::ensure_node_locks(data, 1);
    auto valid = run_once(data);
    CHECK(valid.status == ann::StateStatus::DONE);
    CHECK(valid.metrics.neighbors_evaluated == 1);
    CHECK(valid.metrics.distance_computations == 1);
    CHECK(valid.metrics.local_nodes_expanded == 2);
    CHECK(valid.neighbors.size() == 2);
    CHECK(std::find(valid.neighbors.begin(), valid.neighbors.end(), 1u) != valid.neighbors.end());

    std::cout << "search_source_accounting tests passed\n";
    return 0;
}
