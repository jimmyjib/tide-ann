/**
 * Admin commands of the coordinator.
 *
 * Contains:
 * - send_admin_command: Send reset/drain/shutdown to all servers
 * - process_daemon_command: Process command from client
 * - run_ping_baseline: Network baseline RTT measurement
 */

#include "coordinator_types.h"
#include "state_protocol.h"
#include "protocol/broadcast_messages.h"
#include "protocol/insert_protocol.h"
#include "metrics/statistics.h"
#include "pipeline_types.h"
#include "membership_migration_plan.h"
#include "membership_migration_coordinator.h"
#include "coordinator_stats_window.h"
#include "window_controller_runtime.h"
#include "window_snapshot_transport.h"
#include "window_native_control.h"
#include "window_controller_transitions.h"
#include "npy_writer.h"
#include "sha256.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

namespace tide {

using namespace std::chrono;
using ann::MessageType;
using ann::PingRequest;
using ann::PingResponse;

struct AdminDrainStats {
    uint64_t responses_seen = 0;
    uint64_t insert_responses_seen = 0;
    uint64_t exact_backlink_batches_queued = 0;
    uint64_t exact_backlink_edges_queued = 0;
    uint64_t backlink_acks_seen = 0;
    uint64_t broadcast_acks_seen = 0;
    uint64_t broadcast_acks_accounted = 0;
    // What the final barrier had to mark committed without its reply: every
    // server ACKed the barrier, yet these were still pending. A clean drain
    // settles nothing; a nonzero count is a lost insert response, backlink or
    // broadcast ACK (e.g. a reply another consumer dropped).
    uint64_t settled_inserts = 0;
    uint64_t settled_insert_responses = 0;
    uint64_t settled_backlink_batches = 0;
    uint64_t settled_broadcast_acks = 0;
    // What was pending when the drain began (the coordinator's pending maps):
    // insert responses, backlink batches and broadcasts awaiting their ACKs.
    // A drain at a phase boundary that had work to do shows it here (the drain
    // that controller-start runs first reports the work it settled).
    uint64_t pending_insert_responses_at_start = 0;
    uint64_t pending_backlink_batches_at_start = 0;
    uint64_t pending_broadcasts_at_start = 0;
};

static AdminDrainStats g_last_drain_stats;
static MembershipMigrationPlan g_coordinator_migration_plan;
// Ownership epochs of the accepted migrations, oldest first. Only the epochs
// are kept: an accepted plan (with its rows) is never needed again.
static std::vector<uint64_t> g_coordinator_accepted_epochs;
static uint64_t g_coordinator_ownership_epoch = 0;
// Set once an ACCEPT fanout of the active plan has gone out. A server that
// accepted dropped its journal and can never roll back, so from then on the
// plan resolves only by a (retried, idempotent) ACCEPT; rollback is refused.
static bool g_coordinator_accept_sent = false;
// ACCEPT attempts per call: a server that accepted answers a retry with an
// idempotent success, so a retry completes a partial ACCEPT on the rest.
static constexpr int kMigrationAcceptAttempts = 3;
// Largest search list a recall command may ask for (--search-l).
static constexpr uint32_t kMaxCommandSearchL = 4096;
// Every ACCEPT fanout carries a fresh attempt id that the servers echo in their
// ACK (MembershipMigrationAck::request_id): an ACK that arrives after its
// attempt timed out is told apart from the current attempt's and discarded,
// never counted for a later attempt and never left for another exchange.
static uint16_t g_migration_accept_request_id = 0;
static bool drain_pending_ops_for_admin();

static std::string stats_snapshots_json(const std::vector<ann::StatsResponseHeader>& snapshots,
                                        bool complete) {
    std::ostringstream json;
    json << "{\"status\": \"" << (complete ? "ok" : "partial")
         << "\", \"action\": \"stats\", \"servers\": [";
    for (size_t i = 0; i < snapshots.size(); ++i) {
        if (i != 0) json << ", ";
        const auto& s = snapshots[i];
        json << "{\"server\": " << i
             << ", \"request_id\": " << s.request_id
             << ", \"partition_id\": " << s.partition_id
             << ", \"graph\": {\"total_nodes\": " << s.total_nodes
             << ", \"local_nodes\": " << s.local_nodes
             << ", \"directed_edges\": " << s.directed_edges
             << ", \"adjacency_serialized_bytes\": " << s.adjacency_serialized_bytes
             << ", \"adjacency_allocated_neighbor_bytes\": "
             << s.adjacency_allocated_neighbor_bytes
             << ", \"backlink_edges_requested\": " << s.backlink_edges_requested
             << ", \"backlink_edges_applied\": " << s.backlink_edges_applied
             << ", \"backlink_skip_not_local\": " << s.backlink_skip_not_local
             << ", \"backlink_skip_already_exists\": " << s.backlink_skip_already_exists
             << ", \"backlink_skip_out_of_range\": " << s.backlink_skip_out_of_range
             << ", \"insert_neighbors_undelivered\": " << s.insert_neighbors_undelivered
             << "}"
             << ", \"search\": {\"request_count\": " << s.search_request_count
             << ", \"distance_count\": " << s.search_distance_count
             << ", \"exact_count\": " << s.search_exact_count
             << ", \"adc_count\": " << s.search_adc_count
             << ", \"missing_code_count\": " << s.search_missing_code_count
             << ", \"rerank_count\": " << s.search_rerank_count
             << ", \"adc_table_count\": " << s.search_adc_table_count
             << ", \"adc_table_us\": " << s.search_adc_table_us
             << ", \"neighbors_evaluated\": " << s.search_neighbors_evaluated
             << ", \"total_us\": " << s.search_total_us
             << ", \"distance_us\": " << s.search_distance_us
             << ", \"beam_sort_us\": " << s.search_beam_sort_us << "}"
             << ", \"storage\": {\"mode\": \"" << (s.storage_mode == 1 ? "pq" : "full") << "\""
             << ", \"dim\": " << s.storage_dim
             << ", \"pq_m\": " << s.storage_pq_m
             << ", \"vector_bytes\": " << s.storage_vector_bytes
             << ", \"code_bytes\": " << s.storage_code_bytes
             << ", \"pq_table_bytes\": " << s.storage_pq_table_bytes
             << ", \"process_rss_bytes\": " << s.process_rss_bytes << "}"
             << ", \"queue\": {\"wait_us\": " << s.queue_wait_us
             << ", \"count\": " << s.queue_wait_count
             << ", \"interval_max_wait_us\": " << s.queue_wait_interval_max_us
             << ", \"response_wait_us\": " << s.response_queue_wait_us
             << ", \"response_count\": " << s.response_queue_wait_count
             << ", \"response_interval_max_wait_us\": "
             << s.response_queue_wait_interval_max_us
             << "}}";
    }
    json << "]}";
    return json.str();
}

// Helper: run ping measurement (sequential)
static void run_ping_mode(uint32_t count,
                          std::vector<double>& all_rtt,
                          std::vector<std::vector<double>>& rtt_per_server) {
    rtt_per_server.resize(g_daemon.n_servers);

    for (uint32_t seq = 0; seq < count; seq++) {
        // Send PING to all servers
        for (size_t s = 0; s < g_daemon.sockets.size(); s++) {
            PingRequest ping;
            ping.msg_type = MessageType::PING_REQUEST;
            ping.padding[0] = ping.padding[1] = ping.padding[2] = 0;
            ping.sequence = seq;
            ping.send_time_ns = static_cast<uint64_t>(
                steady_clock::now().time_since_epoch().count());

            zmq::message_t empty_frame(0);
            zmq::message_t ping_frame(&ping, sizeof(ping));
            g_daemon.sockets[s].send(empty_frame, zmq::send_flags::sndmore);
            g_daemon.sockets[s].send(ping_frame, zmq::send_flags::none);
        }

        // Receive PONG from all servers
        std::vector<bool> received(g_daemon.n_servers, false);
        uint32_t recv_count = 0;
        auto timeout = steady_clock::now() + std::chrono::milliseconds(1000);

        while (recv_count < g_daemon.n_servers && steady_clock::now() < timeout) {
            for (size_t s = 0; s < g_daemon.sockets.size(); s++) {
                if (received[s]) continue;

                zmq::message_t empty_frame, response;
                auto r1 = g_daemon.sockets[s].recv(empty_frame, zmq::recv_flags::dontwait);
                if (!r1) continue;
                auto r2 = g_daemon.sockets[s].recv(response, zmq::recv_flags::none);

                if (r2 && response.size() >= sizeof(PingResponse)) {
                    auto recv_time = steady_clock::now();
                    const PingResponse* pong = reinterpret_cast<const PingResponse*>(response.data());

                    if (pong->msg_type == MessageType::PING_RESPONSE && pong->sequence == seq) {
                        uint64_t send_ns = pong->send_time_ns;
                        uint64_t recv_ns = static_cast<uint64_t>(recv_time.time_since_epoch().count());
                        double rtt_us = static_cast<double>(recv_ns - send_ns) / 1000.0;

                        rtt_per_server[s].push_back(rtt_us);
                        received[s] = true;
                        recv_count++;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Collect all RTTs
    for (size_t s = 0; s < g_daemon.n_servers; s++) {
        all_rtt.insert(all_rtt.end(), rtt_per_server[s].begin(), rtt_per_server[s].end());
    }
}

// Pipelined ping measurement
// Each server maintains batch_size in-flight requests independently
static void run_ping_mode_pipelined(uint32_t count, uint32_t batch_size,
                                     std::vector<double>& all_rtt,
                                     std::vector<std::vector<double>>& rtt_per_server) {
    rtt_per_server.resize(g_daemon.n_servers);

    // Per-server tracking
    struct ServerState {
        std::map<uint32_t, uint64_t> in_flight;  // seq -> send_time_ns
        uint32_t next_seq = 0;
        uint32_t completed = 0;
    };
    std::vector<ServerState> servers(g_daemon.n_servers);

    // Helper to send ping to one server
    auto send_ping_to_server = [&](size_t s) {
        uint32_t seq = servers[s].next_seq++;
        uint64_t send_time_ns = static_cast<uint64_t>(
            steady_clock::now().time_since_epoch().count());

        PingRequest ping;
        ping.msg_type = MessageType::PING_REQUEST;
        ping.padding[0] = ping.padding[1] = ping.padding[2] = 0;
        ping.sequence = seq;
        ping.send_time_ns = send_time_ns;

        zmq::message_t empty_frame(0);
        zmq::message_t ping_frame(&ping, sizeof(ping));
        g_daemon.sockets[s].send(empty_frame, zmq::send_flags::sndmore);
        g_daemon.sockets[s].send(ping_frame, zmq::send_flags::none);
        servers[s].in_flight[seq] = send_time_ns;
    };

    // Initial batch: send batch_size pings to each server
    for (size_t s = 0; s < g_daemon.n_servers; s++) {
        for (uint32_t i = 0; i < batch_size && servers[s].next_seq < count; i++) {
            send_ping_to_server(s);
        }
    }

    // Main loop: receive responses and maintain batch_size in-flight
    uint32_t total_completed = 0;
    uint32_t total_target = count * g_daemon.n_servers;
    auto timeout = steady_clock::now() + std::chrono::seconds(30);

    while (total_completed < total_target && steady_clock::now() < timeout) {
        for (size_t s = 0; s < g_daemon.sockets.size(); s++) {
            zmq::message_t empty_frame, response;
            auto r1 = g_daemon.sockets[s].recv(empty_frame, zmq::recv_flags::dontwait);
            if (!r1) continue;
            auto r2 = g_daemon.sockets[s].recv(response, zmq::recv_flags::none);

            if (r2 && response.size() >= sizeof(PingResponse)) {
                auto recv_time = steady_clock::now();
                const PingResponse* pong = reinterpret_cast<const PingResponse*>(response.data());

                if (pong->msg_type == MessageType::PING_RESPONSE) {
                    uint32_t seq = pong->sequence;
                    auto it = servers[s].in_flight.find(seq);
                    if (it != servers[s].in_flight.end()) {
                        uint64_t send_ns = it->second;
                        uint64_t recv_ns = static_cast<uint64_t>(recv_time.time_since_epoch().count());
                        double rtt_us = static_cast<double>(recv_ns - send_ns) / 1000.0;
                        rtt_per_server[s].push_back(rtt_us);
                        servers[s].in_flight.erase(it);
                        servers[s].completed++;
                        total_completed++;

                        // Send next ping to maintain batch_size in-flight
                        if (servers[s].next_seq < count) {
                            send_ping_to_server(s);
                        }
                    }
                }
            }
        }
    }

    // Collect all RTTs
    for (size_t s = 0; s < g_daemon.n_servers; s++) {
        all_rtt.insert(all_rtt.end(), rtt_per_server[s].begin(), rtt_per_server[s].end());
    }
}

// Helper: compute (avg, min, p50, p95, p99, max) from RTT vector
static std::tuple<double, double, double, double, double, double>
compute_rtt_stats(std::vector<double>& rtts) {
    if (rtts.empty()) return {0, 0, 0, 0, 0, 0};
    std::sort(rtts.begin(), rtts.end());
    double sum = 0;
    for (double v : rtts) sum += v;
    double avg = sum / rtts.size();
    double p50 = rtts[rtts.size() * 50 / 100];
    double p95 = rtts[ann::nearest_rank_percentile_index(rtts.size(), 95)];
    double p99 = rtts[rtts.size() * 99 / 100];
    return {avg, rtts.front(), p50, p95, p99, rtts.back()};
}

// Helper: format RTT stats as JSON object (no outer braces)
static std::string rtt_stats_json(double avg, double min_v, double p50, double p95,
                                  double p99, double max_v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(1);
    os << "\"avg_us\": " << avg << ", \"p50_us\": " << p50
       << ", \"p95_us\": " << p95 << ", \"p99_us\": " << p99
       << ", \"min_us\": " << min_v
       << ", \"max_us\": " << max_v;
    return os.str();
}

// Network baseline measurement via PING/PONG
// Runs sequential ping, optionally pipelined ping, stores in g_daemon.baseline, returns JSON
std::string run_ping_baseline(uint32_t count, uint32_t pipelined_batch) {
    daemon_output_line("Running network baseline: " + std::to_string(count) + " pings per server");

    // Sequential ping
    std::vector<double> seq_rtt;
    std::vector<std::vector<double>> seq_per_server;
    run_ping_mode(count, seq_rtt, seq_per_server);

    // Store in g_daemon.baseline (for embedding in next recall/insert result)
    g_daemon.baseline.ping_count = count;
    g_daemon.baseline.servers.resize(g_daemon.n_servers);
    for (size_t s = 0; s < g_daemon.n_servers; s++) {
        auto [avg, min_v, p50, p95, p99, max_v] = compute_rtt_stats(seq_per_server[s]);
        g_daemon.baseline.servers[s].id = static_cast<uint32_t>(s);
        g_daemon.baseline.servers[s].count = static_cast<uint32_t>(seq_per_server[s].size());
        g_daemon.baseline.servers[s].avg_us = avg;
        g_daemon.baseline.servers[s].min_us = min_v;
        g_daemon.baseline.servers[s].p50_us = p50;
        g_daemon.baseline.servers[s].p95_us = p95;
        g_daemon.baseline.servers[s].p99_us = p99;
        g_daemon.baseline.servers[s].max_us = max_v;
    }
    auto [d_avg, d_min, d_p50, d_p95, d_p99, d_max] = compute_rtt_stats(seq_rtt);
    g_daemon.baseline.total_pings = static_cast<uint32_t>(seq_rtt.size());
    g_daemon.baseline.overall_avg_us = d_avg;
    g_daemon.baseline.overall_min_us = d_min;
    g_daemon.baseline.overall_p50_us = d_p50;
    g_daemon.baseline.overall_p95_us = d_p95;
    g_daemon.baseline.overall_p99_us = d_p99;
    g_daemon.baseline.overall_max_us = d_max;
    g_daemon.baseline.direct.avg_us = d_avg;
    g_daemon.baseline.direct.min_us = d_min;
    g_daemon.baseline.direct.p50_us = d_p50;
    g_daemon.baseline.direct.p95_us = d_p95;
    g_daemon.baseline.direct.p99_us = d_p99;
    g_daemon.baseline.direct.max_us = d_max;
    g_daemon.baseline.measured = true;

    // Build JSON
    std::ostringstream json;
    json << "{\n  \"sequential\": {\"count\": " << count << ", "
         << rtt_stats_json(d_avg, d_min, d_p50, d_p95, d_p99, d_max) << "}";

    // Pipelined ping
    if (pipelined_batch > 0) {
        daemon_output_line("Running pipelined ping: batch=" + std::to_string(pipelined_batch));
        std::vector<double> pipe_rtt;
        std::vector<std::vector<double>> pipe_per_server;
        run_ping_mode_pipelined(count, pipelined_batch, pipe_rtt, pipe_per_server);
        auto [p_avg, p_min, p_p50, p_p95, p_p99, p_max] = compute_rtt_stats(pipe_rtt);

        json << ",\n  \"pipelined\": {\"count\": " << count
             << ", \"batch_size\": " << pipelined_batch << ", "
             << rtt_stats_json(p_avg, p_min, p_p50, p_p95, p_p99, p_max);
        json << ", \"per_server\": [";
        for (size_t s = 0; s < pipe_per_server.size(); s++) {
            auto [ps_avg, ps_min, ps_p50, ps_p95, ps_p99, ps_max] =
                compute_rtt_stats(pipe_per_server[s]);
            if (s > 0) json << ", ";
            json << "{\"id\": " << s << ", " << std::fixed << std::setprecision(1)
                 << "\"avg_us\": " << ps_avg << ", \"min_us\": " << ps_min
                 << ", \"p50_us\": " << ps_p50 << ", \"p95_us\": " << ps_p95
                 << ", \"p99_us\": " << ps_p99 << ", \"max_us\": " << ps_max << "}";
        }
        json << "]}";
    }

    json << "\n}";
    return json.str();
}

// Send admin command to all servers via daemon sockets.
bool send_admin_command(ann::MessageType cmd_type) {
    std::string cmd_name;
    const ann::MessageType ack_type = ann::admin_ack_type(cmd_type);
    const bool strict_ack_validation =
        cmd_type == ann::MessageType::SHUTDOWN_REQUEST;
    if (cmd_type == ann::MessageType::RESET_REQUEST) {
        cmd_name = "RESET";
    } else if (cmd_type == ann::MessageType::DRAIN_REQUEST) {
        cmd_name = "DRAIN";
    } else if (cmd_type == ann::MessageType::SHUTDOWN_REQUEST) {
        cmd_name = "SHUTDOWN";
    } else {
        daemon_output_line("[ERROR] Unsupported admin request");
        return false;
    }

    daemon_output_line("Sending " + cmd_name + " to " + std::to_string(g_daemon.n_servers) + " servers...");

    // The message is the type byte alone.
    const std::vector<uint8_t> msg{static_cast<uint8_t>(cmd_type)};

    // Send via server_queues if available (thread-safe), else direct socket
    if (g_daemon.main_response_queue && !g_daemon.server_queues.empty()) {
        for (size_t i = 0; i < g_daemon.server_queues.size(); i++) {
            tide::ServerWorkItem swi;
            swi.request_data.assign(msg.begin(), msg.end());
            swi.target_server = static_cast<uint32_t>(i);
            g_daemon.server_queues[i]->push(std::move(swi));
        }
    } else {
        for (size_t i = 0; i < g_daemon.sockets.size(); i++) {
            zmq::message_t empty_frame(0);
            zmq::message_t msg_frame(msg.data(), msg.size());
            g_daemon.sockets[i].send(empty_frame, zmq::send_flags::sndmore);
            g_daemon.sockets[i].send(msg_frame, zmq::send_flags::none);
        }
    }

    // Wait for ACKs with timeout
    uint32_t acks_received = 0;
    std::vector<bool> server_acked(g_daemon.n_servers, false);
    daemon_output_line("Waiting for ACKs...");

    // The daemon response queue is shared. Replies that are not this
    // command's ACK (late insert responses, backlink/broadcast ACKs, stray
    // stats replies) are set aside and put back on every exit, never dropped,
    // so their owner (drain, the pipeline) still accounts for them. RESET is
    // the exception: it drops them, because it clears every pending map and
    // the servers reload (ann::admin_command_requeues_foreign_replies).
    struct ForeignResponses {
        std::vector<ServerResponse> held;
        bool requeue = true;
        ~ForeignResponses() {
            if (!requeue) {
                if (!held.empty())
                    daemon_output_line("Dropped " + std::to_string(held.size()) +
                                       " pre-reset server replies");
                return;
            }
            if (!g_daemon.main_response_queue) return;
            for (auto& response : held)
                g_daemon.main_response_queue->push(std::move(response));
        }
    } foreign;
    foreign.requeue = ann::admin_command_requeues_foreign_replies(cmd_type);

    const int TIMEOUT_SECONDS = 30;
    auto start_time = std::chrono::steady_clock::now();

    while (acks_received < g_daemon.n_servers) {
        // Check timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed >= TIMEOUT_SECONDS) {
            daemon_output_line("[ERROR] Timeout waiting for ACKs after " + std::to_string(TIMEOUT_SECONDS) + "s");
            for (size_t i = 0; i < g_daemon.n_servers; i++) {
                if (!server_acked[i]) {
                    daemon_output_line("  - Server " + std::to_string(i) + " unresponsive");
                }
            }
            return false;
        }

        // Poll responses from server threads (or direct sockets if no threads)
        if (g_daemon.main_response_queue) {
            tide::ServerResponse srv_resp;
            while (g_daemon.main_response_queue->try_pop(srv_resp)) {
                if (srv_resp.response_data.empty()) {
                    foreign.held.push_back(std::move(srv_resp));
                    continue;
                }
                size_t i = srv_resp.server_idx;
                auto mt = ann::get_message_type(
                    srv_resp.response_data.data(), srv_resp.response_data.size());
                const auto disposition = ann::classify_admin_ack(
                    ack_type, mt, i, g_daemon.n_servers,
                    i < server_acked.size() && server_acked[i]);
                if (disposition == ann::AdminAckDisposition::ACCEPT) {
                    daemon_output_line(
                        "  ACK from server " + std::to_string(i));
                    server_acked[i] = true;
                    ++acks_received;
                } else if (
                    strict_ack_validation &&
                    disposition != ann::AdminAckDisposition::UNMATCHED) {
                    daemon_output_line(
                        "[ERROR] Invalid or duplicate shutdown ACK");
                    return false;
                } else if (disposition == ann::AdminAckDisposition::UNMATCHED) {
                    foreign.held.push_back(std::move(srv_resp));
                }
            }
        } else {
            // Fallback: direct socket recv (no server threads)
            for (size_t i = 0; i < g_daemon.sockets.size(); i++) {
                if (server_acked[i]) continue;
                zmq::message_t empty_frame, response;
                auto r1 = g_daemon.sockets[i].recv(empty_frame, zmq::recv_flags::dontwait);
                if (!r1) continue;
                g_daemon.sockets[i].recv(response, zmq::recv_flags::none);
                if (response.size() == 0) continue;
                auto mt = ann::get_message_type(
                    static_cast<const char*>(response.data()), response.size());
                const auto disposition = ann::classify_admin_ack(
                    ack_type, mt, i, g_daemon.n_servers,
                    server_acked[i]);
                if (disposition == ann::AdminAckDisposition::ACCEPT) {
                    daemon_output_line("  ACK from server " + std::to_string(i));
                    server_acked[i] = true;
                    acks_received++;
                } else if (
                    strict_ack_validation &&
                    disposition != ann::AdminAckDisposition::UNMATCHED) {
                    daemon_output_line(
                        "[ERROR] Invalid or duplicate shutdown ACK");
                    return false;
                }
            }
        }  // if/else main_response_queue
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    daemon_output_line(cmd_name + " completed successfully.");
    return true;
}

static void queue_or_send_server_message(size_t server_idx, std::vector<char>&& data) {
    if (server_idx < g_daemon.server_queues.size() && g_daemon.server_queues[server_idx]) {
        ServerWorkItem work;
        work.request_data = std::move(data);
        work.target_server = static_cast<uint32_t>(server_idx);
        g_daemon.server_queues[server_idx]->push(std::move(work));
        return;
    }

    if (server_idx < g_daemon.sockets.size()) {
        zmq::message_t empty_frame(0);
        zmq::message_t msg_frame(data.data(), data.size());
        g_daemon.sockets[server_idx].send(empty_frame, zmq::send_flags::sndmore);
        g_daemon.sockets[server_idx].send(msg_frame, zmq::send_flags::none);
    }
}

// Blocking all-server exchange for coordinator control requests (the stats
// STOP). Only call it while no workload runs: it reads the shared response queue.
template <typename Handler>
static bool control_exchange(const std::vector<char>& request,
                              ann::MessageType expected_type,
                              Handler&& handler, bool stopping_window = false) {
    std::vector<bool> received(g_daemon.n_servers, false);
    uint32_t received_count = 0;
    for (size_t server = 0; server < g_daemon.n_servers; ++server) {
        queue_or_send_server_message(server, std::vector<char>(request));
    }
    // controller-stop can follow a search command whose late insert-maintenance
    // replies (insert responses, backlink and broadcast ACKs) still sit in the
    // shared queue; they belong to the drain and the pipeline, so they are set
    // aside and put back on every exit, never dropped.
    struct MaintenanceReplies {
        std::vector<ServerResponse> held;
        ~MaintenanceReplies() {
            if (!g_daemon.main_response_queue) return;
            for (auto& response : held)
                g_daemon.main_response_queue->push(std::move(response));
        }
    } maintenance;

    const auto deadline = steady_clock::now() + seconds(30);
    while (received_count < g_daemon.n_servers && steady_clock::now() < deadline) {
        auto accept = [&](size_t server, const char* data, size_t size) -> bool {
            const auto actual_type = ann::get_message_type(data, size);
            // A workload's result collection may leave a late observational
            // GET_STATS response in the daemon response queue.
            // It carries no workload state and must not be mistaken for this
            // command's control response.  Drain only this known type; every
            // other unexpected message remains a hard failure.
            if (actual_type == ann::MessageType::GET_STATS_RESPONSE) {
                daemon_output_line(
                    "[CONTROL] Drained late observational GET_STATS response "
                    "from server " + std::to_string(server));
                return true;
            }
            if (stopping_window && actual_type != expected_type &&
                controller_offer(static_cast<uint32_t>(server), data, size)) return true;
            if (server >= g_daemon.n_servers || (!stopping_window && received[server]) || size == 0 ||
                actual_type != expected_type) {
                daemon_output_line(
                    "[ERROR] Control response envelope invalid: server=" +
                    std::to_string(server) + " size=" + std::to_string(size) +
                    " actual_type=" +
                    std::to_string(static_cast<unsigned>(actual_type)) +
                    " expected_type=" +
                    std::to_string(static_cast<unsigned>(expected_type)));
                return false;
            }
            if (!handler(server, data, size)) {
                daemon_output_line(
                    "[ERROR] Control response payload invalid: server=" +
                    std::to_string(server) + " size=" + std::to_string(size));
                return false;
            }
            if (!received[server]) { received[server] = true; ++received_count; }
            return true;
        };

        if (g_daemon.main_response_queue) {
            ServerResponse response;
            while (g_daemon.main_response_queue->try_pop(response)) {
                if (!response.response_data.empty() &&
                    ann::is_maintenance_reply(ann::get_message_type(
                        response.response_data.data(), response.response_data.size()))) {
                    maintenance.held.push_back(std::move(response));
                    continue;
                }
                if (!accept(response.server_idx, response.response_data.data(),
                            response.response_data.size())) {
                    daemon_output_line(
                        "[ERROR] Unexpected or invalid control response");
                    return false;
                }
            }
        } else {
            for (size_t server = 0; server < g_daemon.sockets.size(); ++server) {
                if (received[server]) continue;
                zmq::message_t empty, body;
                auto first = g_daemon.sockets[server].recv(
                    empty, zmq::recv_flags::dontwait);
                if (!first) continue;
                auto second = g_daemon.sockets[server].recv(
                    body, zmq::recv_flags::none);
                if (!second || !accept(server,
                                       static_cast<const char*>(body.data()),
                                       body.size())) {
                    daemon_output_line(
                        "[ERROR] Unexpected or invalid control response");
                    return false;
                }
            }
        }
        std::this_thread::sleep_for(milliseconds(1));
    }
    if (received_count != g_daemon.n_servers) {
        daemon_output_line("[ERROR] Timeout waiting for control responses");
        return false;
    }
    return true;
}

static std::string g_last_window_stop_receipt;
static uint64_t g_last_window_stop_epoch = 0;

double g_migration_exchange_deadline_sec = 30.0;
double g_migration_refresh_deadline_sec = 60.0;
double g_migration_payload_bytes_per_sec = 8.0 * 1024.0 * 1024.0;
// The refresh deadline allows this many bytes per move for a row it cannot know
// before the sources answer (a row is 4 bytes per neighbour).
static constexpr uint64_t kMigrationRowAllowanceBytes = 1024;

double migration_deadline_sec(double base_sec, uint64_t payload_bytes) {
    const double rate = g_migration_payload_bytes_per_sec;
    return base_sec + (rate > 0.0 ? static_cast<double>(payload_bytes) / rate : 0.0);
}

// The fence payload of the active plan, or of the last one once it resolved.
// Its PREPARE bytes scale the deadline of every exchange of the plan: COMMIT,
// ROLLBACK and ACCEPT carry little, but their server work grows with the moves.
static MigrationPayloadStats g_migration_payload;

const MigrationPayloadStats& coordinator_membership_migration_payload() { return g_migration_payload; }

static double plan_exchange_deadline_sec() {
    return migration_deadline_sec(g_migration_exchange_deadline_sec, g_migration_payload.prepare_request_bytes);
}

// An ACCEPT ACK that does not answer the current ACCEPT attempt (`current_id`;
// 0 outside an ACCEPT exchange): the late reply of an earlier attempt. Servers
// answer migration requests in order on one socket, so it always arrives
// before that server's reply to a later attempt.
static bool stale_accept_ack(size_t server, const char* data, size_t size, uint16_t current_id) {
    if (size != sizeof(ann::MembershipMigrationAck) ||
        ann::get_message_type(data, size) != ann::MessageType::MIGRATION_ACCEPT_ACK) return false;
    ann::MembershipMigrationAck ack{};
    std::memcpy(&ack, data, sizeof(ack));
    if (current_id != 0 && ack.request_id == current_id) return false;
    daemon_output_line("[MIGRATION] Discarded a late ACCEPT ACK of attempt " + std::to_string(ack.request_id) +
                       " from server " + std::to_string(server) + " (status " + std::to_string(ack.status) +
                       ", epoch " + std::to_string(ack.epoch_from) + " -> " + std::to_string(ack.epoch_to) + ")");
    return true;
}

// Closes the stats window on every server (all-rank STOP with receipts). The
// controller-stop path calls it while the controller is still active, and so
// does controller-abort for a terminal controller whose ownership is known.
static bool stop_stats_windows(std::string& receipt, bool aborting = false) {
    auto* runtime = window_controller_active();
    if (runtime && (aborting ? !window_controller_abort_allowed(*runtime, nullptr)
                             : window_controller_has_unresolved_action(*runtime))) return false;
    if (!window_owners_valid(static_cast<uint32_t>(g_daemon.n_servers))) return false;
    if (runtime) {
        window_native_cancel(*runtime);
        runtime->stats_stop_pending = true;
    }
    WindowStopRequest stop;
    stop.ranks = static_cast<uint32_t>(g_daemon.n_servers);
    stop.epoch = runtime ? runtime->window_epoch : g_last_window_stop_epoch;
    WindowStopCollector collector(stop.epoch, stop.ranks);
    std::vector<char> request(sizeof(stop)); std::memcpy(request.data(), &stop, sizeof(stop));
    const bool ok = control_exchange(request, ann::MessageType::STATS_WINDOW_STOP_ACK,
        [&](size_t rank, const char* data, size_t size) { return collector.offer(rank, data, size); }, true) && collector.complete();
    receipt = collector.json();
    if (runtime) {
        window_control_detail::journal_event(*runtime, "stats_off",
            "\"pending_exchange_at_stop\":" + std::to_string(static_cast<unsigned>(runtime->exchange)) + receipt);
        runtime->stats_off_confirmed = ok;
        runtime->stats_stop_pending = !ok;
        // No tick/harvest is synthesized. All-rank STOP follows any queued
        // BEGIN/HARVEST and closes even a partial control exchange explicitly.
        if (ok) runtime->exchange = WindowExchangeKind::NONE;
    }
    if (ok) { g_last_window_stop_epoch = stop.epoch; g_last_window_stop_receipt = receipt; }
    return ok;
}

// One migration fanout: requests[server] to each server (the PREPARE fragments
// differ per server; every other request is the same for all), then one ACK
// per server naming the plan's epochs and its global move count.
static bool migration_exchange_each(std::vector<std::vector<char>> requests,
                                    ann::MessageType expected_type,
                                    uint64_t expected_epoch_from,
                                    uint64_t expected_epoch_to,
                                    uint64_t expected_move_count,
                                    double deadline_sec,
                                    uint16_t accept_request_id = 0) {
    std::vector<bool> received(g_daemon.n_servers, false);
    uint32_t received_count = 0;
    bool all_success = true;
    if (requests.size() != g_daemon.n_servers) return false;
    for (size_t server = 0; server < g_daemon.n_servers; ++server) {
        queue_or_send_server_message(server, std::move(requests[server]));
    }

    const auto deadline = steady_clock::now() +
        duration_cast<steady_clock::duration>(duration<double>(deadline_sec));
    while (received_count < g_daemon.n_servers &&
           steady_clock::now() < deadline) {
        auto accept = [&](size_t server, const char* data, size_t size) {
            const auto actual_type = ann::get_message_type(data, size);
            // A workload may leave a late asynchronous stats snapshot in the
            // shared response queue. It is unrelated to this migration round
            // and must not be mistaken for a malformed migration ACK.
            if (actual_type == ann::MessageType::GET_STATS_RESPONSE) {
                return true;
            }
            // The late ACK of an earlier ACCEPT attempt: consumed, not counted.
            if (stale_accept_ack(server, data, size, accept_request_id)) {
                return true;
            }
            if (server >= g_daemon.n_servers || received[server] ||
                size != sizeof(ann::MembershipMigrationAck)) {
                return false;
            }
            const auto* ack = reinterpret_cast<
                const ann::MembershipMigrationAck*>(data);
            const bool rollback_noop =
                expected_type ==
                    ann::MessageType::MIGRATION_ROLLBACK_ACK &&
                ack->status == 0 && ack->epoch_from == expected_epoch_from &&
                ack->epoch_to == expected_epoch_from && ack->move_count == 0;
            // A server that had already accepted this plan answers a retried
            // ACCEPT with success and move_count 0 (it dropped the plan).
            const bool accept_repeat =
                expected_type == ann::MessageType::MIGRATION_ACCEPT_ACK &&
                ack->status == 0 && ack->epoch_from == expected_epoch_from &&
                ack->epoch_to == expected_epoch_to && ack->move_count == 0;
            const bool expected_plan =
                ack->epoch_from == expected_epoch_from &&
                ack->epoch_to == expected_epoch_to &&
                ack->move_count == expected_move_count;
            if (ack->msg_type != expected_type ||
                ack->server_rank != server ||
                (ack->status == 0 &&
                 !expected_plan && !rollback_noop && !accept_repeat)) {
                return false;
            }
            if (ack->status != 0) all_success = false;
            received[server] = true;
            ++received_count;
            return true;
        };

        if (g_daemon.main_response_queue) {
            ServerResponse response;
            while (g_daemon.main_response_queue->try_pop(response)) {
                if (!accept(response.server_idx,
                            response.response_data.data(),
                            response.response_data.size())) {
                    daemon_output_line(
                        "[ERROR] Invalid migration response envelope");
                    return false;
                }
            }
        } else {
            for (size_t server = 0; server < g_daemon.sockets.size(); ++server) {
                if (received[server]) continue;
                zmq::message_t empty, body;
                auto first = g_daemon.sockets[server].recv(
                    empty, zmq::recv_flags::dontwait);
                if (!first) continue;
                auto second = g_daemon.sockets[server].recv(
                    body, zmq::recv_flags::none);
                if (!second || !accept(
                                   server,
                                   static_cast<const char*>(body.data()),
                                   body.size())) {
                    daemon_output_line(
                        "[ERROR] Invalid migration response envelope");
                    return false;
                }
            }
        }
        std::this_thread::sleep_for(milliseconds(1));
    }
    if (received_count != g_daemon.n_servers) {
        daemon_output_line("[ERROR] Timeout waiting for migration responses");
        return false;
    }
    return all_success;
}

// The same request to every server, with the active plan's deadline.
static bool migration_exchange(const std::vector<char>& request,
                               ann::MessageType expected_type,
                               uint64_t expected_epoch_from,
                               uint64_t expected_epoch_to,
                               uint64_t expected_move_count,
                               uint16_t accept_request_id = 0) {
    return migration_exchange_each(std::vector<std::vector<char>>(g_daemon.n_servers, request), expected_type,
                                   expected_epoch_from, expected_epoch_to, expected_move_count,
                                   plan_exchange_deadline_sec(), accept_request_id);
}

// Sends the move template inline to every server; each source server answers
// with its current rows of the moved nodes.
static bool migration_refresh_exchange(
    const MembershipMigrationPlan& move_template,
    MembershipMigrationPlan& refreshed) {
    std::vector<char> request;
    {
        std::string error;
        if (!serialize_membership_migration_plan(move_template, request, error)) return false;
        request.insert(request.begin(), static_cast<char>(ann::MessageType::MIGRATION_REFRESH_INLINE_REQUEST));
    }
    // The deadline: the template to every server, and per move a row allowance
    // and, from own-only sources, the vector with its digest.
    const uint64_t moves = move_template.moves.size();
    const uint64_t vector_row_bytes = g_daemon.own_only_storage
        ? sizeof(uint64_t) + uint64_t{g_daemon.dim} * sizeof(float) : 0;
    g_migration_payload.refresh_request_bytes = uint64_t{request.size()} * g_daemon.n_servers;
    g_migration_payload.refresh_deadline_s = migration_deadline_sec(
        g_migration_refresh_deadline_sec,
        g_migration_payload.refresh_request_bytes + moves * (kMigrationRowAllowanceBytes + vector_row_bytes));
    const auto started = steady_clock::now();
    for (size_t server = 0; server < g_daemon.n_servers; ++server) {
        queue_or_send_server_message(server, std::vector<char>(request));
    }

    std::vector<bool> received(g_daemon.n_servers, false);
    uint32_t received_count = 0;
    bool all_success = true;
    std::unordered_map<uint32_t, MembershipMove> rows;
    const uint32_t current_n_base =
        g_daemon.n_base + g_daemon.total_inserted;
    const auto deadline = started +
        duration_cast<steady_clock::duration>(duration<double>(g_migration_payload.refresh_deadline_s));
    auto accept = [&](size_t server, const char* data, size_t size) {
        if (stale_accept_ack(server, data, size, 0)) return true;
        if (server >= g_daemon.n_servers || received[server] ||
            size < sizeof(ann::MembershipMigrationRefreshHeader)) {
            return false;
        }
        ann::MembershipMigrationRefreshHeader header{};
        std::memcpy(&header, data, sizeof(header));
        if (header.msg_type !=
                ann::MessageType::MIGRATION_REFRESH_RESPONSE ||
            header.server_rank != server ||
            header.epoch_from != move_template.epoch_from ||
            header.epoch_to != move_template.epoch_to ||
            header.payload_bytes != size - sizeof(header)) {
            return false;
        }
        if (header.status != 0) {
            all_success = false;
        } else if (header.move_count == 0) {
            if (header.payload_bytes != 0) return false;
        } else {
            MembershipMigrationPlan fragment;
            std::string error;
            if (!load_membership_migration_plan_bytes(
                    data + sizeof(header),
                    static_cast<size_t>(header.payload_bytes),
                    current_n_base, g_daemon.n_servers,
                    fragment, error) ||
                fragment.epoch_from != move_template.epoch_from ||
                fragment.epoch_to != move_template.epoch_to ||
                fragment.moves.size() != header.move_count) {
                return false;
            }
            for (auto& move : fragment.moves) {
                if (move.from_partition != server ||
                    !rows.emplace(move.global_id, std::move(move)).second) {
                    return false;
                }
            }
        }
        received[server] = true;
        ++received_count;
        g_migration_payload.refresh_response_bytes += size;
        return true;
    };

    while (received_count < g_daemon.n_servers &&
           steady_clock::now() < deadline) {
        if (g_daemon.main_response_queue) {
            ServerResponse response;
            while (g_daemon.main_response_queue->try_pop(response)) {
                if (!accept(response.server_idx,
                            response.response_data.data(),
                            response.response_data.size())) {
                    daemon_output_line(
                        "[ERROR] Invalid migration refresh response");
                    return false;
                }
            }
        } else {
            for (size_t server = 0; server < g_daemon.sockets.size(); ++server) {
                if (received[server]) continue;
                zmq::message_t empty, body;
                auto first = g_daemon.sockets[server].recv(
                    empty, zmq::recv_flags::dontwait);
                if (!first) continue;
                auto second = g_daemon.sockets[server].recv(
                    body, zmq::recv_flags::none);
                if (!second || !accept(
                                   server,
                                   static_cast<const char*>(body.data()),
                                   body.size())) {
                    daemon_output_line(
                        "[ERROR] Invalid migration refresh response");
                    return false;
                }
            }
        }
        std::this_thread::sleep_for(milliseconds(1));
    }
    g_migration_payload.refresh_s = duration<double>(steady_clock::now() - started).count();
    if (received_count != g_daemon.n_servers || !all_success ||
        rows.size() != move_template.moves.size()) {
        daemon_output_line(
            "[ERROR] Migration refresh did not return every current row");
        return false;
    }

    refreshed = move_template;
    for (auto& move : refreshed.moves) {
        auto it = rows.find(move.global_id);
        if (it == rows.end() ||
            it->second.from_partition != move.from_partition ||
            it->second.to_partition != move.to_partition) {
            daemon_output_line(
                "[ERROR] Migration refresh identity mismatch");
            return false;
        }
        move.adjacency = std::move(it->second.adjacency);
        move.adjacency_digest = it->second.adjacency_digest;
        // Own-only sources ship each node's vector; full replication none.
        move.vector = std::move(it->second.vector);
        move.vector_digest = it->second.vector_digest;
    }
    return true;
}

bool coordinator_membership_migration_rollback() {
    // Only the active (unaccepted) plan can roll back: ACCEPT clears it.
    if (g_coordinator_migration_plan.moves.empty()) return true;
    if (g_coordinator_accept_sent) {
        daemon_output_line(
            "[ERROR] ACCEPT was already sent for the active migration (epoch " +
            std::to_string(g_coordinator_migration_plan.epoch_from) + " -> " +
            std::to_string(g_coordinator_migration_plan.epoch_to) +
            "); a server that accepted it cannot roll back. Retry migration-accept");
        return false;
    }
    // The request names the plan, so a server that accepted that epoch refuses
    // it and one that never prepared it answers a no-op.
    std::vector<char> request(
        1 + 2 * sizeof(uint64_t),
        static_cast<char>(ann::MessageType::MIGRATION_ROLLBACK_REQUEST));
    std::memcpy(request.data() + 1, &g_coordinator_migration_plan.epoch_from, sizeof(uint64_t));
    std::memcpy(request.data() + 1 + sizeof(uint64_t), &g_coordinator_migration_plan.epoch_to,
                sizeof(uint64_t));
    const bool ok = migration_exchange(
        request, ann::MessageType::MIGRATION_ROLLBACK_ACK,
        g_coordinator_migration_plan.epoch_from,
        g_coordinator_migration_plan.epoch_to,
        g_coordinator_migration_plan.moves.size());
    if (!ok) return false;
    for (const auto& move : g_coordinator_migration_plan.moves) {
        g_daemon.membership.at(move.global_id) = move.from_partition;
    }
    g_coordinator_ownership_epoch =
        g_coordinator_migration_plan.epoch_from;
    g_coordinator_migration_plan = MembershipMigrationPlan{};
    return true;
}

bool coordinator_membership_migration_accept() {
    if (g_coordinator_migration_plan.moves.empty() ||
        g_coordinator_ownership_epoch !=
            g_coordinator_migration_plan.epoch_to) {
        daemon_output_line("[ERROR] No committed migration to accept");
        return false;
    }
    try {
        g_coordinator_accepted_epochs.reserve(g_coordinator_accepted_epochs.size() + 1);
    } catch (const std::exception&) { return false; }
    // From the first fanout on, some server may have accepted: the plan can
    // only be completed by ACCEPT, never rolled back.
    g_coordinator_accept_sent = true;
    bool ok = false;
    for (int attempt = 0; attempt < kMigrationAcceptAttempts && !ok; ++attempt) {
        if (attempt > 0) {
            daemon_output_line("[MIGRATION] ACCEPT incomplete; retrying on every server "
                               "(an accepted server answers with an idempotent success), attempt " +
                               std::to_string(attempt + 1));
            std::this_thread::sleep_for(milliseconds(100));
        }
        // A fresh attempt id (never 0): the replies of an attempt that timed
        // out are discarded when they arrive, so attempt k+1 counts only its
        // own ACKs. Request: type, u16 attempt id.
        if (++g_migration_accept_request_id == 0) g_migration_accept_request_id = 1;
        const uint16_t request_id = g_migration_accept_request_id;
        std::vector<char> request(1 + sizeof(request_id));
        request[0] = static_cast<char>(ann::MessageType::MIGRATION_ACCEPT_REQUEST);
        std::memcpy(request.data() + 1, &request_id, sizeof(request_id));
        ok = migration_exchange(
            request, ann::MessageType::MIGRATION_ACCEPT_ACK,
            g_coordinator_migration_plan.epoch_from,
            g_coordinator_migration_plan.epoch_to,
            g_coordinator_migration_plan.moves.size(), request_id);
    }
    if (!ok) return false;
    // The accepted plan and its rows are dropped; only the epoch is kept.
    g_coordinator_accepted_epochs.push_back(g_coordinator_migration_plan.epoch_to);
    g_coordinator_migration_plan = MembershipMigrationPlan{};
    g_coordinator_accept_sent = false;
    return true;
}

// PREPARE of `plan` on every server, which then becomes the active plan with
// its payload dropped. A plan that carries vectors (own-only sources shipped
// them at REFRESH, or a version-3 file) goes out as per-target fragments (wire
// version 4): every server gets the whole move list with every row and digest,
// each vector goes only to its target. A plan without vectors goes out as the
// same version-2 plan to every server (full replication). False before any
// fanout (nothing to roll back) when the plan cannot be sent.
static bool prepare_fanout(MembershipMigrationPlan plan) {
    const bool vectors = !plan.moves.empty() && !plan.moves.front().vector.empty();
    if (g_daemon.own_only_storage && !vectors) {
        daemon_output_line("[ERROR] Own-only servers install each moved node's original vector: this plan carries "
                           "none (use migration-apply --refresh, whose sources ship the vectors)");
        return false;
    }
    if (vectors && plan.moves.front().vector.size() != g_daemon.dim) {
        daemon_output_line("[ERROR] Migration plan vector dimension " +
                           std::to_string(plan.moves.front().vector.size()) + " differs from the cluster's " +
                           std::to_string(g_daemon.dim));
        return false;
    }
    std::string error;
    std::vector<char> shared;
    if (!vectors && !serialize_membership_migration_plan(plan, shared, error)) {
        daemon_output_line("[ERROR] " + error);
        return false;
    }
    std::vector<std::vector<char>> requests(g_daemon.n_servers);
    std::vector<uint64_t> request_bytes(g_daemon.n_servers, 0), vector_bytes_sent(g_daemon.n_servers, 0);
    for (size_t server = 0; server < g_daemon.n_servers; ++server) {
        std::vector<char> fragment;
        // The float bytes the serializer wrote into this server's fragment.
        if (vectors && !serialize_membership_migration_fragment(plan, static_cast<uint32_t>(server), fragment, error,
                                                                &vector_bytes_sent[server])) {
            daemon_output_line("[ERROR] " + error);
            return false;
        }
        const std::vector<char>& payload = vectors ? fragment : shared;
        auto& request = requests[server];
        request.reserve(1 + payload.size());
        request.push_back(static_cast<char>(ann::MessageType::MIGRATION_PREPARE_INLINE_REQUEST));
        request.insert(request.end(), payload.begin(), payload.end());
        request_bytes[server] = request.size();
    }
    g_migration_payload.moves = plan.moves.size();
    g_migration_payload.vector_bytes = membership_migration_vector_bytes(plan);
    g_migration_payload.prepare_request_bytes = 0;
    g_migration_payload.prepare_vector_bytes = 0;
    for (size_t server = 0; server < g_daemon.n_servers; ++server) {
        g_migration_payload.prepare_request_bytes += request_bytes[server];
        g_migration_payload.prepare_vector_bytes += vector_bytes_sent[server];
    }
    g_migration_payload.prepare_request_bytes_by_server = std::move(request_bytes);
    g_migration_payload.prepare_vector_bytes_by_server = std::move(vector_bytes_sent);
    g_migration_payload.exchange_deadline_s = plan_exchange_deadline_sec();
    // COMMIT, ROLLBACK and ACCEPT need only epochs, ids, owners and digests.
    strip_membership_migration_payload(plan);
    g_coordinator_migration_plan = std::move(plan);
    const auto started = steady_clock::now();
    const bool prepared = migration_exchange_each(
        std::move(requests), ann::MessageType::MIGRATION_PREPARE_ACK,
        g_coordinator_migration_plan.epoch_from, g_coordinator_migration_plan.epoch_to,
        g_coordinator_migration_plan.moves.size(), g_migration_payload.exchange_deadline_s);
    g_migration_payload.prepare_s = duration<double>(steady_clock::now() - started).count();
    return prepared;
}

// The plan file's moves must start at the coordinator's epoch and from the
// owners it records.
static bool migration_plan_matches_coordinator(const MembershipMigrationPlan& plan) {
    if (plan.epoch_from != g_coordinator_ownership_epoch) {
        daemon_output_line("[ERROR] Coordinator ownership epoch mismatch");
        return false;
    }
    for (const auto& move : plan.moves) {
        if (move.global_id >= g_daemon.membership.size() ||
            g_daemon.membership.at(move.global_id) != move.from_partition) {
            daemon_output_line(
                "[ERROR] Coordinator membership/from mismatch at node " +
                std::to_string(move.global_id));
            return false;
        }
    }
    return true;
}

// A plan file that carries vectors (wire version 3): each must be its node's
// original vector (the base.npy row, or the inserted vector), byte for byte,
// before anything is sent. A target installs it, and ACCEPT then frees the
// source's copy: a stale or corrupt file vector would replace the only correct
// one. (The sources check the same at PREPARE against their own vectors.)
static bool migration_plan_vectors_are_originals(const MembershipMigrationPlan& plan) {
    const uint64_t dim = g_daemon.dim;
    for (const auto& move : plan.moves) {
        const uint64_t gid = move.global_id;
        const float* original = nullptr;
        if (gid < g_daemon.n_base) {
            if ((gid + 1) * dim <= g_daemon.base.size()) original = g_daemon.base.data() + gid * dim;
        } else if ((gid - g_daemon.n_base + 1) * dim <= g_daemon.all_inserted_vectors.size()) {
            original = g_daemon.all_inserted_vectors.data() + (gid - g_daemon.n_base) * dim;
        }
        if (original == nullptr) {
            daemon_output_line("[ERROR] Migration plan carries a vector for node " + std::to_string(gid) +
                               ", whose original vector the coordinator does not hold: refused");
            return false;
        }
        if (move.vector.size() != dim ||
            std::memcmp(move.vector.data(), original, dim * sizeof(float)) != 0) {
            daemon_output_line("[ERROR] Migration plan vector of node " + std::to_string(gid) +
                               " differs from its original vector (base.npy row or inserted vector): refused "
                               "before any fanout");
            return false;
        }
    }
    return true;
}

// Who asks for a PREPARE: the admin command migration-apply, refused while a
// controller phase is active (a manual MOVE between two search
// commands of a phase moves the ownership epoch away from the controller's
// history, so every later plan is refused and a burst awaiting its verdict can
// neither accept nor revert), or the controller's own fence, the only MOVE a
// phase makes.
enum class MigrationOrigin { Admin, ControllerFence };

static bool migration_preconditions(const char* what, MigrationOrigin origin) {
    if (origin == MigrationOrigin::Admin && window_controller_active()) {
        daemon_output_line(std::string("[ERROR] Migration ") + what +
                           " refused: a controller phase is active (controller-stop first; controller-abort "
                           "ends a terminal one)");
        return false;
    }
    if (!g_coordinator_migration_plan.moves.empty()) {
        daemon_output_line(
            "[ERROR] A migration journal is already active; rollback or reset first");
        return false;
    }
    if (!g_daemon.pending_insert_response_map.empty() ||
        !g_daemon.pending_backlink_map.empty() ||
        !g_daemon.pending_broadcast_map.empty()) {
        daemon_output_line(std::string("[ERROR] Migration ") + what +
                           " requires empty pending maintenance state");
        return false;
    }
    return true;
}

bool coordinator_membership_migration_prepare(const std::string& path) {
    if (!migration_preconditions("PREPARE", MigrationOrigin::Admin)) return false;
    MembershipMigrationPlan plan;
    std::string error;
    if (!load_membership_migration_plan(
            path, g_daemon.n_base + g_daemon.total_inserted, g_daemon.n_servers, plan, error)) {
        daemon_output_line("[ERROR] " + error);
        return false;
    }
    if (!migration_plan_matches_coordinator(plan)) return false;
    if (!plan.moves.front().vector.empty() && !migration_plan_vectors_are_originals(plan)) return false;
    g_migration_payload = MigrationPayloadStats{};
    if (!prepare_fanout(std::move(plan))) {
        if (g_coordinator_migration_plan.moves.empty()) return false;  // nothing was sent
        daemon_output_line(
            "[ERROR] Distributed PREPARE failed; attempting rollback");
        if (!coordinator_membership_migration_rollback()) {
            daemon_output_line("[ERROR] PREPARE rollback incomplete");
        }
        return false;
    }
    return true;
}

// The refreshed PREPARE of a loaded skeleton plan (both callers below).
static bool prepare_refreshed(const MembershipMigrationPlan& move_template) {
    if (!migration_plan_matches_coordinator(move_template)) return false;
    g_migration_payload = MigrationPayloadStats{};
    g_migration_payload.moves = move_template.moves.size();
    MembershipMigrationPlan refreshed;
    if (!migration_refresh_exchange(move_template, refreshed)) {
        return false;
    }
    if (!prepare_fanout(std::move(refreshed))) {
        if (g_coordinator_migration_plan.moves.empty()) return false;  // nothing was sent
        daemon_output_line(
            "[ERROR] Distributed refreshed PREPARE failed; attempting rollback");
        if (!coordinator_membership_migration_rollback()) {
            daemon_output_line("[ERROR] Refreshed PREPARE rollback incomplete");
        }
        return false;
    }
    return true;
}

bool coordinator_membership_migration_prepare_current(const std::string& path) {
    if (!migration_preconditions("REFRESH", MigrationOrigin::Admin)) return false;
    MembershipMigrationPlan move_template;
    std::string error;
    if (!load_membership_migration_plan(path, g_daemon.n_base + g_daemon.total_inserted, g_daemon.n_servers,
                                        move_template, error)) {
        daemon_output_line("[ERROR] " + error);
        return false;
    }
    return prepare_refreshed(move_template);
}

bool coordinator_membership_migration_prepare_controller(const std::vector<char>& plan_bytes) {
    if (!migration_preconditions("REFRESH", MigrationOrigin::ControllerFence)) return false;
    MembershipMigrationPlan move_template;
    std::string error;
    if (!load_membership_migration_plan_bytes(plan_bytes.data(), plan_bytes.size(),
                                              g_daemon.n_base + g_daemon.total_inserted, g_daemon.n_servers,
                                              move_template, error)) {
        daemon_output_line("[ERROR] " + error);
        return false;
    }
    return prepare_refreshed(move_template);
}

bool coordinator_membership_migration_commit() {
    if (g_coordinator_migration_plan.moves.empty()) {
        daemon_output_line("[ERROR] No prepared migration to commit");
        return false;
    }

    std::vector<char> commit(
        1, static_cast<char>(ann::MessageType::MIGRATION_COMMIT_REQUEST));
    const bool committed = migration_exchange(
        commit, ann::MessageType::MIGRATION_COMMIT_ACK,
        g_coordinator_migration_plan.epoch_from,
        g_coordinator_migration_plan.epoch_to,
        g_coordinator_migration_plan.moves.size());
    if (!committed) {
        daemon_output_line(
            "[ERROR] Distributed COMMIT failed; attempting rollback");
        if (!coordinator_membership_migration_rollback()) {
            daemon_output_line("[ERROR] COMMIT rollback incomplete");
        }
        return false;
    }
    for (const auto& move : g_coordinator_migration_plan.moves) {
        g_daemon.membership.at(move.global_id) = move.to_partition;
    }
    g_coordinator_ownership_epoch =
        g_coordinator_migration_plan.epoch_to;
    return true;
}

bool coordinator_membership_migration_is_prepared() {
    return !g_coordinator_migration_plan.moves.empty() &&
           g_coordinator_ownership_epoch ==
               g_coordinator_migration_plan.epoch_from;
}

uint64_t coordinator_membership_ownership_epoch() {
    return g_coordinator_ownership_epoch;
}

static bool migration_apply_prepare_commit(const std::string& path, bool refresh) {
    return (refresh ? coordinator_membership_migration_prepare_current(path)
                    : coordinator_membership_migration_prepare(path)) &&
           coordinator_membership_migration_commit();
}

static std::string migration_payload_json(const MigrationPayloadStats& p) {
    std::ostringstream json;
    json << "{\"moves\":" << p.moves << ",\"fence_payload_bytes\":" << p.total_bytes()
         << ",\"refresh_request_bytes\":" << p.refresh_request_bytes
         << ",\"refresh_response_bytes\":" << p.refresh_response_bytes
         << ",\"prepare_request_bytes\":" << p.prepare_request_bytes
         << ",\"vector_bytes\":" << p.vector_bytes << ",\"prepare_vector_bytes\":" << p.prepare_vector_bytes
         << migration_payload_by_server_fields(p)
         << ",\"refresh_deadline_s\":" << window_control_detail::exact_number(p.refresh_deadline_s)
         << ",\"exchange_deadline_s\":" << window_control_detail::exact_number(p.exchange_deadline_s)
         << ",\"refresh_s\":" << window_control_detail::exact_number(p.refresh_s)
         << ",\"prepare_s\":" << window_control_detail::exact_number(p.prepare_s) << "}";
    return json.str();
}

void coordinator_stats_window_send_begin(uint64_t window_epoch, uint32_t nodes) {
    ann::StatsWindowBeginRequest begin{};
    begin.msg_type = ann::MessageType::STATS_WINDOW_BEGIN_REQUEST;
    begin.version = kStatsWindowBeginVersion;
    begin.nodes = nodes;
    begin.window_epoch = window_epoch;
    std::vector<char> request(sizeof(begin));
    std::memcpy(request.data(), &begin, sizeof(begin));
    for (size_t server = 0; server < g_daemon.n_servers; ++server) {
        queue_or_send_server_message(server, std::vector<char>(request));
    }
}

void coordinator_stats_window_send_harvest(uint64_t window_epoch, uint32_t nodes, uint32_t sections) {
    // Snapshot wire v3 request with the repair controller's structure sections (by its weight).
    WindowSnapshotRequest snapshot;
    snapshot.epoch = window_epoch;
    snapshot.nodes = nodes;
    snapshot.version = kWindowSnapshotVersion;
    snapshot.sections = static_cast<uint16_t>(sections);
    std::vector<char> request(sizeof(snapshot));
    std::memcpy(request.data(), &snapshot, sizeof(snapshot));
    for (size_t server = 0; server < g_daemon.n_servers; ++server)
        queue_or_send_server_message(server, std::vector<char>(request));
}

static void mark_pending_insert_done_if_ready(WorkloadItem& item) {
    if (!item.backlink_done || !item.broadcast_done) return;
    item.insert_phase = tide::InsertPhase::DONE;
    item.done = true;
    if (item.assigned_id != 0) {
        g_daemon.pending_mgr.set_committed(item.assigned_id);
        g_daemon.pending_mgr.remove(item.assigned_id);
    }
}

static uint32_t queue_exact_backlinks_from_pending_insert(
    const ann::InsertResponseHeader& hdr,
    const std::vector<ann::CrossPartitionEdge>& cross_edges,
    uint32_t* exact_edges_queued = nullptr
) {
    if (exact_edges_queued) *exact_edges_queued = 0;
    auto pit = g_daemon.pending_insert_response_map.find(hdr.request_id);
    if (pit == g_daemon.pending_insert_response_map.end()) {
        g_daemon.maintenance.orphan_insert_responses++;
        return 0;
    }

    uint32_t workload_idx = pit->second;
    g_daemon.pending_insert_response_map.erase(pit);
    if (workload_idx >= g_daemon.pending_workloads.size()) {
        g_daemon.maintenance.orphan_insert_responses++;
        return 0;
    }

    auto& item = g_daemon.pending_workloads[workload_idx];
    if (item.insert_response_seen) {
        g_daemon.maintenance.orphan_insert_responses++;
        return 0;
    }

    item.insert_response_seen = true;
    item.assigned_id = hdr.assigned_id;
    item.pending_backlinks = cross_edges;
    item.backlinks_sent = 0;
    item.backlinks_acked = 0;

    std::unordered_map<uint32_t, std::vector<uint32_t>> server_to_to_ids;
    uint32_t exact_edges = 0;
    for (const auto& edge : cross_edges) {
        if (edge.from_partition == item.target_partition) continue;
        uint32_t server_idx = owner_label_checked(edge.from_partition, g_daemon.n_servers, "backlink fanout");
        server_to_to_ids[server_idx].push_back(edge.from_id);
        exact_edges++;
    }
    if (exact_edges_queued) *exact_edges_queued = exact_edges;

    if (server_to_to_ids.empty()) {
        item.backlink_done = true;
        mark_pending_insert_done_if_ready(item);
        return 0;
    }

    item.backlink_done = false;
    for (auto& [server_idx, to_ids] : server_to_to_ids) {
        auto batch_data = ann::serialize_batch_backlink_request(
            item.id,
            item.assigned_id,
            item.target_partition,
            g_daemon.dim,
            g_daemon.insert_R,
            g_daemon.insert_alpha,
            to_ids,
            item.query.data()
        );
        queue_or_send_server_message(server_idx, std::move(batch_data));
        item.backlinks_sent++;
        g_daemon.maintenance.backlink_batches_sent++;
        g_daemon.maintenance.backlink_edges_sent += to_ids.size();
    }
    g_daemon.pending_backlink_map[item.id] = workload_idx;
    return item.backlinks_sent;
}

static bool drain_pending_ops_for_admin() {
    auto& workloads = g_daemon.pending_workloads;
    bool has_pending = !g_daemon.pending_insert_response_map.empty() ||
                       !g_daemon.pending_backlink_map.empty() ||
                       !g_daemon.pending_broadcast_map.empty();
    const uint64_t pending_insert_responses_at_start = g_daemon.pending_insert_response_map.size();
    const uint64_t pending_backlink_batches_at_start = g_daemon.pending_backlink_map.size();
    const uint64_t pending_broadcasts_at_start = g_daemon.pending_broadcast_map.size();

    if (has_pending) {
        daemon_output_line("Draining pending (ir=" +
            std::to_string(g_daemon.pending_insert_response_map.size()) +
            " bl=" +
            std::to_string(g_daemon.pending_backlink_map.size()) +
            " bc=" + std::to_string(g_daemon.pending_broadcast_map.size()) + ")");
    }

    auto flush_pending_ops = [&]() -> uint32_t {
        // No per-server staged ops to flush. Maintenance
        // requests already go directly through g_daemon.server_queues when
        // produced, so nothing to drain here.
        return 0;
    };

    uint64_t settled_inserts = 0;
    uint64_t settled_insert_responses = 0;
    uint64_t settled_backlink_batches = 0;
    uint64_t settled_broadcast_acks = 0;
    auto mark_pending_committed_by_barrier = [&]() {
        // Count what is still pending before the barrier marks it committed.
        settled_insert_responses = g_daemon.pending_insert_response_map.size();
        for (const auto& [batch_id, idx] : g_daemon.pending_backlink_map) {
            (void)batch_id;
            if (idx >= workloads.size()) continue;
            const auto& item = workloads[idx];
            if (item.backlinks_sent > item.backlinks_acked)
                settled_backlink_batches += item.backlinks_sent - item.backlinks_acked;
        }
        for (const auto& [node_id, idx] : g_daemon.pending_broadcast_map) {
            (void)node_id;
            if (idx >= workloads.size()) continue;
            const auto& item = workloads[idx];
            if (g_daemon.n_servers > item.broadcasts_acked)
                settled_broadcast_acks += g_daemon.n_servers - item.broadcasts_acked;
        }
        for (const auto& item : workloads) {
            if (item.type == tide::WorkloadType::INSERT && !item.done) settled_inserts++;
        }
        for (auto& item : workloads) {
            if (item.type != tide::WorkloadType::INSERT) continue;
            item.backlink_done = true;
            item.broadcast_done = true;
            item.insert_phase = tide::InsertPhase::DONE;
            item.done = true;
            if (item.assigned_id != 0) {
                g_daemon.pending_mgr.set_committed(item.assigned_id);
                g_daemon.pending_mgr.remove(item.assigned_id);
            }
        }
        g_daemon.pending_insert_response_map.clear();
        g_daemon.pending_backlink_map.clear();
        g_daemon.pending_broadcast_map.clear();
    };

    uint64_t responses_seen = 0;
    uint64_t insert_responses_seen = 0;
    uint64_t exact_backlink_batches_queued = 0;
    uint64_t exact_backlink_edges_queued = 0;
    uint64_t backlink_acks_seen = 0;
    uint64_t broadcast_acks_seen = 0;
    uint64_t broadcast_acks_accounted = 0;

    std::vector<bool> drain_acked(g_daemon.n_servers, false);
    uint32_t drain_acks = 0;
    bool generated_exact_backlinks = false;

    auto record_drain_stats = [&]() {
        broadcast_acks_accounted = 0;
        for (const auto& item : workloads) {
            if (item.type == tide::WorkloadType::INSERT) {
                broadcast_acks_accounted += item.broadcasts_acked;
            }
        }
        g_last_drain_stats.responses_seen = responses_seen;
        g_last_drain_stats.insert_responses_seen = insert_responses_seen;
        g_last_drain_stats.exact_backlink_batches_queued =
            exact_backlink_batches_queued;
        g_last_drain_stats.exact_backlink_edges_queued =
            exact_backlink_edges_queued;
        g_last_drain_stats.backlink_acks_seen = backlink_acks_seen;
        g_last_drain_stats.broadcast_acks_seen = broadcast_acks_seen;
        g_last_drain_stats.broadcast_acks_accounted =
            broadcast_acks_accounted;
        g_last_drain_stats.settled_inserts = settled_inserts;
        g_last_drain_stats.settled_insert_responses = settled_insert_responses;
        g_last_drain_stats.settled_backlink_batches = settled_backlink_batches;
        g_last_drain_stats.settled_broadcast_acks = settled_broadcast_acks;
        g_last_drain_stats.pending_insert_responses_at_start = pending_insert_responses_at_start;
        g_last_drain_stats.pending_backlink_batches_at_start = pending_backlink_batches_at_start;
        g_last_drain_stats.pending_broadcasts_at_start = pending_broadcasts_at_start;
    };

    auto handle_response = [&](size_t server_idx, const char* data, size_t size) -> bool {
        if (size == 0) return false;
        responses_seen++;
        auto mt = ann::get_message_type(data, size);

        if (mt == ann::MessageType::DRAIN_ACK) {
            if (server_idx < drain_acked.size() && !drain_acked[server_idx]) {
                drain_acked[server_idx] = true;
                drain_acks++;
            }
            return true;
        } else if (mt == ann::MessageType::INSERT_RESPONSE) {
            insert_responses_seen++;
            ann::InsertResponseHeader hdr;
            std::vector<uint32_t> neighbors;
            std::vector<ann::CrossPartitionEdge> cross_edges;
            if (ann::deserialize_insert_response(data, size, hdr, neighbors, cross_edges)) {
                uint32_t exact_edges = 0;
                uint32_t queued = queue_exact_backlinks_from_pending_insert(
                    hdr, cross_edges, &exact_edges);
                exact_backlink_edges_queued += exact_edges;
                if (queued > 0) {
                    generated_exact_backlinks = true;
                    exact_backlink_batches_queued += queued;
                }
            }
            return true;
        } else if (mt == ann::MessageType::BATCH_BACKLINK_ACK) {
            uint32_t batch_id = 0;
            if (size >= 8) {
                std::memcpy(&batch_id, data + 4, sizeof(uint32_t));
            }
            auto bit = g_daemon.pending_backlink_map.find(batch_id);
            if (bit == g_daemon.pending_backlink_map.end()) {
                // No pending entry: counted, never matched to an insert.
                g_daemon.maintenance.orphan_backlink_acks++;
                return false;
            }
            backlink_acks_seen++;
            {
                uint32_t idx = bit->second;
                if (idx >= workloads.size()) {
                    g_daemon.pending_backlink_map.erase(bit);
                    return true;
                }
                auto& item = workloads[idx];
                item.backlinks_acked++;
                if (item.backlinks_acked >= item.backlinks_sent) {
                    item.backlink_done = true;
                    g_daemon.pending_backlink_map.erase(bit);
                    mark_pending_insert_done_if_ready(item);
                }
            }
            return true;
        } else if (mt == ann::MessageType::VECTOR_BROADCAST_ACK) {
            uint32_t node_id = 0;
            if (size >= 8) {
                std::memcpy(&node_id, data + 4, sizeof(uint32_t));
            }
            auto bit = g_daemon.pending_broadcast_map.find(node_id);
            if (bit == g_daemon.pending_broadcast_map.end()) {
                g_daemon.maintenance.orphan_broadcast_acks++;
                return false;
            }
            broadcast_acks_seen++;
            {
                uint32_t idx = bit->second;
                if (idx >= workloads.size()) {
                    g_daemon.pending_broadcast_map.erase(bit);
                    return true;
                }
                auto& item = workloads[idx];
                item.broadcasts_acked++;
                if (item.broadcasts_acked >= g_daemon.n_servers) {
                    item.broadcast_done = true;
                    g_daemon.pending_broadcast_map.erase(bit);
                    mark_pending_insert_done_if_ready(item);
                }
            }
            return true;
        }
        return false;
    };

    std::vector<zmq::pollitem_t> pitems;
    if (!g_daemon.main_response_queue) {
        for (auto& s : g_daemon.sockets)
            pitems.push_back({static_cast<void*>(s), 0, ZMQ_POLLIN, 0});
    }

    while (true) {
        uint32_t flushed = flush_pending_ops();
        if (flushed > 0) {
            daemon_output_line("Flushed " + std::to_string(flushed) +
                               " staged maintenance ops");
        }

        drain_acks = 0;
        std::fill(drain_acked.begin(), drain_acked.end(), false);
        generated_exact_backlinks = false;
        for (size_t i = 0; i < g_daemon.n_servers; i++) {
            std::vector<char> drain_msg(1, static_cast<char>(ann::MessageType::DRAIN_REQUEST));
            queue_or_send_server_message(i, std::move(drain_msg));
        }

        auto last_progress = std::chrono::steady_clock::now();
        auto last_log = last_progress;
        while (drain_acks < g_daemon.n_servers) {
            auto now = std::chrono::steady_clock::now();
            auto idle_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_progress).count();
            if (idle_seconds > 120) {
                daemon_output_line("Drain stalled (no responses for 120s). ir=" +
                    std::to_string(g_daemon.pending_insert_response_map.size()) +
                    " bl=" + std::to_string(g_daemon.pending_backlink_map.size()) +
                    " bc=" + std::to_string(g_daemon.pending_broadcast_map.size()) +
                    " drain_acks=" + std::to_string(drain_acks) + "/" +
                    std::to_string(g_daemon.n_servers) +
                    " responses=" + std::to_string(responses_seen) +
                    " ins_resp=" + std::to_string(insert_responses_seen) +
                    " bl_acks=" + std::to_string(backlink_acks_seen) +
                    " bc_acks=" + std::to_string(broadcast_acks_seen));
                record_drain_stats();
                return false;
            }
            auto log_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_log).count();
            if (log_seconds >= 5) {
                daemon_output_line("Drain progress. ir=" +
                    std::to_string(g_daemon.pending_insert_response_map.size()) +
                    " bl=" + std::to_string(g_daemon.pending_backlink_map.size()) +
                    " bc=" + std::to_string(g_daemon.pending_broadcast_map.size()) +
                    " drain_acks=" + std::to_string(drain_acks) + "/" +
                    std::to_string(g_daemon.n_servers) +
                    " responses=" + std::to_string(responses_seen) +
                    " ins_resp=" + std::to_string(insert_responses_seen) +
                    " bl_acks=" + std::to_string(backlink_acks_seen) +
                    " bc_acks=" + std::to_string(broadcast_acks_seen));
                last_log = now;
            }

            if (g_daemon.main_response_queue) {
                ServerResponse resp;
                bool got_any = false;
                while (g_daemon.main_response_queue->try_pop(resp)) {
                    got_any = true;
                    if (handle_response(resp.server_idx,
                                        resp.response_data.data(),
                                        resp.response_data.size())) {
                        last_progress = std::chrono::steady_clock::now();
                    }
                }
                if (!got_any) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } else {
                zmq::poll(pitems, std::chrono::milliseconds(1));
                for (size_t s = 0; s < g_daemon.sockets.size(); s++) {
                    if (!(pitems[s].revents & ZMQ_POLLIN)) continue;
                    while (true) {
                        zmq::message_t ef, resp;
                        auto rc = g_daemon.sockets[s].recv(ef, zmq::recv_flags::dontwait);
                        if (!rc) break;
                        g_daemon.sockets[s].recv(resp, zmq::recv_flags::dontwait);
                        if (handle_response(s,
                            static_cast<const char*>(resp.data()), resp.size())) {
                            last_progress = std::chrono::steady_clock::now();
                        }
                    }
                }
            }
        }

        if (!generated_exact_backlinks) {
            mark_pending_committed_by_barrier();
            break;
        }

        daemon_output_line("Drain queued exact backlink batches; running another barrier. queued=" +
                           std::to_string(exact_backlink_batches_queued));
    }

    daemon_output_line("Drain complete. responses=" +
                       std::to_string(responses_seen) +
                       " ins_resp=" + std::to_string(insert_responses_seen) +
                       " exact_bl_queued=" + std::to_string(exact_backlink_batches_queued) +
                       " exact_bl_edges=" + std::to_string(exact_backlink_edges_queued) +
                       " bl_acks=" + std::to_string(backlink_acks_seen) +
                       " bc_acks=" + std::to_string(broadcast_acks_seen) +
                       " settled_at_barrier: inserts=" + std::to_string(settled_inserts) +
                       " ir=" + std::to_string(settled_insert_responses) +
                       " bl=" + std::to_string(settled_backlink_batches) +
                       " bc=" + std::to_string(settled_broadcast_acks));
    record_drain_stats();
    return true;
}

// What the last drain found pending when it began (compact JSON object).
static std::string drain_pending_at_start_json() {
    std::ostringstream json;
    json << "{\"insert_responses\":" << g_last_drain_stats.pending_insert_responses_at_start
         << ",\"backlink_batches\":" << g_last_drain_stats.pending_backlink_batches_at_start
         << ",\"broadcasts\":" << g_last_drain_stats.pending_broadcasts_at_start << "}";
    return json.str();
}

// The work of the last drain (controller-start's reply): what was pending when
// it began, the replies it consumed, the exact backlinks it queued and the
// broadcast ACKs accounted for the last insert command's workloads.
static std::string drain_work_json() {
    std::ostringstream json;
    json << "{\"pending_at_start\":" << drain_pending_at_start_json()
         << ",\"responses\":" << g_last_drain_stats.responses_seen
         << ",\"insert_responses\":" << g_last_drain_stats.insert_responses_seen
         << ",\"backlink\":{\"exact_batches\":" << g_last_drain_stats.exact_backlink_batches_queued
         << ",\"exact_edges\":" << g_last_drain_stats.exact_backlink_edges_queued
         << ",\"acks\":" << g_last_drain_stats.backlink_acks_seen << "}"
         << ",\"broadcast\":{\"acks\":" << g_last_drain_stats.broadcast_acks_seen
         << ",\"acks_accounted\":" << g_last_drain_stats.broadcast_acks_accounted << "}}";
    return json.str();
}

// SHA-256 of the coordinator's membership as little-endian uint32 owners, one
// per id (the bytes of a membership .npy's data): controller-start reports the
// membership its window binds, so a client can check the phase's start state
// without a membership-check (which would drain first).
static std::string membership_sha256() {
    std::vector<uint8_t> bytes(g_daemon.membership.size() * sizeof(uint32_t));
    for (size_t i = 0; i < g_daemon.membership.size(); ++i) {
        const uint32_t owner = g_daemon.membership[i];
        for (size_t b = 0; b < sizeof(uint32_t); ++b) bytes[i * 4 + b] = static_cast<uint8_t>(owner >> (8 * b));
    }
    return sha256_hex(bytes.data(), bytes.size());
}

// Process a single daemon command
// Returns false if should quit
bool process_daemon_command(const std::string& cmd) {
    std::istringstream iss(cmd);
    std::string action;
    iss >> action;

    // Inserts and the controller alternate: an insert phase, then a
    // controller phase (controller-start drains the inserts' maintenance and
    // freezes the window's node count), then inserts again after
    // controller-stop. A graph mutation under an active controller would grow
    // the membership past the window's node count and change rows a pending
    // verdict or revert fence relies on, so these commands are refused while a
    // controller is active.
    if (window_controller_active() &&
        (action == "insert" || action == "mixed" || action == "recreate-head")) {
        daemon_output_line("Error: " + action + " requires a stopped controller (controller-stop; "
                           "controller-abort ends a terminal one): inserts and the runtime controller alternate");
        return true;
    }
    // A manual MOVE inside a controller phase would move the
    // ownership epoch away from the controller's history: every later plan is
    // refused, and a burst awaiting its verdict can neither accept nor revert.
    // (migration-accept and migration-rollback stay allowed: they confirm
    // ownership after a terminal controller left a plan unresolved.)
    if (window_controller_active() && action == "migration-apply") {
        daemon_output_line("Error: migration-apply requires a stopped controller (controller-stop; "
                           "controller-abort ends a terminal one): the controller's fence is the only MOVE "
                           "of a controller phase");
        return true;
    }
    // Nor between migration-apply and its verdict: an insert would add
    // backlinks to rows the plan installed (lost again by a rollback), and the
    // empty-maintenance check runs only at PREPARE.
    if (!g_coordinator_migration_plan.moves.empty() &&
        (action == "insert" || action == "mixed" || action == "recreate-head")) {
        daemon_output_line("Error: " + action + " requires no active migration (migration-accept or "
                           "migration-rollback first): the plan of epoch " +
                           std::to_string(g_coordinator_migration_plan.epoch_from) + " -> " +
                           std::to_string(g_coordinator_migration_plan.epoch_to) + " is unresolved");
        return true;
    }

    if (action == "migration-apply") {
        // migration-apply [--refresh] <plan>: stop-the-world PREPARE + COMMIT of a
        // migration plan file; migration-accept or migration-rollback then resolves
        // it (insert, mixed and recreate-head are refused until then). The
        // coordinator reads the file and sends every server its PREPARE fragment
        // inline. --refresh takes the rows (and, from own-only sources, the
        // vectors) from the moved nodes' current owners, as the controller does:
        // a skeleton plan (a version-2 plan whose moves carry no row, like the
        // controller's plan.bin) needs it, and so
        // does every own-only (--pq) cluster unless the file carries each moved
        // node's vector (version 3). A file's vectors must equal the originals
        // (base.npy, the inserted vectors) byte for byte, and each source checks
        // the vector digest against its own vector. A version-1/2 plan that
        // carries the rows applies as written only under full replication
        // (--full-vectors).
        std::string first, path, extra;
        iss >> first;
        const bool refresh = first == "--refresh";
        if (refresh) iss >> path; else path = first;
        if (path.empty() || path.rfind("--", 0) == 0 || (iss >> extra)) {
            daemon_output_line(
                "Error: migration-apply requires exactly one plan path (migration-apply [--refresh] <plan>)");
            return true;
        }
        const bool ok = drain_pending_ops_for_admin() &&
                        migration_apply_prepare_commit(path, refresh);
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"" << (ok ? "ok" : "error")
             << "\",\"action\":\"migration-apply\""
             << ",\"refresh\":" << (refresh ? "true" : "false")
             << ",\"ownership_epoch\":"
             << g_coordinator_ownership_epoch
             << ",\"move_count\":"
             << g_coordinator_migration_plan.moves.size()
             << ",\"payload\":" << migration_payload_json(coordinator_membership_migration_payload()) << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "migration-rollback") {
        std::string extra;
        if (iss >> extra) {
            daemon_output_line("Error: migration-rollback takes no arguments");
            return true;
        }
        const bool had_active =
            !g_coordinator_migration_plan.moves.empty();
        const bool ok = had_active && drain_pending_ops_for_admin() &&
                        coordinator_membership_migration_rollback();
        // Every server confirmed the rollback: an ownership-ambiguous
        // terminal controller is no longer ambiguous (controller-abort may end it).
        if (ok && g_coordinator_migration_plan.moves.empty())
            window_controller_confirm_ownership(g_coordinator_ownership_epoch, "migration-rollback");
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"" << (ok ? "ok" : "error")
             << "\",\"action\":\"migration-rollback\""
             << ",\"ownership_epoch\":"
             << g_coordinator_ownership_epoch << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "migration-accept") {
        std::string extra;
        if (iss >> extra) {
            daemon_output_line("Error: migration-accept takes no arguments");
            return true;
        }
        const uint64_t accepted_epoch = g_coordinator_ownership_epoch;
        const bool ok = drain_pending_ops_for_admin() &&
                        coordinator_membership_migration_accept();
        // Every server confirmed the ACCEPT: an ownership-ambiguous terminal
        // controller is no longer ambiguous (controller-abort may end it).
        if (ok && g_coordinator_migration_plan.moves.empty())
            window_controller_confirm_ownership(g_coordinator_ownership_epoch, "migration-accept");
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"" << (ok ? "ok" : "error")
             << "\",\"action\":\"migration-accept\""
             << ",\"ownership_epoch\":"
             << g_coordinator_ownership_epoch
             << ",\"accepted_epoch\":" << accepted_epoch
             << ",\"history_depth\":"
             << g_coordinator_accepted_epochs.size() << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "membership-check") {
        // Every server's state against the coordinator's membership: per id the
        // owner label (mem_at) on every server, a local row exactly on its owner
        // and, under own-only storage, the original vector exactly on its owner
        // and a PQ code everywhere; the free row-slot pool's invariant. Drains
        // first (the servers answer only when quiescent); refused while a
        // migration is prepared or committed but not resolved.
        // membership-check [--write <path.npy>]: --write also saves the
        // coordinator's membership (uint32, one owner per id) for the caller.
        std::string token, write_path, extra, error;
        if (iss >> token) {
            if (token != "--write" || !(iss >> write_path) || (iss >> extra))
                error = "membership-check takes no arguments but --write <path.npy>";
        }
        if (!error.empty()) {}
        else if (!g_coordinator_migration_plan.moves.empty())
            error = "a migration is active; accept or roll it back first";
        else if (g_daemon.membership.size() > std::numeric_limits<uint32_t>::max())
            error = "too many ids";
        const uint32_t ids = static_cast<uint32_t>(g_daemon.membership.size());
        std::vector<ann::MembershipCheckResponse> replies(g_daemon.n_servers);
        bool ok = false;
        if (error.empty() && !drain_pending_ops_for_admin()) error = "drain failed";
        if (error.empty()) {
            std::vector<char> request(1 + sizeof(ids) + uint64_t{ids} * sizeof(uint32_t));
            request[0] = static_cast<char>(ann::MessageType::MEMBERSHIP_CHECK_REQUEST);
            std::memcpy(request.data() + 1, &ids, sizeof(ids));
            if (ids) std::memcpy(request.data() + 1 + sizeof(ids), g_daemon.membership.data(),
                                 uint64_t{ids} * sizeof(uint32_t));
            ok = control_exchange(request, ann::MessageType::MEMBERSHIP_CHECK_RESPONSE,
                [&](size_t server, const char* data, size_t size) {
                    if (size != sizeof(ann::MembershipCheckResponse) || server >= replies.size()) return false;
                    std::memcpy(&replies[server], data, size);
                    return replies[server].server_rank == server;
                });
            if (!ok) error = "not every server answered the membership check";
        }
        if (ok && !write_path.empty()) {
            try {
                ann::write_npy_uint32(write_path, g_daemon.membership, true);
            } catch (const std::exception& e) {
                ok = false;
                error = std::string("cannot write the membership: ") + e.what();
            }
        }
        std::vector<uint64_t> owned_by(g_daemon.n_servers, 0);
        for (uint32_t owner : g_daemon.membership)
            if (owner < owned_by.size()) ++owned_by[owner];
        // The original vector of every id (base.npy rows, then the inserted
        // vectors in id order), summed per owner: what each own-only server
        // must hold, byte for byte.
        const uint32_t dim = g_daemon.dim;
        const bool originals = g_daemon.base.size() == uint64_t{g_daemon.n_base} * dim &&
            g_daemon.all_inserted_vectors.size() >= (uint64_t{ids} - std::min<uint64_t>(ids, g_daemon.n_base)) * dim;
        std::vector<uint64_t> expected_digest(g_daemon.n_servers, 0);
        if (g_daemon.own_only_storage && originals) {
            for (uint32_t id = 0; id < ids; ++id) {
                const uint32_t owner = g_daemon.membership[id];
                if (owner >= expected_digest.size()) continue;
                const float* vector = id < g_daemon.n_base
                    ? g_daemon.base.data() + uint64_t{id} * dim
                    : g_daemon.all_inserted_vectors.data() + uint64_t{id - g_daemon.n_base} * dim;
                expected_digest[owner] += node_vector_hash(id, vector, dim);
            }
        }
        auto hex = [](uint64_t value) {
            std::ostringstream out;
            out << std::hex << std::setw(16) << std::setfill('0') << value;
            return out.str();
        };
        bool consistent = ok;
        std::ostringstream servers;
        for (size_t rank = 0; ok && rank < replies.size(); ++rank) {
            const auto& r = replies[rank];
            // Own-only: every own vector equals the original, byte for byte.
            const bool vectors_exact = !g_daemon.own_only_storage ||
                (originals && r.vector_digest == expected_digest[rank]);
            const bool server_ok = r.status == 0 && r.ids == ids && r.server_ids == ids &&
                r.membership_mismatches == 0 && r.mapping_mismatches == 0 && r.vector_mismatches == 0 &&
                r.missing_codes == 0 && r.pool_ok == 1 && r.owned == owned_by[rank] && r.mapped == r.owned &&
                (r.own_only == 0 || r.vectors == r.owned) && r.ownership_epoch == g_coordinator_ownership_epoch &&
                (r.own_only == 1) == g_daemon.own_only_storage && vectors_exact;
            consistent = consistent && server_ok;
            servers << (rank ? "," : "") << "{\"rank\":" << rank << ",\"consistent\":" << (server_ok ? "true" : "false")
                    << ",\"status\":" << unsigned(r.status) << ",\"own_only\":" << (r.own_only ? "true" : "false")
                    << ",\"ids\":" << r.ids << ",\"server_ids\":" << r.server_ids
                    << ",\"ownership_epoch\":" << r.ownership_epoch
                    << ",\"owned\":" << r.owned << ",\"mapped\":" << r.mapped << ",\"vectors\":" << r.vectors
                    << ",\"membership_mismatches\":" << r.membership_mismatches
                    << ",\"mapping_mismatches\":" << r.mapping_mismatches
                    << ",\"vector_mismatches\":" << r.vector_mismatches
                    << ",\"missing_codes\":" << r.missing_codes
                    << ",\"vectors_exact\":" << (vectors_exact ? "true" : "false")
                    << ",\"vector_digest\":\"" << hex(r.vector_digest) << "\""
                    << ",\"expected_vector_digest\":\"" << hex(expected_digest[rank]) << "\""
                    << ",\"first_mismatch\":" << (r.first_mismatch == UINT32_MAX ? -1 : int64_t{r.first_mismatch})
                    << ",\"n_local\":" << r.n_local << ",\"free_slots\":" << r.free_slots
                    << ",\"pool_ok\":" << (r.pool_ok ? "true" : "false") << "}";
        }
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"" << (ok ? "ok" : "error") << "\",\"action\":\"membership-check\""
             << ",\"error\":" << window_control_detail::json_quote(error)
             << ",\"ids\":" << ids << ",\"ownership_epoch\":" << g_coordinator_ownership_epoch
             << ",\"own_only_storage\":" << (g_daemon.own_only_storage ? "true" : "false")
             << ",\"original_vectors\":" << (originals ? "true" : "false")
             << ",\"written\":" << window_control_detail::json_quote(ok ? write_path : std::string())
             << ",\"consistent\":" << (consistent ? "true" : "false") << ",\"coordinator_owned_by_rank\":[";
        for (size_t rank = 0; rank < owned_by.size(); ++rank) json << (rank ? "," : "") << owned_by[rank];
        json << "],\"servers\":[" << servers.str() << "]}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "controller-start") {
        // controller-start <journal_dir> [window_sec=S] [cooldown_sec=S]
        //     [min_window_queries=N] [proposal_query_limit=N] [repair=algo:local[,key:value...]]
        // Starts a controller phase: pending insert maintenance is drained
        // first (the start is refused if the drain fails), the window covers
        // every id the coordinator knows (load-time and inserted), and the row
        // bytes come from the phase's first merged harvest (no row-bytes file).
        // insert, mixed and recreate-head are refused until controller-stop.
        WindowControllerConfig config;
        config.repair.enabled = true;  // repair defaults unless a repair= token overrides them
        std::string error;
        if (!(iss >> config.journal_dir)) {
            error = "controller-start requires <journal_dir> [window_sec=S] [cooldown_sec=S] "
                    "[min_window_queries=N] [proposal_query_limit=N] [repair=algo:local[,key:value...]]";
        } else if (config.journal_dir.size() >= 4 &&
                   config.journal_dir.compare(config.journal_dir.size() - 4, 4, ".npy") == 0) {
            error = "controller-start takes no row-bytes file (the row bytes come from the "
                    "harvested owner histogram): controller-start <journal_dir> [options]";
        }
        std::vector<std::string> seen;
        std::string option;
        while (error.empty() && iss >> option) {
            const size_t equals = option.find('=');
            const std::string key = equals == std::string::npos ? option : option.substr(0, equals);
            const std::string value = equals == std::string::npos ? "" : option.substr(equals + 1);
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) { error = "duplicate option " + key; break; }
            seen.push_back(key);
            auto number = [&](double& out) {
                char* end = nullptr;
                out = std::strtod(value.c_str(), &end);
                return !value.empty() && end && *end == '\0' && std::isfinite(out);
            };
            auto count = [&](uint64_t& out) {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) return false;
                try { out = std::stoull(value); } catch (...) { return false; }
                return true;
            };
            if (key == "window_sec") { if (!number(config.window_seconds)) error = "window_sec needs a number"; }
            else if (key == "cooldown_sec") { if (!number(config.cooldown_seconds)) error = "cooldown_sec needs a number"; }
            else if (key == "min_window_queries") { if (!count(config.min_window_queries)) error = "min_window_queries needs a count"; }
            else if (key == "proposal_query_limit") { if (!count(config.proposal_query_limit)) error = "proposal_query_limit needs a count"; }
            else if (key == "repair") { window_repair_parse_token(value, &config.repair, &error); }
            else error = "unknown option " + key;
        }
        config.start_ownership_epoch = g_coordinator_ownership_epoch;
        config.ranks = g_daemon.n_servers;
        config.vector_bytes_per_row = g_daemon.own_only_storage ? g_daemon.dim * static_cast<uint32_t>(sizeof(float)) : 0;
        if (error.empty() && !window_owners_valid(static_cast<uint32_t>(g_daemon.n_servers)))
            error = "the repair controller supports 2 to 64 servers";
        if (error.empty() && !g_coordinator_migration_plan.moves.empty())
            error = "a migration is active; accept or roll it back first";
        if (error.empty() && window_controller_active())
            error = "controller already active";
        // Phase boundary: every insert's maintenance (insert responses,
        // backlinks, broadcasts) is applied on every server before the first
        // window, so each server knows every id the window counts.
        bool drained = false;
        double drain_seconds = 0.0;
        if (error.empty()) {
            const auto drain_start = std::chrono::steady_clock::now();
            drained = drain_pending_ops_for_admin();
            drain_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - drain_start).count();
            if (!drained) error = "the drain before the controller phase failed";
        }
        bool ok = error.empty() && window_controller_start(config, g_daemon.membership.size(), &error) != nullptr;
        if (ok) g_last_window_stop_receipt.clear();
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"" << (ok ? "ok" : "error") << "\",\"action\":\"controller-start\""
             << ",\"error\":" << window_control_detail::json_quote(ok ? std::string() : error)
             << ",\"journal_dir\":" << window_control_detail::json_quote(config.journal_dir)
             << ",\"nodes\":" << g_daemon.membership.size()
             << ",\"n_base\":" << g_daemon.n_base << ",\"inserted\":" << g_daemon.total_inserted
             << ",\"drained\":" << (drained ? "true" : "false")
             << ",\"drain_seconds\":" << window_control_detail::exact_number(drain_seconds)
             << ",\"drain_settled_at_barrier\":{\"inserts\":" << g_last_drain_stats.settled_inserts
             << ",\"insert_responses\":" << g_last_drain_stats.settled_insert_responses
             << ",\"backlink_batches\":" << g_last_drain_stats.settled_backlink_batches
             << ",\"broadcast_acks\":" << g_last_drain_stats.settled_broadcast_acks << "}"
             << ",\"drain\":" << (drained ? drain_work_json() : std::string("null"))
             << ",\"membership_sha256\":\"" << membership_sha256() << "\""
             << ",\"ownership_epoch\":" << g_coordinator_ownership_epoch;
        if (ok) json << ',' << window_controller_status_json(*window_controller_active());
        json << '}';
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "controller-stop") {
        // Closes the servers' stats windows (when this controller opened one) and stops.
        // Refused while a burst is unresolved: its verdict or inverse MOVE is still owed.
        auto* runtime = window_controller_active();
        const bool was_active = runtime != nullptr;
        const bool unresolved = runtime && window_controller_has_unresolved_action(*runtime);
        std::string receipt, extra;
        bool ok = !(iss >> extra) && !unresolved;
        if (ok && runtime && !runtime->stats_off_confirmed) ok = stop_stats_windows(receipt);
        ok = ok && window_controller_stop();
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"" << (ok ? "ok" : "error") << "\",\"action\":\"controller-stop\""
             << ",\"unresolved_action\":" << (unresolved ? "true" : "false")
             << ",\"was_active\":" << (was_active ? "true" : "false") << receipt << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "controller-abort") {
        // Ends a TERMINAL controller (stopped on error) whose ownership is not
        // ambiguous, without a coordinator restart, so inserts and reset run
        // again. Whatever burst the terminal left in place stays; the retained
        // action history is dropped, so the next controller-start needs
        // ownership epoch 0 (reset first after any accepted MOVE). Refused for
        // a running controller (controller-stop), while ownership is
        // ambiguous (complete migration-accept or migration-rollback first),
        // and while a migration is active.
        auto* runtime = window_controller_active();
        const bool was_active = runtime != nullptr;
        const std::string terminal_reason = runtime ? runtime->terminal_reason : std::string();
        std::string error, receipt, extra;
        if (iss >> extra) error = "controller-abort takes no arguments";
        else if (runtime && !window_controller_abort_allowed(*runtime, &error)) {}
        else if (runtime && !g_coordinator_migration_plan.moves.empty())
            error = "a migration is active; accept or roll it back first";
        if (error.empty() && runtime && !runtime->stats_off_confirmed && !stop_stats_windows(receipt, true))
            error = "the all-rank stats STOP failed";
        if (error.empty()) window_controller_abort(&error);
        const bool ok = error.empty();
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"" << (ok ? "ok" : "error") << "\",\"action\":\"controller-abort\""
             << ",\"error\":" << window_control_detail::json_quote(error)
             << ",\"was_active\":" << (was_active ? "true" : "false")
             << ",\"terminal_reason\":" << window_control_detail::json_quote(terminal_reason)
             << ",\"ownership_epoch\":" << g_coordinator_ownership_epoch << receipt << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "controller-status") {
        const WindowControllerRuntime* runtime = window_controller_active();
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\":\"ok\",\"action\":\"controller-status\""
             << ",\"active\":" << (runtime != nullptr ? "true" : "false")
             << ",\"ownership_epoch\":" << coordinator_membership_ownership_epoch()
             << ",\"accepted_migrations\":" << g_coordinator_accepted_epochs.size();
        if (!runtime && !g_last_window_stop_receipt.empty()) json << g_last_window_stop_receipt;
        if (runtime != nullptr) json << ',' << window_controller_status_json(*runtime);
        json << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "quit" || action == "exit" || action == "shutdown") {
        // quit/exit stop the coordinator only. shutdown first drains pending
        // maintenance and stops every server through a quiescent SHUTDOWN
        // exchange.
        bool ok = true;
        if (action == "shutdown") {
            daemon_output_line("Coordinator draining before server shutdown...");
            ok = drain_pending_ops_for_admin();
            if (ok) {
                // The daemon server threads own these sockets during normal
                // workloads. Stop and join them before the synchronous
                // shutdown exchange reads from the same sockets.
                stop_daemon_server_threads();
                ok = send_admin_command(
                    ann::MessageType::SHUTDOWN_REQUEST);
            }
        }
        daemon_output_line(
            ok ? "Coordinator shutting down..."
               : "[ERROR] Graceful shutdown incomplete");
        return false;
    }
    else if (action == "baseline" || action == "recall") {
        uint32_t queries = g_daemon.n_queries;
        std::string gt_path = g_daemon.data_dir + "/groundtruth.npy";
        uint32_t ping_count = 0;
        uint32_t ping_batch = 0;
        std::string query_file;        // self-recall: alt query source
        bool self_recall = false;      // self-recall: GT = expected assigned IDs
        uint32_t query_offset = 0;     // skip first N rows of query_file
        uint32_t self_base_id = 0;     // expected_id = base + query_index; 0 = use g_daemon.n_base
        std::string self_ids_path;     // self-query: row i of the query file is node ids[i]'s vector
        uint32_t search_l = 0;         // --search-l: this command's search list size (0: the start-up L)

        // Reset the per-command merge flag so a --no-merge run does not leak into later
        // recall commands. Recall can also use --no-merge for direct validation.
        g_daemon.no_merge = false;
        g_daemon.trace_routes = true;

        std::string token;
        while (iss >> token) {
            if (token == "--queries") {
                iss >> queries;
            } else if (token == "--groundtruth") {
                iss >> gt_path;
            } else if (token == "--no-groundtruth") {
                gt_path.clear();  // unscored: recall is reported as 0 and the controller sees no recall
            } else if (token == "--batch-size") {
                iss >> g_daemon.batch_size;
            } else if (token == "--dispatch-window") {
                iss >> g_daemon.dispatch_window;
            } else if (token == "--ping-baseline") {
                iss >> ping_count;
            } else if (token == "--ping-batch") {
                iss >> ping_batch;
            } else if (token == "--debug") {
                g_daemon.debug_mode = true;
            } else if (token == "--no-merge") {
                g_daemon.no_merge = true;
            } else if (token == "--no-route-trace") {
                g_daemon.trace_routes = false;
            } else if (token == "--query-file") {
                iss >> query_file;
            } else if (token == "--query-offset") {
                iss >> query_offset;
            } else if (token == "--self-recall") {
                self_recall = true;
            } else if (token == "--self-recall-base-id") {
                iss >> self_base_id;
            } else if (token == "--self-ids") {
                iss >> self_ids_path;
            } else if (token == "--search-l") {
                if (!(iss >> search_l) || search_l < g_daemon.K || search_l > kMaxCommandSearchL) {
                    daemon_output_line("Error: --search-l needs a search list size from K (" +
                                       std::to_string(g_daemon.K) + ") to " + std::to_string(kMaxCommandSearchL));
                    return true;
                }
            }
        }
        std::vector<uint32_t> self_ids;
        if (!self_ids_path.empty()) {
            try {
                uint32_t count = 0;
                self_ids = ann::read_npy_uint32(self_ids_path, count);
            } catch (const std::exception& e) {
                daemon_output_line(std::string("Error: cannot read --self-ids: ") + e.what());
                return true;
            }
            if (self_ids.empty()) {
                daemon_output_line("Error: --self-ids names no id");
                return true;
            }
        }

        // Run ping baseline before search if requested
        if (ping_count > 0) {
            run_ping_baseline(ping_count, ping_batch);
        }

        run_search_workload_ex(queries, gt_path, query_file, query_offset,
                               self_recall, self_base_id, self_ids, search_l);
    }
    else if (action == "insert") {
        uint32_t count = 0;
        std::string insert_file;
        uint32_t start_id = 0;

        // Parse: insert <count> [--file PATH] [--start-id N] [--no-backlink] [--no-broadcast]
        iss >> count;
        g_daemon.no_backlink = false;
        g_daemon.no_broadcast = false;
        g_daemon.insert_search_only = false;
        std::string token;
        while (iss >> token) {
            if (token == "--file") {
                iss >> insert_file;
            } else if (token == "--start-id") {
                iss >> start_id;
            } else if (token == "--batch-size") {
                iss >> g_daemon.batch_size;
            } else if (token == "--dispatch-window") {
                iss >> g_daemon.dispatch_window;
            } else if (token == "--no-backlink") {
                g_daemon.no_backlink = true;
            } else if (token == "--no-broadcast") {
                g_daemon.no_broadcast = true;
            } else if (token == "--search-only") {
                g_daemon.insert_search_only = true;
            } else if (token == "--debug") {
                g_daemon.debug_mode = true;
            }
        }

        if (count == 0) {
            daemon_output_line("Error: insert requires count > 0");
        } else {
            run_insert_workload(count, insert_file, start_id);
        }
    }
    else if (action == "mixed") {
        uint32_t queries = g_daemon.n_queries;
        uint32_t insert_ratio = 50;  // Default: 50% inserts
        std::string gt_path = g_daemon.data_dir + "/groundtruth.npy";
        std::string insert_gt_path;  // Insert GT file
        std::string insert_file;     // Insert file path (optional)
        uint32_t insert_start_id = 0;
        double inserted_query_ratio_pct = 10.0;

        // Reset per-command flags to default (important for repeated runs)
        g_daemon.no_backlink = false;
        g_daemon.no_broadcast = false;
        g_daemon.no_merge = false;

        // Parse: mixed --queries N --insert-ratio P [--groundtruth PATH] [--insert-gt PATH]
        //        [--insert-file PATH] [--insert-start-id N] [--inserted-query-ratio PCT]
        //        [--no-backlink] [--no-broadcast] [--no-merge]
        std::string token;
        while (iss >> token) {
            if (token == "--queries") {
                iss >> queries;
            } else if (token == "--insert-ratio") {
                iss >> insert_ratio;
            } else if (token == "--groundtruth") {
                iss >> gt_path;
            } else if (token == "--insert-gt") {
                iss >> insert_gt_path;
            } else if (token == "--insert-file") {
                iss >> insert_file;
            } else if (token == "--insert-start-id") {
                iss >> insert_start_id;
            } else if (token == "--no-backlink") {
                g_daemon.no_backlink = true;
            } else if (token == "--no-broadcast") {
                g_daemon.no_broadcast = true;
            } else if (token == "--batch-size") {
                iss >> g_daemon.batch_size;
            } else if (token == "--dispatch-window") {
                iss >> g_daemon.dispatch_window;
            } else if (token == "--no-merge") {
                g_daemon.no_merge = true;
            } else if (token == "--inserted-query-ratio") {
                iss >> inserted_query_ratio_pct;
            } else if (token == "--debug") {
                g_daemon.debug_mode = true;
            }
        }

        if (insert_ratio < 1 || insert_ratio > 99) {
            daemon_output_line("Error: insert-ratio must be between 1 and 99");
        } else if (inserted_query_ratio_pct < 0.0 || inserted_query_ratio_pct > 100.0) {
            daemon_output_line("Error: inserted-query-ratio must be between 0 and 100");
        } else {
            run_mixed_workload(queries, insert_ratio, gt_path, insert_gt_path,
                               insert_file, insert_start_id,
                               inserted_query_ratio_pct / 100.0);
        }
    }
    else if (action == "recreate-head") {
        // Rebuild head index with current base + inserted vectors
        uint32_t total_nodes = g_daemon.n_base + g_daemon.total_inserted;

        // g_daemon.membership already contains correct partitions for inserted nodes
        // (set in pipeline_core.cpp INSERT_RESPONSE handler via membership_vec)
        std::vector<uint32_t>& full_membership = g_daemon.membership;

        // Create combined base vectors
        std::vector<float> full_base = g_daemon.base;
        full_base.insert(full_base.end(),
                        g_daemon.all_inserted_vectors.begin(),
                        g_daemon.all_inserted_vectors.end());

        // Rebuild head index
        g_daemon.head_index.rebuild(full_base.data(), total_nodes, g_daemon.dim,
                                    full_membership, g_daemon.n_servers);

        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\": \"ok\", \"action\": \"recreate-head\", "
             << "\"n_samples\": " << g_daemon.head_index.n_samples() << ", "
             << "\"total_nodes\": " << total_nodes << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "status") {
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        const auto& m = g_daemon.maintenance;
        json << "{\"status\": \"ok\", \"n_base\": " << g_daemon.n_base
             << ", \"total_inserted\": " << g_daemon.total_inserted
             << ", \"n_samples\": " << g_daemon.head_index.n_samples()
             << ", \"maintenance\": {\"scope\": \"since-daemon-start\""
             << ", \"backlink_edges_sent\": " << m.backlink_edges_sent
             << ", \"backlink_batches_sent\": " << m.backlink_batches_sent
             << ", \"pending\": {\"insert_responses\": " << g_daemon.pending_insert_response_map.size()
             << ", \"backlink_inserts\": " << g_daemon.pending_backlink_map.size()
             << ", \"broadcast_inserts\": " << g_daemon.pending_broadcast_map.size() << "}"
             << ", \"orphan_responses\": {\"insert_responses\": " << m.orphan_insert_responses
             << ", \"backlink_acks\": " << m.orphan_backlink_acks
             << ", \"broadcast_acks\": " << m.orphan_broadcast_acks << "}}}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "stats") {
        std::vector<ann::StatsResponseHeader> snapshots;
        // Admin stats is the consuming snapshot: it returns and resets the interval maxima.
        const bool complete = query_server_stats(snapshots, true);
        daemon_output_line("===JSON_RESULT===");
        daemon_output_line(stats_snapshots_json(snapshots, complete));
        daemon_output_line("===JSON_END===");
    }
    else if (action == "reset") {
        if (window_controller_active()) {
            daemon_output_line("Error: reset requires a stopped controller (controller-stop; "
                               "controller-abort ends a terminal one)");
            return true;
        }
        // Reset all servers to initial state (reload from disk)
        bool ok = send_admin_command(ann::MessageType::RESET_REQUEST);

        // Also reset daemon state (inserted vectors + pending inserts)
        g_daemon.total_inserted = 0;
        g_daemon.all_inserted_vectors.clear();
        g_daemon.pending_mgr.clear();
        g_daemon.pending_insert_response_map.clear();
        g_daemon.pending_backlink_map.clear();
        g_daemon.pending_broadcast_map.clear();
        g_daemon.pending_workloads.clear();
        g_coordinator_migration_plan = MembershipMigrationPlan{};
        g_coordinator_accepted_epochs.clear();
        g_coordinator_ownership_epoch = 0;
        g_coordinator_accept_sent = false;

        // The servers reload their load-time partitions at epoch 0. Reload the
        // load-time membership file the same way: a committed migration moved
        // owners that a resize alone would keep. Inserted nodes are dropped.
        std::string reload_error;
        try {
            uint32_t loaded_count = 0;
            std::vector<uint32_t> loaded =
                ann::read_npy_int64_as_uint32(g_daemon.membership_path, loaded_count);
            std::string label_error;
            if (loaded.size() < g_daemon.n_base) {
                reload_error = "membership file has " + std::to_string(loaded.size()) +
                               " rows, fewer than n_base " + std::to_string(g_daemon.n_base);
            } else {
                loaded.resize(g_daemon.n_base);
                if (!owner_labels_valid(loaded, g_daemon.n_servers, &label_error))
                    reload_error = label_error;  // never fold a label onto another server
                else
                    g_daemon.membership = std::move(loaded);
            }
        } catch (const std::exception& e) {
            reload_error = e.what();
        }
        if (!reload_error.empty()) {
            ok = false;
            daemon_output_line("[ERROR] reset could not reload the membership file " +
                               g_daemon.membership_path + ": " + reload_error);
        }
        // Reset membership array to original size (remove inserted nodes)
        g_daemon.membership.resize(g_daemon.n_base);
        // The retained controller history describes moves the reset undid; the
        // next controller-start binds a fresh one at epoch 0.
        window_controller_reset_history();

        // The load-time head index (the runtime's ghost sample), as at start:
        // a rebuilt one samples other entry points and changes every search
        // result. Without a loaded one (an in-process test), rebuild it.
        if (g_daemon.head_index_at_load.n_samples() > 0)
            g_daemon.head_index = g_daemon.head_index_at_load;
        else
            g_daemon.head_index.rebuild(g_daemon.base.data(), g_daemon.n_base, g_daemon.dim,
                                        g_daemon.membership, g_daemon.n_servers);

        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\": \"" << (ok ? "ok" : "error") << "\", "
             << "\"action\": \"reset\", "
             << "\"n_base\": " << g_daemon.n_base << ", "
             << "\"membership_size\": " << g_daemon.membership.size() << ", "
             << "\"total_inserted\": " << g_daemon.total_inserted << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "drain") {
        const auto drain_start = std::chrono::steady_clock::now();
        bool ok = drain_pending_ops_for_admin();
        const double drain_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - drain_start).count();
        daemon_output_line("===JSON_RESULT===");
        std::ostringstream json;
        json << "{\"status\": \"" << (ok ? "ok" : "error") << "\", "
             << "\"action\": \"drain\", "
             << "\"drain_seconds\": " << window_control_detail::exact_number(drain_seconds) << ", "
             << "\"pending_at_start\": " << drain_pending_at_start_json() << ", "
             << "\"maintenance\": {"
             << "\"responses\": " << g_last_drain_stats.responses_seen << ", "
             << "\"insert_responses\": " << g_last_drain_stats.insert_responses_seen << ", "
             << "\"backlink\": {"
             << "\"exact_batches\": " << g_last_drain_stats.exact_backlink_batches_queued << ", "
             << "\"exact_edges\": " << g_last_drain_stats.exact_backlink_edges_queued << ", "
             << "\"acks\": " << g_last_drain_stats.backlink_acks_seen << "}, "
             << "\"broadcast\": {\"acks\": " << g_last_drain_stats.broadcast_acks_seen
             << ", \"acks_accounted\": " << g_last_drain_stats.broadcast_acks_accounted
             << "}, "
             << "\"settled_at_barrier\": {\"inserts\": " << g_last_drain_stats.settled_inserts
             << ", \"insert_responses\": " << g_last_drain_stats.settled_insert_responses
             << ", \"backlink_batches\": " << g_last_drain_stats.settled_backlink_batches
             << ", \"broadcast_acks\": " << g_last_drain_stats.settled_broadcast_acks
             << "}}, "
             << "\"drain_barrier_complete\": " << (ok ? "true" : "false") << "}";
        daemon_output_line(json.str());
        daemon_output_line("===JSON_END===");
    }
    else if (action == "ping") {
        // Network baseline measurement: ping [count] [--batch N]
        uint32_t count = 100;
        uint32_t ping_batch = 0;
        iss >> count;
        if (count == 0) count = 100;

        std::string token;
        while (iss >> token) {
            if (token == "--batch") iss >> ping_batch;
        }

        std::string json = run_ping_baseline(count, ping_batch);
        daemon_output_line("===JSON_RESULT===");
        daemon_output_line(json);
        daemon_output_line("===JSON_END===");
    }
    else if (action.empty()) {
        // Empty line, ignore
    }
    else {
        daemon_output_line("Unknown command: " + action);
        daemon_output_line(std::string("Available commands: ") + kAdminCommands);
    }

    return true;
}

} // namespace tide
