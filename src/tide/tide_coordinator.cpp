/**
 * tide-ann coordinator (state-passing search).
 *
 * Coordinator responsibilities:
 * 1. Receive queries from client
 * 2. Initialize beam state and route to first server
 * 3. Route state between servers based on next_partition
 * 4. Collect final results
 *
 * Inter-query batching: --batch-size in-flight queries per server thread (default 8).
 *
 * Usage:
 *   tide-coordinator --data-dir /data/runtime \
 *       --servers tcp://server0:28800,tcp://server1:28800,... \
 *       --queries 1000 --batch-size 8
 */

#include <zmq.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <getopt.h>
#include <thread>
#include <memory>

#include "state_protocol.h"
#include "npy_reader.h"
#include "coordinator_utils.h"
#include "head_index.h"
#include "coordinator_types.h"
#include "pipeline_types.h"

using namespace ann;
using namespace std::chrono;

// =============================================================================
// Global variables
// =============================================================================
namespace tide {
DaemonState g_daemon;
std::ostream* g_current_out = nullptr;
// L2 is the only metric (cosine data is normalised offline).
constexpr const char* kMetricName = "l2";
}

using namespace tide;

// Alias for backward compatibility
inline std::vector<uint32_t> read_npy_int64(const std::string& path, uint32_t& n) {
    return read_npy_int64_as_uint32(path, n);
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --data-dir PATH     Data directory\n"
              << "  --servers LIST      Comma-separated server addresses\n"
              << "  --queries N         Number of queries (default: 200)\n"
              << "  --L N               Search list size (default: 50)\n"
              << "  --K N               Number of neighbors (default: 10)\n"
              << "  --W N               Search width: entries expanded per server visit (default: 8)\n"
              << "  --batch-size N      In-flight queries per thread (default: 8)\n"
              << "  --p2p               Enable P2P mode (servers forward states directly)\n"
              << "  --debug             Write per-hop debug CSVs and the recall-miss analysis\n"
              << "  --insert-R N        Insert: max out-degree R (default: 64)\n"
              << "  --insert-alpha F    Insert: alpha-RNG parameter (default: 1.2)\n"
              << "  --insert-L N        Insert: search list size (default: 64)\n"
              << "  --insert-W N        Insert: search width per server visit (default: 8)\n"
              << "  --cluster-check-sec N  Seconds to wait at start for every server's stats reply (default 300):\n"
              << "                      the coordinator refuses to start unless every server runs the same storage\n"
              << "                      mode (--pq or --full-vectors) with base.npy's dimension\n"
              << "  --print-wire-contract  Print the compiled metric and wire contract as JSON and exit\n"
              << "  --help              Show this help\n"
              << "Workloads, inserts, ground truth files and the controller are given per admin command\n"
              << "(recall, insert, mixed, controller-start, ...), not as process options.\n";
}

int main(int argc, char* argv[]) {
    std::string data_dir;
    std::string servers_str;
    uint32_t n_queries = 200;
    uint32_t L = 50;
    uint32_t K = 10;
    uint32_t W = 8;
    uint32_t batch_size = 8;
    bool p2p_mode = false;
    bool debug_mode = false;

    // Insert parameters
    uint32_t insert_R = 64;
    float insert_alpha = 1.2f;
    uint32_t insert_L = 64;
    uint32_t insert_W = 8;

    bool print_wire_contract = false;
    double cluster_check_sec = 300.0;

    static struct option long_options[] = {
        {"data-dir", required_argument, 0, 'd'},
        {"servers", required_argument, 0, 's'},
        {"queries", required_argument, 0, 'q'},
        {"L", required_argument, 0, 'l'},
        {"K", required_argument, 0, 'k'},
        {"W", required_argument, 0, 'w'},
        {"batch-size", required_argument, 0, 'b'},
        {"p2p", no_argument, 0, 'P'},
        {"debug", no_argument, 0, 'D'},
        {"insert-R", required_argument, 0, 'R'},
        {"insert-alpha", required_argument, 0, 'A'},
        {"insert-L", required_argument, 0, 1001},
        {"insert-W", required_argument, 0, 1002},
        {"print-wire-contract", no_argument, 0, 1020},
        {"cluster-check-sec", required_argument, 0, 1021},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:s:q:l:k:w:b:PDR:A:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'd': data_dir = optarg; break;
            case 's': servers_str = optarg; break;
            case 'q': n_queries = std::atoi(optarg); break;
            case 'l': L = std::atoi(optarg); break;
            case 'k': K = std::atoi(optarg); break;
            case 'w': W = std::atoi(optarg); break;
            case 'b': batch_size = std::atoi(optarg); break;
            case 'P': p2p_mode = true; break;
            case 'D': debug_mode = true; break;
            case 'R': insert_R = std::atoi(optarg); break;
            case 'A': insert_alpha = std::atof(optarg); break;
            case 1001: insert_L = std::atoi(optarg); break;
            case 1002: insert_W = std::atoi(optarg); break;
            case 1020: print_wire_contract = true; break;
            case 1021: cluster_check_sec = std::atof(optarg); break;
            case 'h':
            default:
                print_usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    if (print_wire_contract) {
        std::cout << "{\"metric\":\"" << kMetricName
                  << "\",\"wire_contract\":\""
                  << ann::state_wire_contract_token() << "\"}\n";
        return 0;
    }
    if (servers_str.empty()) {
        std::cerr << "Error: --servers required\n";
        print_usage(argv[0]);
        return 1;
    }
    if (data_dir.empty()) {
        std::cerr << "Error: --data-dir required\n";
        print_usage(argv[0]);
        return 1;
    }
    auto servers = parse_servers(servers_str);
    uint32_t n_servers = static_cast<uint32_t>(servers.size());

    // =========================================================================
    // DAEMON MODE: Load data, connect, and enter command loop
    // =========================================================================
    std::cout << "======================================================================\n";
    std::cout << "tide-ann coordinator (daemon mode)\n";
    std::cout << "======================================================================\n";
    std::cout << "Servers: " << n_servers << std::endl;
    for (size_t i = 0; i < servers.size(); i++) {
        std::cout << "  [" << i << "] " << servers[i] << std::endl;
    }
    std::cout << "Parameters: L=" << L << ", K=" << K << ", W=" << W
              << ", batch_size=" << batch_size << std::endl;

    std::cout << "Debug output: " << (debug_mode ? "on" : "off") << std::endl;

    // Store parameters in global daemon state
    g_daemon.servers = servers;
    g_daemon.n_servers = n_servers;
    g_daemon.data_dir = data_dir;
    g_daemon.L = L;
    g_daemon.K = K;
    g_daemon.W = W;
    g_daemon.batch_size = batch_size;
    g_daemon.insert_R = insert_R;
    g_daemon.insert_L = insert_L;
    g_daemon.insert_W = insert_W;
    g_daemon.insert_alpha = insert_alpha;
    g_daemon.p2p_mode = p2p_mode;
    g_daemon.debug_mode = debug_mode;

    // Load data into daemon state
    std::cout << "\nLoading data..." << std::endl;

    uint32_t n_base, dim;
    g_daemon.base = read_npy_float(data_dir + "/base.npy", n_base, dim);
    g_daemon.n_base = n_base;
    g_daemon.dim = dim;
    std::cout << "  Base: " << n_base << " x " << dim << std::endl;

    uint32_t n_q, q_dim;
    g_daemon.queries_data = read_npy_float(data_dir + "/query.npy", n_q, q_dim);
    if (q_dim != dim) {
        std::cerr << "Error: " << data_dir << "/query.npy has dimension " << q_dim << ", base.npy " << dim << "\n";
        return 1;
    }
    g_daemon.n_queries = std::min(n_queries, n_q);
    std::cout << "  Queries available: " << g_daemon.n_queries << std::endl;

    uint32_t n_membership;
    std::string membership_path = data_dir + "/membership.npy";
    g_daemon.membership = read_npy_int64(membership_path, n_membership);
    g_daemon.membership_path = membership_path;
    std::cout << "  Membership: " << n_membership << " (from " << membership_path << ")" << std::endl;
    {
        // Every owner label must name one of the --servers (K = their count): a
        // label at or above K is never folded onto another server.
        std::string label_error;
        if (!owner_labels_valid(g_daemon.membership, n_servers, &label_error)) {
            std::cerr << "Error: " << membership_path << ": " << label_error << " (--servers lists " << n_servers
                      << ")\n";
            return 1;
        }
    }

    // Load Head Index
    g_daemon.head_index.load(data_dir, g_daemon.base.data(), n_base, dim);
    g_daemon.head_index_at_load = g_daemon.head_index;
    std::cout << "  Head Index: " << g_daemon.head_index.n_samples() << " samples loaded" << std::endl;

    // Connect to servers using DEALER sockets
    std::cout << "\nConnecting to servers (DEALER sockets)..." << std::endl;
    g_daemon.context = std::make_unique<zmq::context_t>(1);
    std::string coordinator_identity = "coord";

    for (size_t i = 0; i < servers.size(); i++) {
        g_daemon.sockets.emplace_back(*g_daemon.context, zmq::socket_type::dealer);
        g_daemon.sockets.back().set(zmq::sockopt::linger, 0);
        g_daemon.sockets.back().set(zmq::sockopt::rcvtimeo, 0);
        g_daemon.sockets.back().set(zmq::sockopt::sndtimeo, 1000);
        g_daemon.sockets.back().set(zmq::sockopt::routing_id, coordinator_identity);
        g_daemon.sockets.back().set(zmq::sockopt::sndhwm, 10000);
        g_daemon.sockets.back().set(zmq::sockopt::rcvhwm, 10000);
        g_daemon.sockets.back().connect(servers[i]);
        std::cout << "  Connected to " << servers[i] << " (DEALER)" << std::endl;
    }

    // Setup poll items
    for (auto& sock : g_daemon.sockets) {
        zmq::pollitem_t item = {static_cast<void*>(sock), 0, ZMQ_POLLIN, 0};
        g_daemon.poll_items.push_back(item);
    }

    // Cluster check: every server's stats reply (servers may still be loading;
    // each round waits 5 s), then one storage mode and base.npy's dimension on
    // every server, or the coordinator does not start.
    {
        if (!(cluster_check_sec > 0.0)) {
            std::cerr << "Error: --cluster-check-sec must be positive\n";
            return 1;
        }
        std::cout << "\nChecking the servers' storage mode and dimension..." << std::endl;
        const auto check_deadline = steady_clock::now() + duration<double>(cluster_check_sec);
        std::vector<ann::StatsResponseHeader> replies;
        bool complete = false;
        while (!complete && steady_clock::now() < check_deadline) {
            complete = query_server_stats(replies, false);
            reset_socket_state();
        }
        if (!complete) {
            std::cerr << "Error: not every server answered the start-up stats request within " << cluster_check_sec
                      << " s\n";
            return 1;
        }
        std::string summary, error;
        if (!cluster_storage_consistent(replies, n_servers, dim, &summary, &error)) {
            std::cerr << "Error: refusing the cluster: " << error << "\n";
            return 1;
        }
        g_daemon.own_only_storage = replies.front().storage_mode == 1;
        std::cout << "  Cluster: " << n_servers << " servers, storage " << summary << std::endl;
    }

    // Create REP socket for client API
    g_daemon.client_socket = std::make_unique<zmq::socket_t>(
        *g_daemon.context, zmq::socket_type::rep
    );
    g_daemon.client_socket->set(zmq::sockopt::linger, 0);
    g_daemon.client_socket->bind(g_daemon.client_endpoint);
    std::cout << "\nClient API listening on " << g_daemon.client_endpoint << std::endl;

    // =========================================================================
    // DAEMON LOOP
    // =========================================================================
    std::cout << "\n======================================================================\n";
    std::cout << "Entering daemon mode. Waiting for ZMQ requests on " << g_daemon.client_endpoint << "\n";
    std::cout << "Commands: " << kAdminCommands << "\n";
    std::cout << "======================================================================\n";
    std::cout.flush();

    bool should_exit = false;
    while (!should_exit) {
        zmq::message_t request;
        try {
            auto result = g_daemon.client_socket->recv(request, zmq::recv_flags::none);
            if (!result || request.size() == 0) {
                continue;
            }
        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                break;
            }
            std::cerr << "ZMQ recv error: " << e.what() << std::endl;
            continue;
        }

        std::string cmd(static_cast<char*>(request.data()), request.size());

        // Trim whitespace
        size_t start = cmd.find_first_not_of(" \t\r\n");
        size_t end = cmd.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) {
            g_daemon.client_socket->send(zmq::buffer(""), zmq::send_flags::none);
            continue;
        }
        cmd = cmd.substr(start, end - start + 1);

        std::cout << "\n>>> Command: " << cmd << "\n";
        std::cout.flush();

        // Capture output
        std::ostringstream response_stream;
        g_current_out = &response_stream;

        bool continue_running = process_daemon_command(cmd);

        g_current_out = nullptr;

        // Send response
        std::string response = response_stream.str();
        try {
            g_daemon.client_socket->send(zmq::buffer(response), zmq::send_flags::none);
        } catch (const zmq::error_t& e) {
            std::cerr << "ZMQ send error: " << e.what() << std::endl;
        }

        if (!continue_running) {
            should_exit = true;
        }

        reset_socket_state();
    }

    std::cout << "\nCoordinator daemon exiting.\n";
    stop_daemon_server_threads();
    return 0;
}
