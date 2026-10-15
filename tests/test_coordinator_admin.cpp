// Coordinator admin commands (admin_commands.cpp, the pipeline loop and the
// search runner) against in-process fake servers: no sockets; one thread answers
// every server queue on the daemon's main response queue.
//   - insert, mixed, recreate-head and migration-apply are refused while the
//     controller is active; the controller's own fence (its in-memory plan) is not;
//     total_inserted is atomic (the server-reply thread reads it);
//   - controller-start drains pending insert maintenance first, takes no
//     row-bytes file and covers every id the coordinator knows;
//   - a search command ends with an error when the controller froze dispatch on
//     an ownership-ambiguous terminal and nothing is in flight;
//   - the drain barrier settles a dropped INSERT_RESPONSE (settled counters) and
//     counts the replies that have no pending entry (orphan counters);
//   - controller-stop's STOP exchange puts maintenance replies back; RESET drops them;
//   - reset reloads the load-time membership and drops the controller history,
//     and refuses a membership file with an owner label at or above K;
//   - a partial ACCEPT completes by a retried ACCEPT, and rollback is refused
//     once ACCEPT was sent;
//   - an ACCEPT attempt that timed out: its late ACK is discarded by attempt
//     id, never counted for the next attempt nor left in the response queue;
//   - a terminal controller ends by controller-abort without a restart: at once
//     when ownership is known, after a confirmed migration-accept when it was
//     ambiguous; reset and graph mutations run again afterwards;
//   - PREPARE is per target: every server gets the whole move list, each moved
//     node's vector goes only to its target (from REFRESH or a v3 file); an
//     own-only cluster refuses a plan without vectors before any fanout; the
//     fence payload is reported; the deadlines grow with the payload;
//   - membership-check compares every server's answer with the coordinator's
//     membership;
//   - a version-3 plan file whose vector is not the node's original (base.npy
//     row) is refused before any fanout; the per-server PREPARE bytes and the
//     vector bytes the serializer wrote into each fragment are reported;
//   - insert, mixed and recreate-head are refused while a migration is active
//     (between migration-apply and migration-accept / migration-rollback);
//   - a query a server refuses (status ERROR, a query of another dimension)
//     ends the search command with an error instead of a silent wait;
//     recall --search-l sets one command's search list size;
//   - an unreadable or malformed ground truth / query file is a refusal, not an
//     exit; the unknown-command reply lists every command; --debug files go to
//     the working directory;
//   - a mixed command with --debug completes searches and inserts against fake
//     servers that answer them: every owner label in the CSV log equals the
//     owner of its id, inserted ids included, while the inserts grow the
//     membership (the metric thread writes the log from the labels the pipeline
//     thread resolved; under ThreadSanitizer this is the race test of that
//     path).
// Each scenario runs in a child process: the daemon state is per process.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <type_traits>
#include <vector>
#include <sys/wait.h>

#include "coordinator_types.h"
#include "pipeline_types.h"
#include "membership_migration_coordinator.h"
#include "membership_migration_plan.h"
#include "npy_reader.h"
#include "npy_writer.h"
#include "protocol/broadcast_messages.h"
#include "protocol/insert_protocol.h"
#include "protocol/search_protocol.h"
#include "window_controller_runtime.h"
#include "window_snapshot_wire.h"

namespace tide {
DaemonState g_daemon;
std::ostream* g_current_out = nullptr;
}  // namespace tide

namespace {
using namespace tide;
using ann::MessageType;

void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", why); std::abort(); }
}

std::filesystem::path g_dir;
const std::vector<uint32_t> kOwners = {0, 0, 1, 1, 2, 2, 3, 3};

// Runs one admin command and returns everything it wrote for the client.
std::string run(const std::string& command) {
    std::ostringstream out;
    g_current_out = &out;
    process_daemon_command(command);
    g_current_out = nullptr;
    return out.str();
}
bool has(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

// The fake servers answer DRAIN, RESET, stats-window STOP and the migration
// exchanges, in order per rank like a real server's I/O thread. ACCEPT can be
// scripted to fail on some ranks for the first attempts, or to be answered
// late: the rank then answers nothing else until its late reply has gone out.
struct FakeServers {
    std::atomic<bool> running{true};
    std::thread thread;
    std::mutex mutex;
    std::array<int, 4> accept_failures{};  // failing ACCEPT attempts left per rank
    std::array<bool, 4> accepted{};
    std::array<double, 4> accept_stall_s{};  // the next ACCEPT on that rank is answered this late
    std::array<std::chrono::steady_clock::time_point, 4> stalled_until{};
    std::array<std::vector<char>, 4> held{};
    uint64_t epoch_from = 0, epoch_to = 1, move_count = 1;
    unsigned accept_requests = 0, rollback_requests = 0, search_requests = 0, unexpected = 0, late_replies = 0;
    // The PREPARE fragment each rank received, parsed (n_base 8 + inserted ids).
    std::array<MembershipMigrationPlan, 4> prepared{};
    std::array<size_t, 4> prepare_bytes{};
    // REFRESH: each source answers with its moved nodes' rows (empty) and, when
    // own_only, their vectors {gid, gid + 0.5}.
    bool own_only = false;
    unsigned refresh_requests = 0, membership_checks = 0;
    // membership-check: what each rank reports (status 0, consistent with the
    // coordinator's membership unless broken_rank names a rank).
    int broken_rank = -1;
    int corrupt_rank = -1;  // own-only: this rank's vector digest is off by one
    // SEARCH: when set, every query is refused with an ERROR reply (a server of
    // another dimension); search_l records the L of the last query received.
    bool refuse_search = false;
    uint32_t search_l = 0;
    // Workloads: when set, a query or an insert's search gets a NEED_HOP to the next rank on its first
    // visit and a DONE on the second (four base ids and the two latest inserted ids the fake has seen); an
    // INSERT_REQUEST gets an INSERT_RESPONSE (no cross-partition edges, so no backlink batch) and every
    // VECTOR_BROADCAST its ACK.
    bool answer_workloads = false;
    std::vector<uint32_t> assigned;  // inserted ids, in the order their INSERT_REQUEST arrived
    unsigned need_hops = 0, done_replies = 0, insert_requests = 0, broadcasts = 0;

    void reply(uint32_t server, std::vector<char> bytes) {
        ServerResponse response;
        response.server_idx = server;
        response.response_data = std::move(bytes);
        g_daemon.main_response_queue->push(std::move(response));
    }
    std::vector<char> ack_bytes(uint32_t server, MessageType type, uint8_t status, uint64_t count,
                                uint16_t request_id = 0) const {
        ann::MembershipMigrationAck ack{};
        ack.msg_type = type; ack.status = status; ack.server_rank = server; ack.request_id = request_id;
        ack.epoch_from = epoch_from; ack.epoch_to = epoch_to; ack.move_count = count;
        std::vector<char> bytes(sizeof(ack));
        std::memcpy(bytes.data(), &ack, sizeof(ack));
        return bytes;
    }
    void migration_ack(uint32_t server, MessageType type, uint8_t status, uint64_t count) {
        reply(server, ack_bytes(server, type, status, count));
    }
    void handle(uint32_t server, const std::vector<char>& request) {
        if (request.empty()) { ++unexpected; return; }
        std::lock_guard<std::mutex> lock(mutex);
        switch (static_cast<MessageType>(request[0])) {
        case MessageType::DRAIN_REQUEST: reply(server, {static_cast<char>(MessageType::DRAIN_ACK)}); break;
        case MessageType::RESET_REQUEST: reply(server, {static_cast<char>(MessageType::RESET_ACK)}); break;
        case MessageType::STATS_WINDOW_STOP_REQUEST: {
            WindowStopRequest stop{};
            std::memcpy(&stop, request.data(), std::min(request.size(), sizeof(stop)));
            WindowStopAck ack{};
            ack.status = 0; ack.active = 0; ack.rank = server;
            ack.requested_epoch = stop.epoch; ack.actual_epoch = stop.epoch;
            std::vector<char> bytes(sizeof(ack));
            std::memcpy(bytes.data(), &ack, sizeof(ack));
            reply(server, std::move(bytes));
            break;
        }
        case MessageType::MIGRATION_PREPARE_INLINE_REQUEST: {
            std::string error;
            prepare_bytes[server] = request.size();
            if (!load_membership_migration_plan_bytes(request.data() + 1, request.size() - 1, 64, 4,
                                                      prepared[server], error)) {
                ++unexpected;
                migration_ack(server, MessageType::MIGRATION_PREPARE_ACK, 1, move_count);
                break;
            }
            migration_ack(server, MessageType::MIGRATION_PREPARE_ACK, 0, prepared[server].moves.size());
            break;
        }
        case MessageType::MIGRATION_REFRESH_INLINE_REQUEST: {
            ++refresh_requests;
            MembershipMigrationPlan move_template, fragment;
            std::string error;
            if (!load_membership_migration_plan_bytes(request.data() + 1, request.size() - 1, 64, 4, move_template,
                                                      error)) { ++unexpected; break; }
            fragment.n_servers = 4; fragment.epoch_from = move_template.epoch_from;
            fragment.epoch_to = move_template.epoch_to;
            for (auto move : move_template.moves) {
                if (move.from_partition != server) continue;
                if (own_only) {
                    move.vector = {float(move.global_id), float(move.global_id) + 0.5f};
                    move.vector_digest = migration_vector_digest(move.vector);
                }
                fragment.moves.push_back(move);
            }
            std::vector<char> payload;
            if (!fragment.moves.empty() && !serialize_membership_migration_plan(fragment, payload, error)) {
                ++unexpected; break;
            }
            ann::MembershipMigrationRefreshHeader header{};
            header.msg_type = MessageType::MIGRATION_REFRESH_RESPONSE; header.status = 0;
            header.server_rank = server; header.epoch_from = move_template.epoch_from;
            header.epoch_to = move_template.epoch_to; header.move_count = fragment.moves.size();
            header.payload_bytes = payload.size();
            std::vector<char> bytes(sizeof(header) + payload.size());
            std::memcpy(bytes.data(), &header, sizeof(header));
            if (!payload.empty()) std::memcpy(bytes.data() + sizeof(header), payload.data(), payload.size());
            reply(server, std::move(bytes));
            break;
        }
        case MessageType::MEMBERSHIP_CHECK_REQUEST: {
            ++membership_checks;
            uint32_t ids = 0;
            std::memcpy(&ids, request.data() + 1, sizeof(ids));
            ann::MembershipCheckResponse r{};
            r.msg_type = MessageType::MEMBERSHIP_CHECK_RESPONSE; r.status = 0; r.server_rank = server;
            r.own_only = own_only ? 1 : 0; r.pool_ok = 1; r.ids = ids; r.server_ids = ids;
            r.first_mismatch = UINT32_MAX; r.ownership_epoch = epoch_from;  // no migration in that scenario
            for (uint32_t id = 0; id < ids; ++id) {
                uint32_t owner = 0;
                std::memcpy(&owner, request.data() + 1 + sizeof(ids) + 4 * id, sizeof(owner));
                r.owned += owner == server ? 1 : 0;
            }
            r.mapped = r.owned; r.vectors = own_only ? r.owned : 0;
            if (own_only) {
                // The digest of the originals of the owned ids (base.npy rows here).
                for (uint32_t id = 0; id < ids; ++id) {
                    uint32_t owner = 0;
                    std::memcpy(&owner, request.data() + 1 + sizeof(ids) + 4 * id, sizeof(owner));
                    if (owner == server)
                        r.vector_digest += node_vector_hash(id, g_daemon.base.data() + 2 * id, 2);
                }
                if (static_cast<int>(server) == corrupt_rank) r.vector_digest += 1;
            }
            if (static_cast<int>(server) == broken_rank) { r.membership_mismatches = 1; r.first_mismatch = 5; }
            std::vector<char> bytes(sizeof(r));
            std::memcpy(bytes.data(), &r, sizeof(r));
            reply(server, std::move(bytes));
            break;
        }
        case MessageType::MIGRATION_COMMIT_REQUEST:
            migration_ack(server, MessageType::MIGRATION_COMMIT_ACK, 0, move_count); break;
        case MessageType::MIGRATION_ACCEPT_REQUEST: {
            ++accept_requests;
            uint16_t id = 0;  // the coordinator's attempt id, echoed in the ACK
            if (request.size() != 1 + sizeof(id)) { ++unexpected; break; }
            std::memcpy(&id, request.data() + 1, sizeof(id));
            std::vector<char> bytes;
            if (accept_failures[server] > 0) {
                --accept_failures[server];
                bytes = ack_bytes(server, MessageType::MIGRATION_ACCEPT_ACK, 1, move_count, id);
            } else if (accepted[server]) {
                bytes = ack_bytes(server, MessageType::MIGRATION_ACCEPT_ACK, 0, 0, id);  // the idempotent repeat
            } else {
                accepted[server] = true;
                bytes = ack_bytes(server, MessageType::MIGRATION_ACCEPT_ACK, 0, move_count, id);
            }
            if (accept_stall_s[server] > 0) {
                held[server] = std::move(bytes);
                stalled_until[server] = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(accept_stall_s[server]));
                accept_stall_s[server] = 0;
            } else {
                reply(server, std::move(bytes));
            }
            break;
        }
        case MessageType::MIGRATION_ROLLBACK_REQUEST:
            ++rollback_requests;
            migration_ack(server, MessageType::MIGRATION_ROLLBACK_ACK, 0, move_count);
            break;
        case MessageType::SEARCH_REQUEST: {
            ++search_requests;
            ann::StateHeader header{};
            std::vector<float> query;
            std::vector<ann::BeamEntry> beam;
            std::vector<uint32_t> seen;
            std::vector<ann::RouteHop> route;
            if (!ann::deserialize_state(request.data() + 1, request.size() - 1, header, query, beam, seen, route)) {
                ++unexpected; break;
            }
            search_l = header.L;
            if (refuse_search) reply(server, ann::serialize_search_error(header.query_id, header.hop_count));
            else if (answer_workloads) answer_search(server, header);
            break;
        }
        case MessageType::INSERT_REQUEST: {
            ann::InsertHeader insert{};
            std::vector<float> vector, distances;
            std::vector<uint32_t> candidates;
            if (!answer_workloads ||
                !ann::deserialize_insert_request(request.data(), request.size(), insert, vector, candidates, distances) ||
                insert.target_partition != server) { ++unexpected; break; }
            ++insert_requests;
            assigned.push_back(insert.assigned_global_id);
            reply(server, ann::serialize_insert_response(insert.request_id, insert.assigned_global_id, true, {}, {}, 0));
            break;
        }
        case MessageType::VECTOR_BROADCAST: {
            ann::VectorBroadcastHeader broadcast{};
            if (!answer_workloads || request.size() < sizeof(broadcast)) { ++unexpected; break; }
            std::memcpy(&broadcast, request.data(), sizeof(broadcast));
            ++broadcasts;
            ann::VectorBroadcastAckHeader ack{};
            ack.msg_type = MessageType::VECTOR_BROADCAST_ACK; ack.node_id = broadcast.node_id;
            std::vector<char> bytes(sizeof(ack));
            std::memcpy(bytes.data(), &ack, sizeof(ack));
            reply(server, std::move(bytes));
            break;
        }
        default: ++unexpected; break;
        }
    }
    // Called under the mutex.
    void answer_search(uint32_t server, const ann::StateHeader& header) {
        std::vector<uint32_t> ids;
        for (uint32_t j = 0; j < 4; ++j) ids.push_back((header.query_id + j) % 8);
        for (size_t j = 0; j < 2 && j < assigned.size(); ++j) ids.push_back(assigned[assigned.size() - 1 - j]);
        std::vector<float> distances;
        std::vector<ann::BeamEntry> beam;
        for (size_t j = 0; j < ids.size(); ++j) {
            distances.push_back(0.25f * float(j + 1));
            beam.push_back({ids[j], distances.back(), 1, {0, 0, 0}});
        }
        if (header.hop_count == 0) {
            ++need_hops;
            reply(server, ann::serialize_response(header.query_id, ann::StateStatus::NEED_HOP, (server + 1) % 4, beam,
                                                  {}, {}, 4, 1));
        } else {
            ++done_replies;
            reply(server, ann::serialize_response(header.query_id, ann::StateStatus::DONE, UINT32_MAX, beam, ids,
                                                  distances, 4, header.hop_count + 1));
        }
    }
    // Read and written only by the fake's own thread.
    bool stalled(uint32_t server) const { return std::chrono::steady_clock::now() < stalled_until[server]; }
    void start() {
        thread = std::thread([this] {
            while (running.load()) {
                bool any = false;
                for (uint32_t server = 0; server < 4; ++server) {
                    if (stalled(server)) continue;
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (!held[server].empty()) {  // the late reply leaves before any later one
                            ++late_replies; any = true;
                            reply(server, std::move(held[server]));
                            held[server].clear();
                        }
                    }
                    ServerWorkItem item;
                    while (!stalled(server) && g_daemon.server_queues[server]->try_pop(item)) {
                        any = true; handle(server, item.request_data);
                    }
                }
                if (!any) std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });
    }
    void stop() { running = false; if (thread.joinable()) thread.join(); }
};

// Eight 2-d base vectors on four servers, four queries, server queues and the
// main response queue made in place of the coordinator's server threads.
void setup() {
    char pattern[] = "/tmp/coordinator-admin-XXXXXX";
    const char* made = mkdtemp(pattern);
    check(made != nullptr, "owned directory");
    g_dir = made;
    g_daemon.n_servers = 4;
    g_daemon.servers = {"fake-0", "fake-1", "fake-2", "fake-3"};
    g_daemon.dim = 2;
    g_daemon.n_base = 8;
    for (uint32_t v = 0; v < 8; ++v) { g_daemon.base.push_back(float(v)); g_daemon.base.push_back(float(v % 2)); }
    g_daemon.queries_data = {0.5f, 0.0f, 2.5f, 1.0f, 4.5f, 0.0f, 6.5f, 1.0f};
    g_daemon.n_queries = 4;
    g_daemon.membership = kOwners;
    g_daemon.data_dir = g_dir.string();
    g_daemon.membership_path = (g_dir / "membership.npy").string();
    ann::write_npy_uint32(g_daemon.membership_path, kOwners);
    ann::write_npy_uint32((g_dir / "rows.npy").string(), std::vector<uint32_t>(8, 12));
    g_daemon.head_index.rebuild(g_daemon.base.data(), 8, 2, g_daemon.membership, 4);
    g_daemon.main_response_queue = std::make_unique<ann::ThreadSafeQueue<ServerResponse>>();
    for (uint32_t server = 0; server < 4; ++server)
        g_daemon.server_queues.push_back(std::make_unique<ann::ThreadSafeQueue<ServerWorkItem>>());
}
// controller-start drains first, so a fake cluster must answer its DRAIN.
std::string controller_start() {
    return run("controller-start " + g_dir.string() + " window_sec=1000");
}
size_t queued_requests() {
    size_t queued = 0;
    for (const auto& queue : g_daemon.server_queues) queued += queue->size();
    return queued;
}
// A MOVE plan file: node 0 from `source` to `target`, epoch from -> from + 1.
// With `vectors`, a version-3 plan carrying {gid, gid + 0.5} per move (what
// the fake REFRESH sources ship); `vector_of` overrides it.
using VectorOf = std::vector<float> (*)(uint32_t);
std::vector<float> fake_source_vector(uint32_t gid) { return {float(gid), float(gid) + 0.5f}; }
std::vector<float> original_vector(uint32_t gid) { return {float(gid), float(gid % 2)}; }  // setup()'s base.npy row
std::string write_plan_moves(uint64_t from, const std::vector<std::array<uint32_t, 3>>& moves, bool vectors,
                             const std::string& name, VectorOf vector_of = fake_source_vector) {
    MembershipMigrationPlan plan;
    plan.n_servers = 4; plan.epoch_from = from; plan.epoch_to = from + 1;
    for (const auto& [gid, source, target] : moves) {
        MembershipMove move;
        move.global_id = gid; move.from_partition = source; move.to_partition = target;
        move.adjacency_digest = migration_row_digest(move.adjacency);
        if (vectors) {
            move.vector = vector_of(gid);
            move.vector_digest = migration_vector_digest(move.vector);
        }
        plan.moves.push_back(move);
    }
    std::vector<char> bytes;
    std::string error;
    check(serialize_membership_migration_plan(plan, bytes, error), "plan serializes");
    const std::string path = (g_dir / name).string();
    std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}
std::string write_plan(uint64_t from, uint32_t source = 0, uint32_t target = 3) {
    return write_plan_moves(from, {{0, source, target}}, false, "plan-" + std::to_string(from) + ".bin");
}
void push_reply(uint32_t server, std::vector<char> bytes) {
    ServerResponse response;
    response.server_idx = server;
    response.response_data = std::move(bytes);
    g_daemon.main_response_queue->push(std::move(response));
}
std::vector<char> broadcast_ack(uint32_t node) {
    std::vector<char> bytes(8, 0);
    bytes[0] = static_cast<char>(MessageType::VECTOR_BROADCAST_ACK);
    std::memcpy(bytes.data() + 4, &node, sizeof(node));
    return bytes;
}

// The long-lived server-reply thread reads it for the snapshot queue bound.
static_assert(std::is_same_v<decltype(DaemonState::total_inserted), std::atomic<uint32_t>>,
              "total_inserted is read off the command thread");

void refusals_under_controller() {
    setup();
    FakeServers fake; fake.start();
    const std::string start = controller_start();
    check(has(start, "\"status\":\"ok\"") && window_controller_active() && has(start, "\"drained\":true") &&
          has(start, "\"drain_seconds\":") && has(start, "\"nodes\":8") &&
          has(start, "\"drain_settled_at_barrier\":{\"inserts\":0,"), "controller starts after a drain");
    check(has(controller_start(), "controller already active"), "a second start is refused before any drain");
    for (const std::string command : {"insert 10", "mixed --queries 4 --insert-ratio 50", "recreate-head"}) {
        const std::string out = run(command);
        check(has(out, "requires a stopped controller (controller-stop; controller-abort ends a terminal one)") &&
              !has(out, "===JSON_RESULT==="),
              "a graph mutation is refused under an active controller");
    }
    // A manual MOVE is refused inside a controller phase before
    // any exchange (it would move the ownership epoch under the controller's
    // history); the controller's own fence, which PREPAREs the plan it holds
    // in memory, is not.
    const std::string plan = write_plan(0);
    for (const std::string& command : {"migration-apply " + plan, "migration-apply --refresh " + plan}) {
        const std::string out = run(command);
        check(has(out, "migration-apply requires a stopped controller") && !has(out, "===JSON_RESULT==="),
              "migration-apply is refused under an active controller");
    }
    std::ostringstream direct;
    g_current_out = &direct;
    const bool admin_prepare = coordinator_membership_migration_prepare(plan) ||
                               coordinator_membership_migration_prepare_current(plan);
    g_current_out = nullptr;
    check(!admin_prepare && has(direct.str(), "refused: a controller phase is active") &&
          fake.refresh_requests == 0 && queued_requests() == 0 && coordinator_membership_ownership_epoch() == 0,
          "the admin PREPARE entries refuse while a controller phase is active, before any exchange");
    std::ifstream plan_file(plan, std::ios::binary);
    const std::vector<char> plan_bytes((std::istreambuf_iterator<char>(plan_file)), std::istreambuf_iterator<char>());
    g_current_out = &direct;
    const bool fence_prepared = coordinator_membership_migration_prepare_controller(plan_bytes);
    const bool rolled_back = coordinator_membership_migration_rollback();
    g_current_out = nullptr;
    check(fence_prepared && rolled_back && fake.refresh_requests == 4 &&
          !coordinator_membership_migration_is_prepared() && coordinator_membership_ownership_epoch() == 0,
          "the controller's fence PREPAREs its in-memory plan while the controller is active");
    fake.stop();
    check(g_daemon.total_inserted == 0 && g_daemon.membership == kOwners && queued_requests() == 0 &&
          fake.search_requests == 0 && fake.unexpected == 0, "nothing was searched or inserted and nothing changed");
    check(has(run("controller-stop"), "\"status\":\"ok\"") && !window_controller_active(), "controller stops");
    check(has(run("recreate-head"), "\"action\": \"recreate-head\""), "after controller-stop the command runs");
}

// controller-start takes no row-bytes file (the row bytes come from the
// harvested owner histogram) and drains pending insert maintenance first: an
// insert whose response never arrived is settled at the drain barrier before
// the phase starts, and the window covers every id the coordinator knows.
void controller_start_phase_boundary() {
    setup();
    FakeServers fake; fake.start();
    const std::string with_rows = run("controller-start " + (g_dir / "rows.npy").string() + " " + g_dir.string());
    check(has(with_rows, "\"status\":\"error\"") && has(with_rows, "takes no row-bytes file") && !window_controller_active(),
          "a row-bytes file argument is refused with the reason");
    WorkloadItem lost{};
    lost.type = WorkloadType::INSERT; lost.id = 5; lost.insert_phase = InsertPhase::INSERT_PENDING;
    g_daemon.pending_workloads.push_back(lost);
    g_daemon.pending_insert_response_map[77] = 0;
    g_daemon.membership.push_back(1); g_daemon.total_inserted = 1;  // one inserted id (8), owned by rank 1
    const std::string start = controller_start();
    check(has(start, "\"status\":\"ok\"") && has(start, "\"drained\":true") &&
          has(start, "\"drain_settled_at_barrier\":{\"inserts\":1,\"insert_responses\":1,") &&
          has(start, "\"nodes\":9,\"n_base\":8,\"inserted\":1") && window_controller_active()->nodes == 9 &&
          g_daemon.pending_insert_response_map.empty(), "the drain settles the pending insert before the phase");
    // Its reply shows the drain's work (the pending insert response it found) and
    // the digest of the membership the window binds (little-endian uint32 owners).
    std::vector<uint8_t> owners;
    for (const uint32_t owner : g_daemon.membership)
        for (int b = 0; b < 4; ++b) owners.push_back(static_cast<uint8_t>(owner >> (8 * b)));
    check(has(start, "\"drain\":{\"pending_at_start\":{\"insert_responses\":1,\"backlink_batches\":0,"
                     "\"broadcasts\":0},\"responses\":") &&
          has(start, "\"membership_sha256\":\"" + sha256_hex(owners.data(), owners.size()) + "\""),
          "controller-start reports its drain's work and the bound membership's digest");
    check(has(run("controller-stop"), "\"status\":\"ok\""), "stop");
    fake.stop();
    check(fake.unexpected == 0, "no unexpected request");
}

void ambiguous_terminal_ends_search() {
    setup();
    FakeServers fake; fake.start();
    check(has(controller_start(), "\"status\":\"ok\""), "controller starts");
    fake.stop();  // the search below must find nothing dispatched
    // The state an ACCEPT failure leaves (the lifecycle test drives the real path).
    auto* r = window_controller_active();
    r->ownership_ambiguous = true; r->stopped_on_error = true; r->proposals_disabled = true;
    r->terminal_reason = "accept_failed"; r->state = WindowControllerState::RECOVERY_FAILED;
    alarm(60);  // the defect was a silent infinite loop
    const std::string out = run("recall --queries 4 --no-groundtruth");
    alarm(0);
    check(has(out, "Error: search command ended early") && has(out, "accept_failed") &&
          has(out, "migration-accept") && has(out, "controller-abort ends the terminal controller") &&
          has(out, "also needs reset (or a server restart)") &&
          has(out, "\"status\": \"error\"") && has(out, "\"completed_queries\": 0"),
          "the search command returns an error naming the terminal reason and the recovery");
    check(queued_requests() == 0, "no query was dispatched while ownership was ambiguous");
    check(has(run("status"), "\"status\": \"ok\""), "admin commands run afterwards");
    check(has(run("controller-status"), "\"ownership_ambiguous\":true"), "the controller stays terminal");
}

void drain_settles_and_counts_orphans() {
    setup();
    FakeServers fake; fake.start();
    // An insert whose INSERT_RESPONSE never arrives (a reply some consumer dropped).
    WorkloadItem lost{};
    lost.type = WorkloadType::INSERT; lost.id = 5; lost.insert_phase = InsertPhase::INSERT_PENDING;
    g_daemon.pending_workloads.push_back(lost);
    g_daemon.pending_insert_response_map[77] = 0;
    // Replies that have no pending entry.
    push_reply(1, ann::serialize_insert_response(999, 100, true, {}, {}, 0));
    push_reply(2, ann::serialize_batch_backlink_ack(4242, 1, 0, 0));
    push_reply(3, broadcast_ack(12345));
    const std::string out = run("drain");
    check(has(out, "\"settled_at_barrier\": {\"inserts\": 1, \"insert_responses\": 1, \"backlink_batches\": 0, "
                   "\"broadcast_acks\": 0}") && has(out, "\"drain_barrier_complete\": true"),
          "the barrier settled the insert whose response was dropped");
    check(has(out, "\"pending_at_start\": {\"insert_responses\":1,\"backlink_batches\":0,\"broadcasts\":0}"),
          "the drain reports what was pending when it began");
    const auto& m = g_daemon.maintenance;
    check(m.orphan_insert_responses == 1 && m.orphan_backlink_acks == 1 && m.orphan_broadcast_acks == 1,
          "every reply without a pending entry is counted as an orphan");
    check(has(run("status"), "\"orphan_responses\": {\"insert_responses\": 1, \"backlink_acks\": 1, "
                             "\"broadcast_acks\": 1}"), "status reports the orphans");
    check(has(run("drain"), "\"settled_at_barrier\": {\"inserts\": 0, \"insert_responses\": 0"),
          "a clean drain settles nothing");
    fake.stop();
    check(fake.unexpected == 0, "no unexpected request");
}

void maintenance_replies_across_stop_and_reset() {
    setup();
    FakeServers fake; fake.start();
    check(has(controller_start(), "\"status\":\"ok\""), "controller starts");
    window_controller_active()->stats_off_confirmed = false;  // a window was opened: stop sends STOP
    push_reply(0, ann::serialize_insert_response(321, 100, true, {}, {}, 0));
    push_reply(1, ann::serialize_batch_backlink_ack(654, 1, 0, 0));
    push_reply(2, broadcast_ack(4321));
    const std::string out = run("controller-stop");
    check(has(out, "\"status\":\"ok\",\"action\":\"controller-stop\"") && has(out, "\"stats_off_confirmed\":true") &&
          !window_controller_active(), "the STOP exchange completes despite the maintenance replies");
    std::vector<MessageType> types;
    ServerResponse response;
    while (g_daemon.main_response_queue->try_pop(response)) {
        types.push_back(ann::get_message_type(response.response_data.data(), response.response_data.size()));
        push_reply(response.server_idx, response.response_data);
        if (types.size() == 3) break;
    }
    check(types == std::vector<MessageType>({MessageType::INSERT_RESPONSE, MessageType::BATCH_BACKLINK_ACK,
                                             MessageType::VECTOR_BROADCAST_ACK}),
          "the STOP exchange put every maintenance reply back, in order");
    const std::string reset = run("reset");
    check(has(reset, "\"status\": \"ok\"") && has(reset, "Dropped 3 pre-reset server replies") &&
          g_daemon.main_response_queue->size() == 0, "RESET drops the pre-reset maintenance replies");
    fake.stop();
}

void reset_reloads_membership_and_history() {
    setup();
    FakeServers fake; fake.start();
    // The controller history binds at epoch 0 and records one committed MOVE.
    check(has(controller_start(), "\"status\":\"ok\""), "first start");
    window_controller_active()->action_history->expected_epoch = 1;
    check(has(run("controller-stop"), "\"status\":\"ok\""), "stop");
    // The committed MOVE itself: node 0 to rank 3 at epoch 0 -> 1.
    check(has(run("migration-apply " + write_plan(0)), "\"status\":\"ok\""), "migration applied");
    check(has(run("migration-accept"), "\"status\":\"ok\""), "migration accepted");
    check(g_daemon.membership[0] == 3, "the coordinator membership moved");
    check(has(controller_start(), "\"status\":\"ok\"") && has(run("controller-stop"), "\"status\":\"ok\""),
          "the history and the coordinator agree at epoch 1");
    // The load-time head index (here 8 samples; a rebuild takes 1 per partition) is restored by reset.
    g_daemon.head_index_at_load.rebuild(g_daemon.base.data(), 8, 2, kOwners, 4, 1.0f);
    check(g_daemon.head_index_at_load.n_samples() == 8 && g_daemon.head_index.n_samples() == 4,
          "a load-time head index distinct from the rebuilt one");
    const std::string reset = run("reset");
    check(has(reset, "\"status\": \"ok\"") && g_daemon.membership == kOwners,
          "reset reloads the load-time membership, moved owners included");
    check(g_daemon.head_index.n_samples() == 8, "reset restores the load-time head index, not a new sample");
    const std::string start = controller_start();
    check(has(start, "\"status\":\"ok\"") && has(start, "\"ownership_epoch\":0"),
          "after reset a controller starts with a fresh history at epoch 0");
    check(window_controller_active()->action_history->expected_epoch == 0, "fresh history");
    // A membership file with an owner label at or above K is refused, never folded.
    check(has(run("controller-stop"), "\"status\":\"ok\""), "stop before the corrupt reset");
    std::vector<uint32_t> corrupt = kOwners;
    corrupt[5] = 4;
    ann::write_npy_uint32(g_daemon.membership_path, corrupt);
    const std::string refused = run("reset");
    check(has(refused, "\"status\": \"error\"") &&
          has(refused, "node 5 has owner label 4, not below the server count 4") && g_daemon.membership == kOwners,
          "reset refuses a label at or above K and keeps the membership");
    ann::write_npy_uint32(g_daemon.membership_path, kOwners);
    fake.stop();
}

void partial_accept_completes_by_retry() {
    setup();
    FakeServers fake; fake.start();
    check(has(run("migration-apply " + write_plan(0)), "\"status\":\"ok\""), "migration applied");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.accept_failures = {0, 0, 1, 1}; }
    const std::string out = run("migration-accept");
    check(has(out, "\"status\":\"ok\"") && has(out, "ACCEPT incomplete; retrying") &&
          has(out, "\"accepted_epoch\":1,\"history_depth\":1"), "a retried ACCEPT completes the partial ACCEPT");
    check(fake.accept_requests == 8 && fake.rollback_requests == 0, "two ACCEPT fanouts, no rollback");

    // ACCEPT keeps failing on ranks 2 and 3: the plan can no longer roll back.
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.epoch_from = 1; fake.epoch_to = 2; }
    check(has(run("migration-apply " + write_plan(1, 3, 0)), "\"status\":\"ok\""), "second migration applied");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.accepted = {}; fake.accept_failures = {0, 0, 9, 9}; }
    check(has(run("migration-accept"), "\"status\":\"error\""), "ACCEPT fails after every attempt");
    const std::string rollback = run("migration-rollback");
    check(has(rollback, "\"status\":\"error\"") && has(rollback, "ACCEPT was already sent") &&
          fake.rollback_requests == 0, "rollback is refused once ACCEPT was sent; nothing is sent to the servers");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.accept_failures = {}; }
    check(has(run("migration-accept"), "\"accepted_epoch\":2,\"history_depth\":2"), "a later ACCEPT resolves it");
    fake.stop();
}

// ACCEPT attempt ids. Rank 3 answers attempt 1 only after it timed out; the
// late ACK reaches the coordinator during attempt 2, before rank 3's answer to
// attempt 2. Matched by attempt id, it is discarded: attempt 2 counts only its
// own ACKs and nothing is left on the response queue for a later exchange.
void accept_timeout_discards_late_acks() {
    setup();
    g_migration_exchange_deadline_sec = 0.3;
    FakeServers fake; fake.start();
    check(has(run("migration-apply " + write_plan(0)), "\"status\":\"ok\""), "migration applied");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.accept_stall_s = {0, 0, 0, 0.5}; }
    std::string out = run("migration-accept");
    check(has(out, "Timeout waiting for migration responses") && has(out, "ACCEPT incomplete; retrying"),
          "attempt 1 timed out on rank 3");
    check(has(out, "Discarded a late ACCEPT ACK of attempt 1 from server 3 (status 0, epoch 0 -> 1)") &&
          has(out, "\"status\":\"ok\"") && has(out, "\"accepted_epoch\":1,\"history_depth\":1"),
          "attempt 2 discards the late ACK of attempt 1 and completes on its own ACKs");
    check(!has(out, "Invalid migration response envelope") && g_daemon.main_response_queue->size() == 0,
          "no ACK is counted twice or left behind for another exchange");
    check(fake.accept_requests == 8 && fake.late_replies == 1, "two ACCEPT fanouts, one late reply");

    // Every attempt times out on rank 3; its three late ACKs arrive after the
    // command failed. The manual recovery still completes the ACCEPT.
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.epoch_from = 1; fake.epoch_to = 2; }
    check(has(run("migration-apply " + write_plan(1, 3, 0)), "\"status\":\"ok\""), "second migration applied");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.accepted = {}; fake.accept_stall_s = {0, 0, 0, 1.5}; }
    check(has(run("migration-accept"), "\"status\":\"error\""), "every ACCEPT attempt timed out on rank 3");
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (g_daemon.main_response_queue->size() < 3 && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(g_daemon.main_response_queue->size() == 3, "rank 3's three late ACKs arrive afterwards");
    out = run("migration-accept");
    check(has(out, "\"status\":\"ok\"") && has(out, "\"accepted_epoch\":2,\"history_depth\":2") &&
          g_daemon.main_response_queue->size() == 0, "the manual migration-accept recovers; nothing is left behind");
    fake.stop();
    check(fake.unexpected == 0, "every ACCEPT request carried an attempt id");
}

// controller-abort ends a terminal controller without a coordinator restart.
void terminal_controller_abort() {
    setup();
    FakeServers fake; fake.start();
    auto terminal = [](const char* reason, bool ambiguous) {
        auto* r = window_controller_active();
        r->stopped_on_error = true; r->proposals_disabled = true; r->ownership_ambiguous = ambiguous;
        r->terminal_reason = reason; r->state = WindowControllerState::RECOVERY_FAILED;
    };
    // Known ownership: a COMMIT failure whose rollback every server confirmed.
    check(has(controller_start(), "\"status\":\"ok\""), "controller starts");
    terminal("commit_failed", false);
    window_controller_active()->stats_off_confirmed = false;  // a window was open: abort sends the STOP
    check(has(run("insert 10"), "controller-abort ends a terminal one"), "an insert is refused, naming the abort");
    check(has(run("reset"), "controller-abort ends a terminal one"), "reset is refused, naming the abort");
    check(has(run("controller-stop"), "\"status\":\"error\"") && window_controller_active(),
          "controller-stop still refuses a terminal controller");
    std::string out = run("controller-abort");
    check(has(out, "\"status\":\"ok\",\"action\":\"controller-abort\"") &&
          has(out, "\"terminal_reason\":\"commit_failed\"") && has(out, "\"stats_off_confirmed\":true") &&
          !window_controller_active(), "controller-abort closes the stats windows and ends the controller");
    check(has(run("controller-abort"), "\"was_active\":false"), "nothing left to abort");
    check(has(run("recreate-head"), "\"action\": \"recreate-head\"") && has(run("reset"), "\"status\": \"ok\""),
          "graph mutations and reset run again without a restart");

    // Ambiguous ownership: the controller's fence committed (its own PREPARE of
    // the plan it holds in memory, then COMMIT), ACCEPT did not complete (an
    // admin migration-apply cannot stage this: it is refused inside a
    // controller phase).
    check(has(controller_start(), "\"status\":\"ok\""), "a new controller at epoch 0");
    check(has(run("migration-apply " + write_plan(0)), "migration-apply requires a stopped controller"),
          "a manual MOVE is refused inside the phase");
    {
        std::ifstream plan_file(write_plan(0), std::ios::binary);
        const std::vector<char> plan_bytes((std::istreambuf_iterator<char>(plan_file)),
                                           std::istreambuf_iterator<char>());
        std::ostringstream fence_out;
        g_current_out = &fence_out;
        const bool committed = coordinator_membership_migration_prepare_controller(plan_bytes) &&
                               coordinator_membership_migration_commit();
        g_current_out = nullptr;
        check(committed && coordinator_membership_ownership_epoch() == 1 && g_daemon.membership[0] == 3,
              "the controller's fence committed a migration");
    }
    terminal("accept_failed", true);
    out = run("controller-abort");
    check(has(out, "\"status\":\"error\"") && has(out, "ownership is ambiguous after accept_failed") &&
          window_controller_active(), "abort is refused while ownership is ambiguous");
    check(has(run("migration-accept"), "\"accepted_epoch\":1"), "the manual ACCEPT completes");
    const std::string status = run("controller-status");
    check(has(status, "\"ownership_ambiguous\":false") && has(status, "\"stopped_on_error\":true") &&
          has(status, "\"dispatch_frozen\":false"), "ownership confirmed; the controller stays terminal");
    check(has(run("controller-abort"), "\"status\":\"ok\"") && !window_controller_active(),
          "after the confirmed ACCEPT the terminal controller aborts");
    out = controller_start();
    check(has(out, "\"status\":\"error\"") && has(out, "requires ownership epoch 0"),
          "the dropped history cannot be rebuilt at epoch 1");
    check(has(run("reset"), "\"status\": \"ok\"") && has(controller_start(), "\"status\":\"ok\"") &&
          has(run("controller-stop"), "\"status\":\"ok\""), "after reset a controller starts again");
    fake.stop();
    check(fake.unexpected == 0, "no unexpected request");
}

// PREPARE is per target. An own-only cluster: node 0 (rank 0 -> 3) and node 2
// (rank 1 -> 2). With --refresh the sources ship the vectors; every server
// receives both moves (ids, owners, digests) and only the target's vector
// floats; the ACK is the global move count. A v3 file with vectors goes out the
// same way; a plan without vectors is refused before any fanout. The fence
// payload is reported, and the deadlines grow with it.
void per_target_prepare() {
    setup();
    g_daemon.own_only_storage = true;
    FakeServers fake; fake.own_only = true; fake.move_count = 2; fake.start();
    auto check_fragments = [&](const char* what, VectorOf vector_of) {
        std::lock_guard<std::mutex> lock(fake.mutex);
        for (uint32_t rank = 0; rank < 4; ++rank) {
            const auto& plan = fake.prepared[rank];
            check(plan.vector_target == rank && plan.moves.size() == 2, what);
            for (const auto& move : plan.moves) {
                check(move.vector_digest == migration_vector_digest(vector_of(move.global_id)),
                      "every server sees every move's vector digest");
                check(move.vector.empty() == (move.to_partition != rank), "the vector goes only to its target");
            }
        }
    };
    // PREPARE bytes per server: type byte, the v4 header (52 bytes), per move
    // 32 bytes (ids, owners, degree, row digest, vector digest) plus 4 per row
    // entry (the fake rows are empty) to every server, and 8 vector bytes (two
    // floats) per move only to its target: ranks 2 and 3 here.
    auto check_by_server = [&](const std::string& out) {
        const auto& p = coordinator_membership_migration_payload();
        std::lock_guard<std::mutex> lock(fake.mutex);
        const std::vector<uint64_t> vectors = {0, 0, 8, 8};
        check(p.prepare_vector_bytes_by_server == vectors && p.prepare_vector_bytes == 16,
              "the serializer wrote each vector into its target's fragment only");
        for (uint32_t rank = 0; rank < 4; ++rank)
            check(p.prepare_request_bytes_by_server[rank] == fake.prepare_bytes[rank] &&
                  p.prepare_request_bytes_by_server[rank] == 1 + 52 + 2 * 32 + vectors[rank],
                  "each server's PREPARE bytes are the move list plus its own vectors");
        check(has(out, "\"prepare_request_bytes_by_server\":[117,117,125,125]") &&
              has(out, "\"prepare_vector_bytes_by_server\":[0,0,8,8]"), "both lists are in the reply");
    };
    const std::string plan = write_plan_moves(0, {{0, 0, 3}, {2, 1, 2}}, false, "skeleton.bin");
    std::string out = run("migration-apply " + plan);
    check(has(out, "\"status\":\"error\"") && has(out, "carries none") && fake.refresh_requests == 0 &&
          queued_requests() == 0, "an own-only cluster refuses a plan without vectors before any fanout");
    out = run("migration-apply --refresh " + plan);
    check(has(out, "\"status\":\"ok\"") && has(out, "\"refresh\":true") && has(out, "\"move_count\":2") &&
          fake.refresh_requests == 4, "migration-apply --refresh: REFRESH, then the per-target PREPARE");
    check_fragments("--refresh: each rank's fragment names it and carries both moves", fake_source_vector);
    check_by_server(out);
    const auto& payload = coordinator_membership_migration_payload();
    check(payload.moves == 2 && payload.vector_bytes == 16 && payload.prepare_vector_bytes == 16 &&
          payload.refresh_response_bytes > 2 * 16 && payload.refresh_request_bytes > 0 &&
          payload.prepare_request_bytes == fake.prepare_bytes[0] + fake.prepare_bytes[1] + fake.prepare_bytes[2] +
          fake.prepare_bytes[3] && payload.total_bytes() == payload.refresh_request_bytes +
          payload.refresh_response_bytes + payload.prepare_request_bytes,
          "the fence payload counts every byte through the coordinator; each vector is sent once");
    check(fake.prepare_bytes[3] == fake.prepare_bytes[0] + 8 && fake.prepare_bytes[2] == fake.prepare_bytes[0] + 8,
          "a target's fragment is the bystanders' plus its one vector (2 floats)");
    check(has(out, "\"fence_payload_bytes\":" + std::to_string(payload.total_bytes())), "reported in the reply");
    check(payload.exchange_deadline_s > g_migration_exchange_deadline_sec &&
          payload.refresh_deadline_s > g_migration_refresh_deadline_sec, "the deadlines include the payload");
    check(has(run("migration-accept"), "\"status\":\"ok\"") && g_daemon.membership[0] == 3 &&
          g_daemon.membership[2] == 2, "accepted; the coordinator's membership moved");
    // A version-3 file that carries the vectors: each must be the node's
    // original (setup()'s base.npy row) byte for byte, or nothing is sent; with
    // the originals it goes out per target like a refreshed plan.
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.epoch_from = 1; fake.epoch_to = 2; fake.prepared = {}; }
    out = run("migration-apply " + write_plan_moves(1, {{0, 3, 0}, {2, 2, 1}}, true, "v3-wrong.bin"));
    check(has(out, "\"status\":\"error\"") && has(out, "vector of node 0 differs from its original vector") &&
          queued_requests() == 0 && fake.prepared[0].moves.empty() && g_daemon.membership[0] == 3,
          "a v3 file whose vector is not the original is refused before any fanout");
    out = run("migration-apply " + write_plan_moves(1, {{0, 3, 0}, {2, 2, 1}}, true, "v3.bin", original_vector));
    check(has(out, "\"status\":\"ok\"") && fake.refresh_requests == 4, "a v3 file needs no refresh");
    check_fragments("v3 file: per-target fragments", original_vector);
    check(has(run("migration-accept"), "\"status\":\"ok\"") && g_daemon.membership == kOwners, "moved back");
    // Deadlines: base + payload / rate.
    g_migration_payload_bytes_per_sec = 1024.0;
    check(migration_deadline_sec(30.0, 2048) == 32.0 && migration_deadline_sec(30.0, 0) == 30.0,
          "deadline = base + payload bytes / rate");
    fake.stop();
    check(fake.unexpected == 0, "every PREPARE fragment parsed");
}

// insert, mixed and recreate-head are refused between migration-apply and its
// verdict (a graph mutation would change rows the plan installed, and a
// rollback would lose them); they run again once the plan is resolved.
void migration_refuses_mutations() {
    setup();
    FakeServers fake; fake.start();
    check(has(run("migration-apply " + write_plan(0)), "\"status\":\"ok\""), "migration applied");
    for (const std::string command : {"insert 10", "mixed --queries 4 --insert-ratio 50", "recreate-head"}) {
        const std::string out = run(command);
        check(has(out, "requires no active migration (migration-accept or migration-rollback first)") &&
              has(out, "epoch 0 -> 1") && !has(out, "===JSON_RESULT==="),
              "a graph mutation is refused while a migration is active");
    }
    check(g_daemon.total_inserted == 0 && fake.search_requests == 0, "nothing was dispatched");
    check(has(run("migration-rollback"), "\"status\":\"ok\"") && g_daemon.membership == kOwners, "rolled back");
    check(has(run("recreate-head"), "\"action\": \"recreate-head\""), "after the verdict the command runs");
    fake.stop();
    check(fake.unexpected == 0, "no unexpected request");
}

// A server that refuses a query (status ERROR: a state of another dimension)
// answers it; the search command stops dispatching, waits for the queries in
// flight and ends with an error naming the refusal, instead of waiting for
// ever. --search-l sets one command's L (from K to 4096).
void refused_query_ends_search() {
    setup();
    FakeServers fake; fake.refuse_search = true; fake.start();
    alarm(60);  // the defect was a silent wait for a reply that never comes
    std::string out = run("recall --queries 4 --no-groundtruth --search-l 40");
    alarm(0);
    check(has(out, "Error: search command ended early") && has(out, "refused query") &&
          has(out, "status ERROR") && has(out, "\"status\": \"error\"") && has(out, "\"completed_queries\": 0"),
          "the search command returns an error naming the refused query");
    check(fake.search_requests >= 1 && fake.search_l == 40, "the queries carried the command's search list size");
    check(has(run("recall --queries 4 --no-groundtruth --search-l 5"), "--search-l needs a search list size"),
          "a search list below K is refused");
    check(has(run("status"), "\"status\": \"ok\""), "admin commands run afterwards");
    fake.stop();
    check(fake.unexpected == 0, "every query parsed");
}

// An unreadable or malformed ground truth or query file is a refusal of that one
// command (an Error: line, no JSON), never an exception that ends the coordinator;
// the unknown-command reply lists every command; --debug writes into the working
// directory, not the (possibly read-only) data directory.
void unreadable_inputs_are_refusals() {
    setup();
    FakeServers fake; fake.start();
    const std::string missing = (g_dir / "missing.npy").string();
    std::string out = run("recall --queries 4 --groundtruth " + missing);
    check(has(out, "Error: cannot read the ground truth") && has(out, missing) && !has(out, "===JSON_RESULT==="),
          "a missing --groundtruth is refused");
    out = run("baseline --queries 4");
    check(has(out, "Error: cannot read the ground truth") && has(out, "groundtruth.npy"),
          "a missing default <data-dir>/groundtruth.npy is refused");
    out = run("recall --queries 4 --no-groundtruth --query-file " + missing);
    check(has(out, "Error: cannot read the query file") && !has(out, "===JSON_RESULT==="),
          "a missing --query-file is refused");
    std::ofstream(g_dir / "bad.npy") << "not an npy file";
    out = run("recall --queries 4 --groundtruth " + (g_dir / "bad.npy").string());
    check(has(out, "Error: cannot read the ground truth"), "a malformed ground truth file is refused");
    check(has(run("status"), "\"status\": \"ok\""), "the coordinator keeps running");
    out = run("no-such-command");
    for (const char* name : {"recall", "drain", "membership-check", "migration-apply", "migration-accept",
                             "migration-rollback", "controller-start", "controller-status", "controller-stop",
                             "controller-abort"})
        check(has(out, "Unknown command: no-such-command") && has(out, name),
              "the unknown-command reply lists every command");
    const std::string debug_log = generate_debug_log_path("q", 4);
    check(debug_log.rfind("debug_", 0) == 0 && !has(debug_log, g_dir.string()),
          "the --debug logs go to the working directory");
    fake.stop();
    check(fake.search_requests == 0 && fake.unexpected == 0, "no query was sent");
}

// A raw NPY file (version 1.0) of the given dtype and shape holding `bytes` data bytes.
std::string write_raw_npy(const std::string& name, const std::string& descr, const std::string& shape,
                          const void* data, size_t bytes) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    const std::string path = (g_dir / name).string();
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    const uint16_t length = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&length), 2);
    out << header;
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return path;
}

// A ground truth must be a 2D int32/uint32/int64 array that covers every query the
// command scores (row i % <loaded queries>) with at least K ids per row; anything
// else is refused before a query is sent (it was read past its end), for recall and
// for mixed, and the coordinator keeps running.
void ground_truth_must_cover_the_queries() {
    setup();
    FakeServers fake; fake.start();
    const uint32_t K = g_daemon.K, queries = g_daemon.n_queries;  // 10 and 4
    const std::vector<float> floats(queries * K, 1.0f);
    const std::vector<int32_t> ids(queries * K, 1);
    const std::string shape = "(" + std::to_string(queries) + ", " + std::to_string(K) + ")";
    struct Case { std::string file, expected; };
    const std::vector<Case> cases = {
        {write_raw_npy("gt-f4.npy", "<f4", shape, floats.data(), floats.size() * 4), "is not int32, uint32 or int64"},
        {write_raw_npy("gt-short.npy", "<i4", shape, ids.data(), ids.size() * 4 - 8), "shorter than its shape"},
        {write_raw_npy("gt-1d.npy", "<i4", "(40,)", ids.data(), ids.size() * 4), "not a 2D"},
        {write_raw_npy("gt-rows.npy", "<i4", "(2, 10)", ids.data(), 2 * K * 4), "does not cover the 4 queries x K=10"},
        {write_raw_npy("gt-cols.npy", "<i4", "(4, 5)", ids.data(), 4 * 5 * 4), "does not cover the 4 queries x K=10"},
    };
    for (const auto& c : cases) {
        const std::string out = run("recall --queries 4 --groundtruth " + c.file);
        check(has(out, "Error: ") && has(out, c.expected) && !has(out, "===JSON_RESULT==="),
              "recall refuses a ground truth that is malformed or does not cover the queries");
    }
    // mixed: the same refusals before anything is dispatched (2 searches here, so 2 rows suffice).
    const std::vector<float> inserts = {0.25f, 0.5f, 1.25f, 0.5f};
    const std::string insert_file = write_raw_npy("inserts.npy", "<f4", "(2, 2)", inserts.data(), inserts.size() * 4);
    const std::string one_row = write_raw_npy("gt-one.npy", "<i4", "(1, 10)", ids.data(), K * 4);
    std::string out = run("mixed --queries 4 --insert-ratio 50 --insert-file " + insert_file + " --groundtruth " + one_row);
    check(has(out, "Error: ground truth 1 x 10 does not cover the 2 queries x K=10") && !has(out, "===JSON_RESULT==="),
          "mixed refuses a ground truth that does not cover its searches");
    out = run("mixed --queries 4 --insert-ratio 50 --insert-file " + insert_file + " --groundtruth " + cases[0].file);
    check(has(out, "Error: cannot read the ground truth") && has(out, "is not int32"),
          "mixed refuses an unreadable ground truth (it ran unscored before)");
    out = run("mixed --queries 4 --insert-ratio 50 --insert-file " + insert_file + " --groundtruth " +
              write_raw_npy("gt-two.npy", "<i4", "(2, 10)", ids.data(), 2 * K * 4) + " --insert-gt " + cases[4].file);
    check(has(out, "Error: insert ground truth 4 x 5 has fewer than K=10 ids per row"),
          "mixed refuses an insert ground truth with fewer than K ids per row");
    check(g_daemon.total_inserted == 0 && fake.search_requests == 0 && queued_requests() == 0, "nothing was dispatched");
    check(has(run("status"), "\"status\": \"ok\""), "the coordinator keeps running");
    fake.stop();
    check(fake.unexpected == 0, "no unexpected request");
}

// --debug CSV logs: the headers name one partition column per server, as the rows
// write them, at every K (the topK_p* and beam_p* columns).
void debug_csv_columns_follow_k() {
    setup();
    for (const uint32_t servers : {2u, 4u, 8u}) {
        const std::string path = (g_dir / ("debug-" + std::to_string(servers) + ".csv")).string();
        enable_partition_debug(servers, path, &g_daemon.membership);
        const std::vector<uint32_t> neighbors = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1};
        std::vector<uint32_t> labels(neighbors.size());
        for (size_t i = 0; i < labels.size(); ++i) labels[i] = uint32_t(i) % servers;
        const std::vector<float> distances(neighbors.size(), 0.5f);
        const std::vector<int32_t> gt(10, 1);
        log_search_result("QUERY", 7, 0, 1, 2, neighbors, labels, distances, 10, gt.data(), 10);
        const std::vector<ann::BeamEntry> beam = {{1, 0.5f, 0, {0, 0, 0}}, {2, 0.7f, 0, {0, 0, 0}}};
        log_hop_state("QUERY", 7, 1, 0, 1, beam, {1 % servers, 0});
        disable_partition_debug();
        const std::string hops_path = path.substr(0, path.rfind('.')) + "_hops.csv";
        for (const std::string& file : {path, hops_path}) {
            std::ifstream in(file);
            std::string header, row;
            check(std::getline(in, header) && std::getline(in, row), "a header and a row");
            check(std::count(header.begin(), header.end(), ',') == std::count(row.begin(), row.end(), ','),
                  "the header and the row have the same number of fields");
            const std::string column = file == path ? "topK_p" : "beam_p";
            check(has(header, column + std::to_string(servers - 1) + ",") &&
                  !has(header, column + std::to_string(servers) + ","), "one partition column per server");
        }
    }
}

// --debug on a mixed command against fake servers that complete its searches and inserts: the CSV log's
// owner labels equal the owners of their ids (load-time ids from setup(), inserted ids from the membership
// the inserts grew), while the inserts grow the membership and the metric thread writes the log.
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, ',')) fields.push_back(field);
    return fields;
}
std::vector<std::filesystem::path> debug_logs() {
    std::vector<std::filesystem::path> logs;
    for (const auto& entry : std::filesystem::directory_iterator(g_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("debug_", 0) == 0 && name.size() > 4 && name.substr(name.size() - 4) == ".csv")
            logs.push_back(entry.path());
    }
    std::sort(logs.begin(), logs.end());
    return logs;
}
void debug_labels_follow_membership_growth() {
    setup();
    check(chdir(g_dir.c_str()) == 0, "the working directory is the owned directory");
    g_daemon.K = 6;
    FakeServers fake; fake.answer_workloads = true; fake.start();
    // kPre inserts first (no --debug), so the fake names inserted ids in its results from the start of the mixed
    // command (with every workload queued at once, all searches can complete before the first INSERT_REQUEST
    // arrives, and then no row names an inserted id); the mixed command's own inserts then grow the
    // membership while the metric thread writes the log.
    constexpr uint32_t kPre = 8, kSearches = 48, kInserts = 48;
    std::vector<float> inserts;
    for (uint32_t i = 0; i < kPre + kInserts; ++i) { inserts.push_back(0.1f * float(i)); inserts.push_back(0.5f); }
    const std::string insert_file = write_raw_npy("debug-inserts.npy", "<f4",
                                                  "(" + std::to_string(kPre + kInserts) + ", 2)",
                                                  inserts.data(), inserts.size() * 4);
    // A ground truth (rolling recall on the pipeline thread) and the merge of pending inserts on (the metric
    // thread rewrites a search's results after the pipeline thread has queued it).
    std::vector<int32_t> gt;
    for (uint32_t q = 0; q < 4; ++q)
        for (uint32_t j = 0; j < 6; ++j) gt.push_back(int32_t((q + j) % 8));
    const std::string gt_file = write_raw_npy("debug-gt.npy", "<i4", "(4, 6)", gt.data(), gt.size() * 4);
    alarm(120);
    std::string out = run("insert " + std::to_string(kPre) + " --file " + insert_file);
    check(has(out, "===JSON_RESULT===") && !has(out, "Error:") && debug_logs().empty(), "the first inserts, no log");
    check(has(run("drain"), "{\"status\": \"ok\""), "drain settles the first inserts");
    out = run("mixed --queries " + std::to_string(kSearches + kInserts) + " --insert-ratio 50 --insert-file " +
              insert_file + " --insert-start-id " + std::to_string(kPre) + " --groundtruth " + gt_file + " --debug");
    check(has(out, "===JSON_RESULT===") && !has(out, "Error:") && !has(out, "--debug: cannot write"),
          "the mixed command completes and writes its --debug logs");
    const size_t merge = out.find("\"merge_search\": {");
    const size_t calls = out.find("\"calls\": ", merge);
    check(merge != std::string::npos && calls != std::string::npos &&
          std::stoul(out.substr(calls + 9)) > 0, "the metric thread merged pending inserts into results");
    check(has(run("drain"), "{\"status\": \"ok\""), "drain settles the inserts' maintenance");
    check(g_daemon.membership.size() == 8 + kPre + kInserts, "the inserts grew the membership");
    const auto logs = debug_logs();
    check(logs.size() == 2, "one result log and one hop log");
    const std::filesystem::path result_log = logs[0].string().find("_hops.csv") == std::string::npos ? logs[0] : logs[1];
    const std::filesystem::path hop_log = result_log == logs[0] ? logs[1] : logs[0];
    alarm(0);
    fake.stop();
    disable_partition_debug();  // closes (flushes) both logs

    auto owner = [](uint32_t id) { return id < 8 ? kOwners[id] : g_daemon.membership.at(id); };
    std::ifstream in(result_log);
    std::string line;
    check(std::getline(in, line) && line.rfind("type,id,entry_partition,final_partition,hops,", 0) == 0, "the header");
    unsigned queries = 0, inserts_logged = 0, inserted_ids = 0;
    while (std::getline(in, line)) {
        const auto fields = split_csv(line);
        check(fields.size() > 5 + 30, "a full row");
        if (fields[0] == "QUERY") ++queries;
        else if (fields[0] == "INSERT") ++inserts_logged;
        else check(false, "a QUERY or INSERT row");
        check(fields[4] == "2", "two visits: the NEED_HOP and the DONE");
        for (uint32_t i = 0; i < 10; ++i) {
            const std::string& id = fields[5 + 3 * i];
            if (id == "-1") continue;
            const uint32_t node = uint32_t(std::stoul(id));
            inserted_ids += node >= 8 ? 1 : 0;
            check(fields[6 + 3 * i] == std::to_string(owner(node)), "each id's owner label is its owner");
            if (i == 0) check(fields[3] == std::to_string(owner(node)), "the final partition is the first id's owner");
        }
    }
    check(queries == kSearches && inserts_logged == kInserts, "one row per search and per insert");
    check(inserted_ids > 0, "the rows name inserted ids");
    // The response path logs no hop rows: the hop log holds its header only.
    std::ifstream hops(hop_log);
    check(std::getline(hops, line) && line.rfind("type,id,hop,", 0) == 0 && !std::getline(hops, line),
          "the hop log has its header only");
    check(fake.unexpected == 0 && fake.insert_requests == kPre + kInserts && fake.broadcasts == 4 * (kPre + kInserts) &&
          fake.need_hops == kPre + kSearches + kInserts && fake.done_replies == kPre + kSearches + kInserts,
          "every query and insert was answered once per visit");
}

// membership-check: every server answers; the reply says whether all agree.
void membership_check_command() {
    setup();
    FakeServers fake; fake.start();
    std::string out = run("membership-check");
    check(has(out, "\"status\":\"ok\"") && has(out, "\"consistent\":true") && has(out, "\"ids\":8") &&
          has(out, "\"coordinator_owned_by_rank\":[2,2,2,2]") && fake.membership_checks == 4,
          "a consistent cluster");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.broken_rank = 2; }
    out = run("membership-check");
    check(has(out, "\"consistent\":false") && has(out, "\"rank\":2,\"consistent\":false") &&
          has(out, "\"first_mismatch\":5"), "one server's mismatch makes the cluster inconsistent");
    check(has(run("membership-check extra"), "takes no arguments"), "no arguments but --write");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.broken_rank = -1; }
    const std::string written = (g_dir / "written-membership.npy").string();
    out = run("membership-check --write " + written);
    uint32_t count = 0;
    check(has(out, "\"consistent\":true") && has(out, "\"written\":\"" + written + "\"") &&
          ann::read_npy_uint32(written, count) == kOwners && count == 8,
          "--write saves the coordinator's membership");
    // Own-only: each server's own vectors must equal the originals, byte for byte.
    g_daemon.own_only_storage = true;
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.own_only = true; }
    out = run("membership-check");
    check(has(out, "\"consistent\":true") && has(out, "\"vectors_exact\":true") &&
          has(out, "\"original_vectors\":true"), "own-only: the vector digests match the originals");
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.corrupt_rank = 1; }
    out = run("membership-check");
    check(has(out, "\"consistent\":false") && has(out, "\"rank\":1,\"consistent\":false") &&
          has(out, "\"vectors_exact\":false"), "a server holding a changed vector is inconsistent");
    fake.stop();
    check(fake.unexpected == 0, "no unexpected request");
}

// The coordinator's start-up cluster check (cluster_storage_consistent): one
// storage mode, the coordinator's dimension and one PQ code size on every server,
// each answering as its rank; a mixed cluster is refused.
void cluster_storage_check() {
    auto reply = [](uint32_t rank, uint32_t mode, uint32_t dim, uint32_t m) {
        ann::StatsResponseHeader r{};
        r.partition_id = rank; r.storage_mode = mode; r.storage_dim = dim; r.storage_pq_m = m;
        return r;
    };
    std::string summary, error;
    const std::vector<ann::StatsResponseHeader> pq = {reply(0, 1, 96, 24), reply(1, 1, 96, 24), reply(2, 1, 96, 24),
                                                      reply(3, 1, 96, 24)};
    check(cluster_storage_consistent(pq, 4, 96, &summary, &error) && summary == "pq, dim 96, PQ M 24" && error.empty(),
          "an own-only cluster passes");
    const std::vector<ann::StatsResponseHeader> full = {reply(0, 0, 96, 0), reply(1, 0, 96, 0)};
    check(cluster_storage_consistent(full, 2, 96, &summary, &error) && summary == "full, dim 96",
          "a full-replication cluster passes");
    auto mixed = pq; mixed[2] = reply(2, 0, 96, 0);
    check(!cluster_storage_consistent(mixed, 4, 96, &summary, &error) && has(error, "mixed storage modes") &&
          has(error, "2: full"), "a mixed cluster is refused, naming each server's mode");
    auto other_dim = pq; other_dim[1].storage_dim = 64;
    check(!cluster_storage_consistent(other_dim, 4, 96, &summary, &error) && has(error, "dimension"),
          "a server of another dimension (PQ pivots of another dataset) is refused");
    check(!cluster_storage_consistent(pq, 4, 128, &summary, &error) && has(error, "base.npy dimension 128"),
          "servers of another dimension than the coordinator's base.npy are refused");
    auto other_m = pq; other_m[3].storage_pq_m = 16;
    check(!cluster_storage_consistent(other_m, 4, 96, &summary, &error) && has(error, "PQ code sizes"),
          "PQ code sizes must agree");
    auto swapped = pq; swapped[1].partition_id = 2;
    check(!cluster_storage_consistent(swapped, 4, 96, &summary, &error) && has(error, "answered as partition"),
          "every server answers as its rank");
    check(!cluster_storage_consistent(pq, 8, 96, &summary, &error) && has(error, "expected 8"),
          "one reply per --servers entry");
    auto unknown = full; unknown[0].storage_mode = 7;
    check(!cluster_storage_consistent(unknown, 2, 96, &summary, &error) && has(error, "unknown storage mode"),
          "an unknown mode is refused");
}

void isolated(const char* name, void (*scenario)()) {
    std::fflush(stdout);
    const pid_t child = fork();
    check(child >= 0, "fork");
    if (child == 0) {
        scenario();
        std::error_code ec;
        std::filesystem::remove_all(g_dir, ec);
        std::fflush(stdout);
        _exit(0);
    }
    int status = 0;
    check(waitpid(child, &status, 0) == child, "wait");
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "scenario %s failed\n", name);
        std::abort();
    }
    std::printf("%s OK\n", name);
}
}  // namespace

int main() {
    isolated("refusals_under_controller", refusals_under_controller);
    isolated("ambiguous_terminal_ends_search", ambiguous_terminal_ends_search);
    isolated("drain_settles_and_counts_orphans", drain_settles_and_counts_orphans);
    isolated("controller_start_phase_boundary", controller_start_phase_boundary);
    isolated("maintenance_replies_across_stop_and_reset", maintenance_replies_across_stop_and_reset);
    isolated("reset_reloads_membership_and_history", reset_reloads_membership_and_history);
    isolated("partial_accept_completes_by_retry", partial_accept_completes_by_retry);
    isolated("accept_timeout_discards_late_acks", accept_timeout_discards_late_acks);
    isolated("terminal_controller_abort", terminal_controller_abort);
    isolated("cluster_storage_check", cluster_storage_check);
    isolated("per_target_prepare", per_target_prepare);
    isolated("membership_check_command", membership_check_command);
    isolated("migration_refuses_mutations", migration_refuses_mutations);
    isolated("refused_query_ends_search", refused_query_ends_search);
    isolated("unreadable_inputs_are_refusals", unreadable_inputs_are_refusals);
    isolated("ground_truth_must_cover_the_queries", ground_truth_must_cover_the_queries);
    isolated("debug_csv_columns_follow_k", debug_csv_columns_follow_k);
    isolated("debug_labels_follow_membership_growth", debug_labels_follow_membership_growth);
    std::printf("coordinator admin tests passed\n");
    return 0;
}
