#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "membership_migration.h"

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__   \
                      << ": " #condition << std::endl;                       \
            return 1;                                                         \
        }                                                                     \
    } while (0)

namespace {

constexpr uint32_t NODES = 8;
using Cluster = std::array<tide::ServerData, 4>;

std::vector<std::vector<uint32_t>> graph_rows() {
    return {{1, 2}, {0, 3}, {0, 4}, {1, 5},
            {2, 6}, {3, 7}, {4, 7}, {5, 6}};
}

std::array<uint32_t, NODES> membership() {
    return {0, 0, 1, 1, 2, 2, 3, 3};
}

void initialize(Cluster& cluster) {
    const auto owners = membership();
    const auto rows = graph_rows();
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        auto& data = cluster[rank];
        data.my_partition = rank;
        data.n_servers = 4;
        data.dim = 2;
        data.base_static_capacity = NODES;
        tide::n_base_set(data, NODES);
        data.membership.assign(owners.begin(), owners.end());
        data.global_to_local.ensure_capacity_for(NODES - 1, UINT32_MAX);
        uint32_t local = 0;
        for (uint32_t gid = 0; gid < NODES; ++gid) {
            if (owners[gid] != rank) continue;
            data.adj.ensure_capacity_for(local);
            data.local_to_global.ensure_capacity_for(local, UINT32_MAX);
            data.global_to_local.at(gid) = local;
            data.local_to_global.at(local) = gid;
            data.adj.at(local) = rows[gid];
            ++local;
        }
        tide::n_local_set(data, local);
        tide::ensure_node_locks(data, local);
    }
}

bool active_graph_digest(const Cluster& cluster, uint64_t& output) {
    uint64_t digest = 1469598103934665603ULL;
    for (uint32_t gid = 0; gid < NODES; ++gid) {
        const uint32_t owner = tide::mem_at(cluster[0], gid);
        for (const auto& server : cluster) {
            if (tide::mem_at(server, gid) != owner) return false;
        }
        const auto& server = cluster[owner];
        const uint32_t local = server.global_to_local.at(gid);
        if (local == UINT32_MAX ||
            server.local_to_global.at(local) != gid) {
            return false;
        }
        digest ^= gid;
        digest *= 1099511628211ULL;
        digest ^= tide::migration_row_digest(server.adj.at(local));
        digest *= 1099511628211ULL;
    }
    output = digest;
    return true;
}

std::vector<tide::MembershipMove> moves() {
    const auto rows = graph_rows();
    std::vector<tide::MembershipMove> result;
    for (auto [gid, from, to] :
         {std::array<uint32_t, 3>{1, 0, 2},
          std::array<uint32_t, 3>{4, 2, 3}}) {
        tide::MembershipMove move{};
        move.global_id = gid;
        move.from_partition = from;
        move.to_partition = to;
        move.adjacency = rows[gid];
        move.adjacency_digest =
            tide::migration_row_digest(move.adjacency);
        result.push_back(std::move(move));
    }
    return result;
}

void write_u32_le(std::ostream& output, uint32_t value) {
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xff));
    }
}

void write_u64_le(std::ostream& output, uint64_t value) {
    for (uint32_t shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xff));
    }
}

void write_plan(const std::string& path,
                const std::vector<tide::MembershipMove>& batch,
                bool trailing_byte = false) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write("TIDEMIG\0", 8);
    write_u32_le(output, 1);
    write_u32_le(output, 4);
    write_u64_le(output, 7);
    write_u64_le(output, 8);
    write_u64_le(output, batch.size());
    for (const auto& move : batch) {
        write_u32_le(output, move.global_id);
        write_u32_le(output, move.from_partition);
        write_u32_le(output, move.to_partition);
        write_u32_le(output, move.adjacency.size());
        write_u64_le(output, move.adjacency_digest);
        for (uint32_t neighbor : move.adjacency) {
            write_u32_le(output, neighbor);
        }
    }
    if (trailing_byte) output.put('x');
}

int test_plan_round_trip_and_trailing_rejection() {
    const std::string path = "/tmp/test_membership_migration_plan.bin";
    const auto batch = moves();
    write_plan(path, batch);
    tide::MembershipMigrationPlan plan;
    std::string error;
    CHECK(tide::load_membership_migration_plan(
        path, NODES, 4, plan, error));
    CHECK(plan.epoch_from == 7);
    CHECK(plan.epoch_to == 8);
    CHECK(plan.moves.size() == batch.size());
    CHECK(plan.moves[0].adjacency == batch[0].adjacency);
    std::vector<char> serialized;
    CHECK(tide::serialize_membership_migration_plan(
        plan, serialized, error));
    tide::MembershipMigrationPlan reparsed;
    CHECK(tide::load_membership_migration_plan_bytes(
        serialized.data(), serialized.size(), NODES, 4,
        reparsed, error));
    CHECK(reparsed.moves.size() == plan.moves.size());
    CHECK(reparsed.moves[1].adjacency == plan.moves[1].adjacency);

    write_plan(path, batch, true);
    CHECK(!tide::load_membership_migration_plan(
        path, NODES, 4, plan, error));
    CHECK(error.find("trailing bytes") != std::string::npos);
    std::remove(path.c_str());
    return 0;
}

// K = 16: a MOVE plan over sixteen servers (sources and targets up to 15)
// serializes and loads back exactly; a loader expecting another server count,
// or a partition id beyond the plan's servers, refuses it.
int test_plan_round_trip_k16() {
    constexpr uint32_t kServers = 16, kNodes = 64;
    tide::MembershipMigrationPlan plan;
    plan.n_servers = kServers; plan.epoch_from = 11; plan.epoch_to = 12;
    const uint32_t pairs[][2] = {{9, 13}, {15, 0}, {0, 15}, {4, 12}, {13, 9}};
    for (uint32_t i = 0; i < 5; ++i) {
        tide::MembershipMove move;
        move.global_id = 7 * i + 3;
        move.from_partition = pairs[i][0];
        move.to_partition = pairs[i][1];
        move.adjacency = {i, 63 - i, 32 + i};
        move.adjacency_digest = tide::migration_row_digest(move.adjacency);
        plan.moves.push_back(move);
    }
    std::vector<char> bytes;
    std::string error;
    CHECK(tide::serialize_membership_migration_plan(plan, bytes, error));
    tide::MembershipMigrationPlan parsed;
    CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), kNodes, kServers, parsed, error));
    CHECK(parsed.n_servers == kServers && parsed.epoch_from == 11 && parsed.epoch_to == 12 &&
          parsed.mode == tide::kMigrationPlanModeMove && parsed.moves.size() == plan.moves.size());
    for (size_t i = 0; i < plan.moves.size(); ++i) {
        CHECK(parsed.moves[i].global_id == plan.moves[i].global_id &&
              parsed.moves[i].from_partition == plan.moves[i].from_partition &&
              parsed.moves[i].to_partition == plan.moves[i].to_partition &&
              parsed.moves[i].adjacency == plan.moves[i].adjacency &&
              parsed.moves[i].adjacency_digest == plan.moves[i].adjacency_digest);
    }
    std::vector<char> again;
    CHECK(tide::serialize_membership_migration_plan(parsed, again, error) && again == bytes);
    CHECK(!tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), kNodes, 4, parsed, error));
    auto beyond = plan; beyond.moves[0].to_partition = kServers;
    std::vector<char> beyond_bytes;
    CHECK(!tide::serialize_membership_migration_plan(beyond, beyond_bytes, error) ||
          !tide::load_membership_migration_plan_bytes(beyond_bytes.data(), beyond_bytes.size(), kNodes, kServers,
                                                        parsed, error));
    return 0;
}

int test_plan_mode_refuses_copy() {
    // A version-1 file (write_plan) parses as a MOVE plan; a version-2 plan
    // whose mode word says 1 (not a MOVE) is refused.
    const std::string path = "/tmp/test_membership_migration_mode.bin";
    write_plan(path, moves());
    tide::MembershipMigrationPlan v1;
    std::string error;
    CHECK(tide::load_membership_migration_plan(path, NODES, 4, v1, error));
    CHECK(v1.mode == tide::kMigrationPlanModeMove);
    std::vector<char> bytes;
    CHECK(tide::serialize_membership_migration_plan(v1, bytes, error));
    tide::MembershipMigrationPlan parsed;
    CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, parsed, error) &&
          parsed.mode == tide::kMigrationPlanModeMove && parsed.moves.size() == v1.moves.size());
    bytes[12] = 1;  // "TIDEMIG\0", version 2, then the mode word
    CHECK(!tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, parsed, error));
    CHECK(error.find("mode is invalid") != std::string::npos);
    std::remove(path.c_str());
    return 0;
}

int test_prepare_commit_and_rollback() {
    Cluster cluster;
    initialize(cluster);
    const auto batch = moves();
    uint64_t before_digest = 0;
    CHECK(active_graph_digest(cluster, before_digest));
    std::array<uint32_t, 4> before_n_local{};
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        before_n_local[rank] = tide::n_local_load(cluster[rank]);
    }

    std::array<tide::MembershipMigrationJournal, 4> journals;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::prepare_membership_migration(
            cluster[rank], batch, 7, 8, journals[rank], error));
    }

    // Prepare installed target mappings, but membership and source authority
    // are still old everywhere.
    for (const auto& server : cluster) {
        CHECK(tide::mem_at(server, 1) == 0);
        CHECK(tide::mem_at(server, 4) == 2);
    }
    CHECK(cluster[2].global_to_local.at(1) != UINT32_MAX);
    CHECK(cluster[0].global_to_local.at(1) != UINT32_MAX);
    CHECK(cluster[3].global_to_local.at(4) != UINT32_MAX);
    CHECK(cluster[2].global_to_local.at(4) != UINT32_MAX);

    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::commit_membership_migration(
            cluster[rank], batch, journals[rank], error));
    }
    for (const auto& server : cluster) {
        CHECK(tide::mem_at(server, 1) == 2);
        CHECK(tide::mem_at(server, 4) == 3);
    }
    CHECK(cluster[0].global_to_local.at(1) == UINT32_MAX);
    CHECK(cluster[2].global_to_local.at(4) == UINT32_MAX);
    uint64_t committed_digest = 0;
    CHECK(active_graph_digest(cluster, committed_digest));
    CHECK(committed_digest == before_digest);

    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::rollback_membership_migration(
            cluster[rank], batch, journals[rank], error));
        CHECK(tide::n_local_load(cluster[rank]) == before_n_local[rank]);
    }
    uint64_t rolled_back_digest = 0;
    CHECK(active_graph_digest(cluster, rolled_back_digest));
    CHECK(rolled_back_digest == before_digest);
    return 0;
}

int test_accept_revokes_rollback() {
    // ACCEPT is final: an accepted journal, and every copy of it, refuses
    // ROLLBACK and a second ACCEPT, and nothing changes. The validating ACCEPT
    // (accept_membership_migration) frees nothing; the server's ACCEPT
    // (accept_membership_migration_history) frees the source rows.
    Cluster cluster;
    initialize(cluster);
    const auto batch = moves();
    std::array<tide::MembershipMigrationJournal, 4> journals, copies;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::prepare_membership_migration(
            cluster[rank], batch, 7, 8, journals[rank], error));
        CHECK(tide::commit_membership_migration(
            cluster[rank], batch, journals[rank], error));
        copies[rank] = journals[rank];
        CHECK(tide::accept_membership_migration(
            cluster[rank], batch, journals[rank], error));
        CHECK(journals[rank].lifetime->accepted && journals[rank].lifetime->rollback_revoked &&
              copies[rank].lifetime->rollback_revoked);
    }
    uint64_t accepted_digest = 0;
    CHECK(active_graph_digest(cluster, accepted_digest));
    const auto n_local_target = tide::n_local_load(cluster[2]);
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(!tide::rollback_membership_migration(cluster[rank], batch, journals[rank], error));
        CHECK(error.find("revoked") != std::string::npos);
        CHECK(!tide::rollback_membership_migration(cluster[rank], batch, copies[rank], error));
        CHECK(!tide::accept_membership_migration(cluster[rank], batch, copies[rank], error));
    }
    for (const auto& server : cluster) {
        CHECK(tide::mem_at(server, 1) == 2);
        CHECK(tide::mem_at(server, 4) == 3);
    }
    uint64_t digest_after = 0;
    CHECK(active_graph_digest(cluster, digest_after));
    CHECK(digest_after == accepted_digest && tide::n_local_load(cluster[2]) == n_local_target);
    CHECK(cluster[0].global_to_local.at(1) == UINT32_MAX && !cluster[0].adj.at(1).empty());
    return 0;
}

int test_accept_history_keeps_epochs_only() {
    // The server's ACCEPT drops the plan (with its rows) and the journal and
    // records only the accepted epoch; a rejected ACCEPT changes nothing.
    Cluster cluster;
    initialize(cluster);
    std::array<tide::MembershipMigrationJournal, 4> journals;
    std::array<tide::MembershipMigrationPlan, 4> plans;
    std::array<std::vector<uint64_t>, 4> epochs;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        plans[rank].n_servers = 4; plans[rank].epoch_from = 7; plans[rank].epoch_to = 8; plans[rank].moves = moves();
        CHECK(tide::prepare_membership_migration(
            cluster[rank], plans[rank].moves, 7, 8, journals[rank], error));
    }
    std::string error;
    CHECK(!tide::accept_membership_migration_history(cluster[0], plans[0], journals[0], epochs[0], error));
    CHECK(epochs[0].empty() && !plans[0].moves.empty() && journals[0].prepared && !journals[0].lifetime->rollback_revoked);
    for (uint32_t rank = 0; rank < cluster.size(); ++rank)
        CHECK(tide::commit_membership_migration(cluster[rank], plans[rank].moves, journals[rank], error));
    const auto retained = journals[2];
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        CHECK(tide::accept_membership_migration_history(cluster[rank], plans[rank], journals[rank], epochs[rank], error));
        CHECK(epochs[rank] == std::vector<uint64_t>{8});
        CHECK(plans[rank].moves.empty() && journals[rank].entries.empty() && !journals[rank].lifetime &&
              !journals[rank].prepared);
    }
    auto copy = retained;
    CHECK(!tide::rollback_membership_migration(cluster[2], moves(), copy, error));
    CHECK(tide::mem_at(cluster[2], 1) == 2);
    return 0;
}

int test_partial_accept_completes_by_retry() {
    // Ranks 0 and 1 accept the committed plan 7 -> 8 (they drop plan and
    // journal); ranks 2 and 3 do not. A rollback of epoch 8 is refused where it
    // was accepted, a retried ACCEPT is an idempotent success there, and the
    // retry completes the ACCEPT on ranks 2 and 3.
    Cluster cluster;
    initialize(cluster);
    std::array<tide::MembershipMigrationJournal, 4> journals;
    std::array<tide::MembershipMigrationPlan, 4> plans;
    std::array<std::vector<uint64_t>, 4> epochs;
    std::string error;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        plans[rank].n_servers = 4; plans[rank].epoch_from = 7; plans[rank].epoch_to = 8; plans[rank].moves = moves();
        CHECK(tide::prepare_membership_migration(cluster[rank], plans[rank].moves, 7, 8, journals[rank], error));
        CHECK(tide::commit_membership_migration(cluster[rank], plans[rank].moves, journals[rank], error));
    }
    for (uint32_t rank = 0; rank < 2; ++rank)
        CHECK(tide::accept_membership_migration_history(cluster[rank], plans[rank], journals[rank], epochs[rank], error));
    using D = tide::MigrationRollbackDisposition;
    for (uint32_t rank = 0; rank < 2; ++rank) {
        CHECK(tide::membership_migration_accept_is_repeat(plans[rank], epochs[rank], 8));
        CHECK(tide::membership_migration_rollback_disposition(plans[rank], epochs[rank], 8, 7, 8) == D::REFUSE_ACCEPTED);
        // A later plan this rank never prepared is still an idempotent no-op rollback.
        CHECK(tide::membership_migration_rollback_disposition(plans[rank], epochs[rank], 8, 8, 9) == D::NO_OP);
        CHECK(tide::membership_migration_rollback_disposition(plans[rank], epochs[rank], 8, 5, 6) == D::REFUSE_MISMATCH);
    }
    for (uint32_t rank = 2; rank < 4; ++rank) {
        CHECK(!tide::membership_migration_accept_is_repeat(plans[rank], epochs[rank], 8));
        CHECK(tide::membership_migration_rollback_disposition(plans[rank], epochs[rank], 8, 7, 8) == D::ROLL_BACK);
        CHECK(tide::membership_migration_rollback_disposition(plans[rank], epochs[rank], 8, 8, 9) == D::REFUSE_MISMATCH);
        // The retried ACCEPT reaches the ranks that had not accepted.
        CHECK(tide::accept_membership_migration_history(cluster[rank], plans[rank], journals[rank], epochs[rank], error));
    }
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        CHECK(epochs[rank] == std::vector<uint64_t>{8} && plans[rank].moves.empty());
        CHECK(tide::membership_migration_accept_is_repeat(plans[rank], epochs[rank], 8));
        CHECK(tide::mem_at(cluster[rank], 1) == 2 && tide::mem_at(cluster[rank], 4) == 3);
    }
    uint64_t digest = 0;
    CHECK(active_graph_digest(cluster, digest));
    // Nothing accepted, no plan: neither a repeat nor anything to refuse.
    tide::MembershipMigrationPlan none;
    CHECK(!tide::membership_migration_accept_is_repeat(none, {}, 0));
    CHECK(!tide::membership_migration_accept_is_repeat(none, {8}, 9));
    CHECK(tide::membership_migration_rollback_disposition(none, {}, 0, 0, 1) == D::NO_OP);
    return 0;
}

int test_inverse_move_restores_state() {
    // A MOVE plan, then its inverse (a forward plan from the current owners,
    // rows refreshed from them, as the controller's revert does), each PREPARE,
    // COMMIT and ACCEPT on every server. Restored exactly: ownership on every
    // server, every active adjacency row (the graph digest), and the local maps
    // of every node, the moved ones included: the forward ACCEPT freed each
    // source row into its server's free row-slot pool, and the inverse PREPARE
    // reused that slot, so each moved node returns to its original slot. The
    // forward targets' slots end in their pools (freed by the inverse ACCEPT),
    // and each target appended one slot (n_local + 1 on ranks 2 and 3).
    Cluster cluster;
    initialize(cluster);
    uint64_t before_digest = 0;
    CHECK(active_graph_digest(cluster, before_digest));
    std::array<uint32_t, 4> before_n_local{};
    std::array<std::vector<uint32_t>, 4> before_g2l, before_l2g;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        before_n_local[rank] = tide::n_local_load(cluster[rank]);
        for (uint32_t gid = 0; gid < NODES; ++gid) before_g2l[rank].push_back(cluster[rank].global_to_local.at(gid));
        for (uint32_t slot = 0; slot < before_n_local[rank]; ++slot)
            before_l2g[rank].push_back(cluster[rank].local_to_global.at(slot));
    }
    auto apply = [&](tide::MembershipMigrationPlan plan) -> int {
        // The refresh: each source server supplies the current row of its moved nodes.
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            tide::MembershipMigrationPlan fragment;
            std::string error;
            CHECK(tide::snapshot_membership_migration_source_rows(cluster[rank], plan, fragment, error));
            for (const auto& row : fragment.moves)
                for (auto& move : plan.moves)
                    if (move.global_id == row.global_id) {
                        move.adjacency = row.adjacency; move.adjacency_digest = row.adjacency_digest;
                    }
        }
        std::array<tide::MembershipMigrationJournal, 4> journals;
        std::array<tide::MembershipMigrationPlan, 4> plans;
        std::array<std::vector<uint64_t>, 4> epochs;
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            std::string error;
            plans[rank] = plan;
            CHECK(tide::prepare_membership_migration(cluster[rank], plan.moves, plan.epoch_from, plan.epoch_to,
                                                        journals[rank], error));
            tide::strip_membership_migration_payload(plans[rank]);  // as the server keeps it
        }
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            std::string error;
            CHECK(tide::commit_membership_migration(cluster[rank], plans[rank].moves, journals[rank], error));
        }
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            std::string error;
            CHECK(tide::accept_membership_migration_history(cluster[rank], plans[rank], journals[rank],
                                                              epochs[rank], error));
        }
        return 0;
    };
    tide::MembershipMigrationPlan forward;
    forward.n_servers = 4; forward.epoch_from = 7; forward.epoch_to = 8; forward.moves = moves();
    CHECK(apply(forward) == 0);
    // The forward ACCEPT freed the source rows: gid 1's slot on rank 0, gid 4's on rank 2.
    CHECK(cluster[0].free_slot_pool == std::vector<uint32_t>{before_g2l[0][1]});
    CHECK(cluster[2].free_slot_pool == std::vector<uint32_t>{before_g2l[2][4]});
    CHECK(cluster[0].adj.at(before_g2l[0][1]).capacity() == 0 &&
          tide::migration_slot_generation(cluster[0], before_g2l[0][1]) == 1);
    tide::MembershipMigrationPlan inverse = forward;
    inverse.epoch_from = 8; inverse.epoch_to = 9;
    for (auto& move : inverse.moves) {
        std::swap(move.from_partition, move.to_partition);
        move.adjacency.clear(); move.adjacency_digest = tide::migration_row_digest(move.adjacency);
    }
    CHECK(apply(inverse) == 0);

    // Restored exactly, local maps of the moved nodes included.
    const auto owners = membership();
    for (const auto& server : cluster)
        for (uint32_t gid = 0; gid < NODES; ++gid) CHECK(tide::mem_at(server, gid) == owners[gid]);
    uint64_t after_digest = 0;
    CHECK(active_graph_digest(cluster, after_digest));
    CHECK(after_digest == before_digest);
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        for (uint32_t gid = 0; gid < NODES; ++gid)
            CHECK(cluster[rank].global_to_local.at(gid) == before_g2l[rank][gid]);
        for (uint32_t slot = 0; slot < before_n_local[rank]; ++slot)
            CHECK(cluster[rank].local_to_global.at(slot) == before_l2g[rank][slot]);
    }
    // Slot accounting: each forward target appended one slot, which the inverse
    // ACCEPT freed; the reused source slots left the pools again.
    const std::array<uint32_t, 4> appended = {0, 0, 1, 1};
    for (uint32_t rank = 0; rank < cluster.size(); ++rank)
        CHECK(tide::n_local_load(cluster[rank]) == before_n_local[rank] + appended[rank]);
    CHECK(cluster[0].free_slot_pool.empty() && cluster[1].free_slot_pool.empty());
    CHECK(cluster[2].free_slot_pool == std::vector<uint32_t>{before_n_local[2]});
    CHECK(cluster[3].free_slot_pool == std::vector<uint32_t>{before_n_local[3]});
    CHECK(cluster[2].local_to_global.at(before_n_local[2]) == UINT32_MAX &&
          cluster[2].adj.at(before_n_local[2]).capacity() == 0);
    // Generations: gid 1's original slot on rank 0 was freed (1) and reused (2).
    CHECK(tide::migration_slot_generation(cluster[0], before_g2l[0][1]) == 2);
    return 0;
}

int test_reset_clears_free_slot_pool() {
    // Server RESET clears the rows and calls n_local_set(data, 0) before
    // reloading (it now also clears the pool explicitly); the free row-slot pool
    // names slots of the cleared rows and empties with them.
    Cluster cluster;
    initialize(cluster);
    cluster[3].free_slot_pool = {1};
    tide::n_local_set(cluster[3], 0);
    CHECK(cluster[3].free_slot_pool.empty());
    return 0;
}

int test_free_slot_pool_reuse_and_rollback_return() {
    // A free row slot (released adjacency, no mapping) is reused by the next
    // PREPARE before appending, and a rollback returns exactly that slot.
    Cluster cluster;
    initialize(cluster);
    auto& target = cluster[2];
    const uint32_t appended = tide::n_local_load(target);  // 2: the next append slot
    target.adj.ensure_capacity_for(appended);
    target.local_to_global.ensure_capacity_for(appended, UINT32_MAX);
    tide::n_local_set(target, appended + 1);  // slot `appended` exists, unmapped and empty
    tide::ensure_node_locks(target, appended + 1);
    target.free_slot_pool = {appended};
    const auto batch = moves();  // gid 1 lands on rank 2
    std::array<tide::MembershipMigrationJournal, 4> journals;
    std::string error;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank)
        CHECK(tide::prepare_membership_migration(cluster[rank], batch, 7, 8, journals[rank], error));
    CHECK(journals[2].entries[0].target_reused && journals[2].entries[0].target_local_idx == appended);
    CHECK(target.free_slot_pool.empty() && tide::n_local_load(target) == appended + 1);
    CHECK(target.global_to_local.at(1) == appended && target.adj.at(appended) == batch[0].adjacency);
    for (uint32_t rank = 0; rank < cluster.size(); ++rank)
        CHECK(tide::rollback_membership_migration(cluster[rank], batch, journals[rank], error));
    CHECK(target.free_slot_pool == std::vector<uint32_t>{appended});
    CHECK(target.adj.at(appended).capacity() == 0 && target.local_to_global.at(appended) == UINT32_MAX &&
          target.global_to_local.at(1) == UINT32_MAX);
    // A pool entry that still holds a row violates the invariant: PREPARE refuses before any change.
    target.adj.at(appended) = {5};
    tide::MembershipMigrationJournal refused;
    CHECK(!tide::prepare_membership_migration(target, batch, 7, 8, refused, error));
    CHECK(error.find("pool invariant") != std::string::npos && !refused.prepared);
    return 0;
}

int test_commit_requires_target_prepare() {
    Cluster cluster;
    initialize(cluster);
    auto batch = moves();
    std::array<tide::MembershipMigrationJournal, 4> journals;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::prepare_membership_migration(
            cluster[rank], batch, 10, 11, journals[rank], error));
    }
    const auto target_idx = cluster[2].global_to_local.at(1);
    cluster[2].global_to_local.at(1) = UINT32_MAX;
    std::string error;
    CHECK(!tide::commit_membership_migration(
        cluster[2], batch, journals[2], error));
    CHECK(error.find("target shadow row missing") != std::string::npos);
    cluster[2].global_to_local.at(1) = target_idx;
    return 0;
}

int test_bad_payload_fails_before_mutation() {
    Cluster cluster;
    initialize(cluster);
    auto batch = moves();
    batch[0].adjacency_digest++;
    tide::MembershipMigrationJournal journal;
    std::string error;
    const uint32_t before = tide::n_local_load(cluster[2]);
    CHECK(!tide::prepare_membership_migration(
        cluster[2], batch, 1, 2, journal, error));
    CHECK(tide::n_local_load(cluster[2]) == before);
    CHECK(cluster[2].global_to_local.at(1) == UINT32_MAX);
    return 0;
}

int test_target_capacity_failure_fails_before_mapping_publication() {
    Cluster cluster;
    initialize(cluster);
    const auto batch = moves();
    tide::n_local_set(cluster[2], std::numeric_limits<uint32_t>::max());
    tide::MembershipMigrationJournal journal;
    std::string error;
    CHECK(!tide::prepare_membership_migration(
        cluster[2], batch, 3, 4, journal, error));
    CHECK(error.find("local-index overflow") != std::string::npos);
    CHECK(cluster[2].global_to_local.at(1) == UINT32_MAX);
    CHECK(!journal.prepared);
    CHECK(journal.entries.empty());
    return 0;
}

int test_refresh_uses_current_post_maintenance_row() {
    Cluster cluster;
    initialize(cluster);
    tide::MembershipMigrationPlan move_template{};
    move_template.n_servers = 4;
    move_template.epoch_from = 7;
    move_template.epoch_to = 8;
    move_template.moves = moves();

    const uint32_t source_local = cluster[0].global_to_local.at(1);
    cluster[0].adj.at(source_local).push_back(7);
    tide::MembershipMigrationJournal stale_journal;
    std::string error;
    CHECK(!tide::prepare_membership_migration(
        cluster[0], move_template.moves, 7, 8, stale_journal, error));
    CHECK(error.find("source adjacency differs") != std::string::npos);

    tide::MembershipMigrationPlan fragment;
    CHECK(tide::snapshot_membership_migration_source_rows(
        cluster[0], move_template, fragment, error));
    CHECK(fragment.moves.size() == 1);
    CHECK(fragment.moves[0].global_id == 1);
    CHECK(fragment.moves[0].adjacency == cluster[0].adj.at(source_local));
    CHECK(fragment.moves[0].adjacency_digest !=
          move_template.moves[0].adjacency_digest);

    auto refreshed = move_template.moves;
    refreshed[0] = fragment.moves[0];
    tide::MembershipMigrationJournal refreshed_journal;
    CHECK(tide::prepare_membership_migration(
        cluster[0], refreshed, 7, 8, refreshed_journal, error));
    return 0;
}

int test_current_post_maintenance_row_survives_commit_and_rollback() {
    Cluster cluster;
    initialize(cluster);
    tide::MembershipMigrationPlan refreshed_plan{};
    refreshed_plan.n_servers = 4;
    refreshed_plan.epoch_from = 7;
    refreshed_plan.epoch_to = 8;
    refreshed_plan.moves = moves();

    // Model a completed backlink update before PREPARE.  The refreshed plan
    // must migrate this exact current row and rollback must restore that same
    // row at the original owner, not the static row embedded in the plan.
    const uint32_t source_local = cluster[0].global_to_local.at(1);
    cluster[0].adj.at(source_local).push_back(7);
    refreshed_plan.moves[0].adjacency = cluster[0].adj.at(source_local);
    refreshed_plan.moves[0].adjacency_digest =
        tide::migration_row_digest(refreshed_plan.moves[0].adjacency);

    uint64_t post_maintenance_digest = 0;
    CHECK(active_graph_digest(cluster, post_maintenance_digest));
    std::array<tide::MembershipMigrationJournal, 4> journals;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::prepare_membership_migration(
            cluster[rank], refreshed_plan.moves, 7, 8,
            journals[rank], error));
    }
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::commit_membership_migration(
            cluster[rank], refreshed_plan.moves, journals[rank], error));
    }
    uint64_t committed_digest = 0;
    CHECK(active_graph_digest(cluster, committed_digest));
    CHECK(committed_digest == post_maintenance_digest);

    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::rollback_membership_migration(
            cluster[rank], refreshed_plan.moves, journals[rank], error));
    }
    uint64_t rolled_back_digest = 0;
    CHECK(active_graph_digest(cluster, rolled_back_digest));
    CHECK(rolled_back_digest == post_maintenance_digest);
    CHECK(cluster[0].adj.at(cluster[0].global_to_local.at(1)) ==
          refreshed_plan.moves[0].adjacency);
    return 0;
}

int test_interleaved_insert_survives_rollback() {
    Cluster cluster;
    initialize(cluster);
    tide::MembershipMigrationPlan plan{};
    plan.n_servers = 4;
    plan.epoch_from = 7;
    plan.epoch_to = 8;
    plan.moves = moves();
    std::array<tide::MembershipMigrationJournal, 4> journals;
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        std::string error;
        CHECK(tide::prepare_membership_migration(
            cluster[rank], plan.moves, 7, 8, journals[rank], error));
    }

    // Model an insert allocated after PREPARE on a server that already has a
    // target shadow. Rollback must not rewind n_local across this live node.
    constexpr uint32_t INSERTED = NODES;
    auto& target = cluster[2];
    target.global_to_local.ensure_capacity_for(INSERTED, UINT32_MAX);
    tide::server_ensure_mem_capacity(target, INSERTED);
    const uint32_t inserted_local = tide::n_local_alloc(target);
    target.adj.ensure_capacity_for(inserted_local);
    target.local_to_global.ensure_capacity_for(inserted_local, UINT32_MAX);
    tide::ensure_node_locks(target, inserted_local + 1);
    target.global_to_local.at(INSERTED) = inserted_local;
    target.local_to_global.at(inserted_local) = INSERTED;
    target.adj.at(inserted_local) = {0};
    tide::mem_set(target, INSERTED, 2);
    tide::n_base_bump(target, INSERTED + 1);
    const uint32_t target_n_local_with_insert = tide::n_local_load(target);

    std::string error;
    uint64_t post_update_digest = 0;
    CHECK(active_graph_digest(cluster, post_update_digest));
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        CHECK(tide::commit_membership_migration(
            cluster[rank], plan.moves, journals[rank], error));
    }
    uint64_t committed_digest = 0;
    CHECK(active_graph_digest(cluster, committed_digest));
    CHECK(committed_digest == post_update_digest);

    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        CHECK(tide::rollback_membership_migration(
            cluster[rank], plan.moves, journals[rank], error));
    }
    uint64_t rolled_back_digest = 0;
    CHECK(active_graph_digest(cluster, rolled_back_digest));
    CHECK(rolled_back_digest == post_update_digest);
    CHECK(tide::n_local_load(target) == target_n_local_with_insert);
    CHECK(target.global_to_local.at(INSERTED) == inserted_local);
    CHECK(target.local_to_global.at(inserted_local) == INSERTED);
    CHECK(target.adj.at(inserted_local) == std::vector<uint32_t>{0});
    // n_local could not shrink back over the slot PREPARE appended for node 1:
    // that slot, unmapped with a zero-capacity row, is returned to the free
    // pool, and the next PREPARE onto this server reuses it.
    const uint32_t appended = journals[2].entries[0].target_local_idx;
    CHECK(!journals[2].entries[0].target_reused && appended < inserted_local);
    CHECK(target.free_slot_pool == std::vector<uint32_t>{appended});
    CHECK(target.local_to_global.at(appended) == UINT32_MAX && target.adj.at(appended).capacity() == 0 &&
          target.global_to_local.at(1) == UINT32_MAX);
    std::array<uint32_t, NODES + 1> owners{};
    for (uint32_t gid = 0; gid < NODES; ++gid) owners[gid] = membership()[gid];
    owners[INSERTED] = 2;
    CHECK(tide::membership_check(target, owners.data(), NODES + 1).pool_ok);
    tide::MembershipMigrationJournal again;
    CHECK(tide::prepare_membership_migration(target, plan.moves, 7, 8, again, error));
    CHECK(again.entries[0].target_reused && again.entries[0].target_local_idx == appended &&
          target.free_slot_pool.empty() && tide::n_local_load(target) == target_n_local_with_insert);
    return 0;
}


}  // namespace

// Plan wire version 3: every move carries its vector and the vector's digest.
// It round-trips exactly; the serializer writes version 2 when no move carries
// a vector, refuses moves with vectors of different lengths or a wrong digest,
// and the loader refuses a corrupted vector.
int test_plan_v3_vectors() {
    tide::MembershipMigrationPlan plan;
    plan.n_servers = 4; plan.epoch_from = 7; plan.epoch_to = 8;
    plan.moves = moves();
    std::vector<char> v2;
    std::string error;
    CHECK(tide::serialize_membership_migration_plan(plan, v2, error));
    uint32_t version = 0;
    std::memcpy(&version, v2.data() + 8, 4);
    CHECK(version == 2);
    for (auto& move : plan.moves) {
        move.vector = {static_cast<float>(move.global_id), -1.5f, 0.25f};
        move.vector_digest = tide::migration_vector_digest(move.vector);
    }
    std::vector<char> v3;
    CHECK(tide::serialize_membership_migration_plan(plan, v3, error));
    std::memcpy(&version, v3.data() + 8, 4);
    CHECK(version == 3 && v3.size() == v2.size() + 4 + plan.moves.size() * (8 + 3 * 4));
    tide::MembershipMigrationPlan parsed;
    CHECK(tide::load_membership_migration_plan_bytes(v3.data(), v3.size(), NODES, 4, parsed, error));
    CHECK(parsed.moves.size() == 2);
    for (size_t i = 0; i < 2; ++i) {
        CHECK(parsed.moves[i].vector == plan.moves[i].vector &&
              parsed.moves[i].vector_digest == plan.moves[i].vector_digest &&
              parsed.moves[i].adjacency == plan.moves[i].adjacency);
    }
    std::vector<char> again;
    CHECK(tide::serialize_membership_migration_plan(parsed, again, error) && again == v3);
    // A flipped vector byte: the digest refuses the plan.
    auto corrupted = v3;
    corrupted[corrupted.size() - 2] ^= 0x10;
    CHECK(!tide::load_membership_migration_plan_bytes(corrupted.data(), corrupted.size(), NODES, 4, parsed,
                                                        error) && error.find("vector digest") != std::string::npos);
    // Mixed lengths and a stale digest are refused by the serializer.
    auto mixed = plan;
    mixed.moves[1].vector.clear();
    CHECK(!tide::serialize_membership_migration_plan(mixed, again, error));
    auto stale = plan;
    stale.moves[0].vector[0] += 1.0f;
    CHECK(!tide::serialize_membership_migration_plan(stale, again, error));
    return 0;
}

// Own-only storage: a MOVE ships the node's original vector. The source's
// REFRESH fragment carries it; a plan without it is refused before any change;
// PREPARE installs it on the target before the mapping is published, COMMIT
// checks it and the source then holds only the code; ROLLBACK restores the
// source's own vector (never freed before ACCEPT).
int test_own_only_move_ships_vector() {
    Cluster cluster;
    initialize(cluster);
    ann::PQTable pq;
    pq.dim = 2; pq.M = 1; pq.sub_dim = 2;
    pq.centroids.assign(2 * ann::PQTable::K, 0.0f);
    pq.build_sym_table();
    auto vector_of = [](uint32_t gid) { return std::vector<float>{static_cast<float>(gid), 10.0f + gid}; };
    for (auto& server : cluster) {
        server.own_only = true;
        server.pq_table = pq;
        server.pq_codes_static.assign(NODES, 0);
        server.pq_codes_delta.init(1);
        server.full_own.init(2);
        for (uint32_t slot = 0; slot < tide::n_local_load(server); ++slot) {
            tide::store_own_vector(server, slot, vector_of(server.local_to_global.at(slot)).data());
        }
    }
    tide::MembershipMigrationPlan move_template;
    move_template.n_servers = 4; move_template.epoch_from = 7; move_template.epoch_to = 8;
    move_template.moves = moves();  // node 1: 0 -> 2, node 4: 2 -> 3
    // REFRESH: each source fills its rows' vectors.
    std::vector<tide::MembershipMove> refreshed;
    for (uint32_t rank = 0; rank < 4; ++rank) {
        tide::MembershipMigrationPlan fragment;
        std::string error;
        CHECK(tide::snapshot_membership_migration_source_rows(cluster[rank], move_template, fragment, error));
        for (auto& move : fragment.moves) {
            CHECK(move.vector == vector_of(move.global_id) &&
                  move.vector_digest == tide::migration_vector_digest(move.vector));
            refreshed.push_back(move);
        }
    }
    std::sort(refreshed.begin(), refreshed.end(),
              [](const auto& a, const auto& b) { return a.global_id < b.global_id; });
    CHECK(refreshed.size() == 2 && refreshed[0].global_id == 1 && refreshed[1].global_id == 4);
    // A vectorless plan is refused on an own-only server and changes nothing.
    {
        tide::MembershipMigrationJournal journal;
        std::string error;
        const uint32_t before = tide::n_local_load(cluster[2]);
        CHECK(!tide::prepare_membership_migration(cluster[2], move_template.moves, 7, 8, journal, error));
        CHECK(error.find("lacks the node's vector") != std::string::npos);
        CHECK(tide::n_local_load(cluster[2]) == before && cluster[2].global_to_local.at(1) == UINT32_MAX);
    }
    std::array<tide::MembershipMigrationJournal, 4> journals;
    for (uint32_t rank = 0; rank < 4; ++rank) {
        std::string error;
        CHECK(tide::prepare_membership_migration(cluster[rank], refreshed, 7, 8, journals[rank], error));
    }
    const auto vector_1 = vector_of(1), vector_4 = vector_of(4);
    // Target 2 holds node 1's vector and target 3 node 4's, before COMMIT.
    CHECK(tide::own_vec(cluster[2], 1) != nullptr &&
          std::equal(vector_1.begin(), vector_1.end(), tide::own_vec(cluster[2], 1)));
    CHECK(tide::own_vec(cluster[3], 4) != nullptr &&
          std::equal(vector_4.begin(), vector_4.end(), tide::own_vec(cluster[3], 4)));
    CHECK(tide::own_vec(cluster[1], 1) == nullptr);  // a bystander holds no copy
    for (uint32_t rank = 0; rank < 4; ++rank) {
        std::string error;
        CHECK(tide::commit_membership_migration(cluster[rank], refreshed, journals[rank], error));
    }
    // The sources now hold only the codes; the targets the vectors.
    CHECK(tide::own_vec(cluster[0], 1) == nullptr && tide::own_vec(cluster[2], 4) == nullptr);
    CHECK(tide::own_vec(cluster[2], 1) != nullptr && tide::own_vec(cluster[3], 4) != nullptr);
    for (uint32_t rank = 0; rank < 4; ++rank) {
        std::string error;
        CHECK(tide::rollback_membership_migration(cluster[rank], refreshed, journals[rank], error));
    }
    // Rollback restores every source vector byte for byte; the targets lose the mapping.
    CHECK(tide::own_vec(cluster[0], 1) != nullptr &&
          std::equal(vector_1.begin(), vector_1.end(), tide::own_vec(cluster[0], 1)));
    CHECK(tide::own_vec(cluster[2], 4) != nullptr &&
          std::equal(vector_4.begin(), vector_4.end(), tide::own_vec(cluster[2], 4)));
    CHECK(tide::own_vec(cluster[2], 1) == nullptr && tide::own_vec(cluster[3], 4) == nullptr);
    // COMMIT refuses a target whose vector is missing (an own-only target that
    // prepared from a full-replication journal is the only way; simulate it).
    tide::MembershipMigrationJournal journal;
    std::string error;
    auto& target = cluster[2];
    CHECK(tide::prepare_membership_migration(target, refreshed, 9, 10, journal, error));
    const uint32_t slot = target.global_to_local.at(1);
    target.global_to_local.at(1) = UINT32_MAX;  // hide the vector, keep the journal slot
    target.full_own.clear();
    target.global_to_local.at(1) = slot;
    CHECK(!tide::commit_membership_migration(target, refreshed, journal, error));
    CHECK(error.find("target vector missing") != std::string::npos);
    return 0;
}

// Own-only storage: a committed MOVE, then its inverse (a forward plan from the
// new owners, rows and vectors refreshed from them, as the controller's revert
// does), each PREPARE, COMMIT and ACCEPT on every server. After the forward
// ACCEPT the targets hold the moved nodes' original vectors byte for byte and the
// sources and bystanders hold none; after the inverse ACCEPT the original owners
// hold them again, byte for byte (in their original slots, freed and reused),
// the forward targets hold none, ownership and every active row are restored,
// and every node's vector on its owner is its original vector (a self-query at
// distance exactly 0). The codes never change. Each ACCEPT frees the source
// rows and own-vector slots, and the inverse reuses them.
int test_own_only_inverse_move_and_accept() {
    Cluster cluster;
    initialize(cluster);
    ann::PQTable pq;
    pq.dim = 2; pq.M = 1; pq.sub_dim = 2;
    pq.centroids.assign(2 * ann::PQTable::K, 0.0f);
    pq.build_sym_table();
    auto vector_of = [](uint32_t gid) { return std::vector<float>{static_cast<float>(gid) + 0.25f, 10.5f - gid}; };
    for (auto& server : cluster) {
        server.own_only = true;
        server.pq_table = pq;
        server.pq_codes_static.assign(NODES, 7);
        server.pq_codes_delta.init(1);
        server.full_own.init(2);
        for (uint32_t slot = 0; slot < tide::n_local_load(server); ++slot)
            tide::store_own_vector(server, slot, vector_of(server.local_to_global.at(slot)).data());
    }
    uint64_t before_digest = 0;
    CHECK(active_graph_digest(cluster, before_digest));
    const std::vector<uint8_t> codes_before = cluster[1].pq_codes_static;
    auto apply = [&](tide::MembershipMigrationPlan plan) -> int {
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            tide::MembershipMigrationPlan fragment;
            std::string error;
            CHECK(tide::snapshot_membership_migration_source_rows(cluster[rank], plan, fragment, error));
            for (const auto& row : fragment.moves)
                for (auto& move : plan.moves)
                    if (move.global_id == row.global_id) {
                        move.adjacency = row.adjacency; move.adjacency_digest = row.adjacency_digest;
                        move.vector = row.vector; move.vector_digest = row.vector_digest;
                    }
        }
        for (const auto& move : plan.moves) {
            const auto expected = vector_of(move.global_id);
            CHECK(move.vector == expected && move.vector_digest == tide::migration_vector_digest(expected));
        }
        std::array<tide::MembershipMigrationJournal, 4> journals;
        std::array<tide::MembershipMigrationPlan, 4> plans;
        std::array<std::vector<uint64_t>, 4> epochs;
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            std::string error;
            plans[rank] = plan;
            CHECK(tide::prepare_membership_migration(cluster[rank], plan.moves, plan.epoch_from, plan.epoch_to,
                                                        journals[rank], error));
        }
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            std::string error;
            CHECK(tide::commit_membership_migration(cluster[rank], plan.moves, journals[rank], error));
        }
        for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
            std::string error;
            CHECK(tide::accept_membership_migration_history(cluster[rank], plans[rank], journals[rank],
                                                              epochs[rank], error));
            CHECK(epochs[rank] == std::vector<uint64_t>{plan.epoch_to});
        }
        return 0;
    };
    // Every node's vector is held exactly by its owner (distance 0 to the
    // original) and by no other server.
    auto vectors_on_owners = [&]() -> int {
        for (uint32_t gid = 0; gid < NODES; ++gid) {
            const uint32_t owner = tide::mem_at(cluster[0], gid);
            const auto original = vector_of(gid);
            for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
                const float* held = tide::own_vec(cluster[rank], gid);
                if (rank != owner) { CHECK(held == nullptr); continue; }
                CHECK(held != nullptr && std::equal(original.begin(), original.end(), held));
                CHECK(ann::compute_distance(original.data(), held, 2) == 0.0f);
            }
        }
        return 0;
    };
    tide::MembershipMigrationPlan forward;
    forward.n_servers = 4; forward.epoch_from = 7; forward.epoch_to = 8; forward.moves = moves();
    for (auto& move : forward.moves) { move.adjacency.clear(); move.adjacency_digest = tide::migration_row_digest({}); }
    CHECK(apply(forward) == 0);
    CHECK(tide::mem_at(cluster[3], 1) == 2 && tide::mem_at(cluster[0], 4) == 3);
    CHECK(vectors_on_owners() == 0);
    uint64_t forward_digest = 0;
    CHECK(active_graph_digest(cluster, forward_digest) && forward_digest == before_digest);
    tide::MembershipMigrationPlan inverse = forward;
    inverse.epoch_from = 8; inverse.epoch_to = 9;
    for (auto& move : inverse.moves) {
        std::swap(move.from_partition, move.to_partition);
        move.adjacency.clear(); move.adjacency_digest = tide::migration_row_digest(move.adjacency);
        move.vector.clear(); move.vector_digest = 0;
    }
    CHECK(apply(inverse) == 0);
    const auto owners = membership();
    for (const auto& server : cluster)
        for (uint32_t gid = 0; gid < NODES; ++gid) CHECK(tide::mem_at(server, gid) == owners[gid]);
    CHECK(vectors_on_owners() == 0);
    uint64_t after_digest = 0;
    CHECK(active_graph_digest(cluster, after_digest) && after_digest == before_digest);
    for (const auto& server : cluster) CHECK(server.pq_codes_static == codes_before);
    // Slots: the forward ACCEPT freed each source row and own-vector slot, the
    // inverse PREPARE reused it, so each moved node is back in its original slot
    // (gid 1: rank 0 slot 1; gid 4: rank 2 slot 0); the forward targets' slots
    // (rank 2 and rank 3, slot 2) are free, their vectors zeroed.
    CHECK(cluster[0].global_to_local.at(1) == 1 && cluster[2].global_to_local.at(4) == 0);
    CHECK(cluster[2].free_slot_pool == std::vector<uint32_t>{2} &&
          cluster[3].free_slot_pool == std::vector<uint32_t>{2});
    for (uint32_t rank : {2u, 3u}) {
        const float* freed = cluster[rank].full_own.vec(2);
        CHECK(freed[0] == 0.0f && freed[1] == 0.0f && cluster[rank].adj.at(2).capacity() == 0);
    }
    return 0;
}


// An own-only cluster of the eight test nodes: every server keeps the vectors
// of its own nodes in full_own (by local slot) and a code of every node.
std::vector<float> own_vector(uint32_t gid) { return {static_cast<float>(gid) * 1.5f + 0.125f, 7.75f - gid}; }
void make_own_only(Cluster& cluster) {
    ann::PQTable pq;
    pq.dim = 2; pq.M = 1; pq.sub_dim = 2;
    pq.centroids.assign(2 * ann::PQTable::K, 0.0f);
    pq.build_sym_table();
    for (auto& server : cluster) {
        server.own_only = true;
        server.pq_table = pq;
        server.pq_codes_static.assign(NODES, 3);
        server.pq_codes_delta.init(1);
        server.full_own.init(2);
        for (uint32_t slot = 0; slot < tide::n_local_load(server); ++slot)
            tide::store_own_vector(server, slot, own_vector(server.local_to_global.at(slot)).data());
    }
}

// What the coordinator does before PREPARE: the move template goes to every
// server, each source answers with its current rows and (own-only) vectors.
int refresh(Cluster& cluster, tide::MembershipMigrationPlan& plan) {
    for (uint32_t rank = 0; rank < cluster.size(); ++rank) {
        tide::MembershipMigrationPlan fragment;
        std::string error;
        CHECK(tide::snapshot_membership_migration_source_rows(cluster[rank], plan, fragment, error));
        for (const auto& row : fragment.moves)
            for (auto& move : plan.moves)
                if (move.global_id == row.global_id) {
                    move.adjacency = row.adjacency; move.adjacency_digest = row.adjacency_digest;
                    move.vector = row.vector; move.vector_digest = row.vector_digest;
                }
    }
    return 0;
}

// Plan wire version 4, the per-target PREPARE fragment: every server's fragment
// has the whole move list (ids, owners, rows, row and vector digests); only the
// target's moves carry their floats. The loader enforces that split, the digest
// and the target's range; the serializer needs a vector on every move.
int test_plan_v4_per_target_fragments() {
    tide::MembershipMigrationPlan plan;
    plan.n_servers = 4; plan.epoch_from = 7; plan.epoch_to = 8;
    plan.moves = moves();  // node 1: 0 -> 2, node 4: 2 -> 3
    std::string error;
    std::vector<char> bytes;
    CHECK(!tide::serialize_membership_migration_fragment(plan, 2, bytes, error) &&
          error.find("needs a vector") != std::string::npos);
    for (auto& move : plan.moves) {
        move.vector = {static_cast<float>(move.global_id), -2.0f, 0.5f};
        move.vector_digest = tide::migration_vector_digest(move.vector);
    }
    std::vector<char> v3;
    CHECK(tide::serialize_membership_migration_plan(plan, v3, error));
    for (uint32_t target = 0; target < 4; ++target) {
        CHECK(tide::serialize_membership_migration_fragment(plan, target, bytes, error));
        uint32_t version = 0, vector_target = 0;
        std::memcpy(&version, bytes.data() + 8, 4);
        std::memcpy(&vector_target, bytes.data() + 20, 4);
        CHECK(version == 4 && vector_target == target);
        size_t carried = 0;
        for (const auto& move : plan.moves) carried += move.to_partition == target ? 1 : 0;
        // v3 + the target field - the floats of every move not addressed here.
        CHECK(bytes.size() == v3.size() + 4 - (plan.moves.size() - carried) * 3 * 4);
        tide::MembershipMigrationPlan parsed;
        CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, parsed, error));
        CHECK(parsed.vector_target == target && parsed.moves.size() == plan.moves.size() &&
              parsed.epoch_from == 7 && parsed.epoch_to == 8 && parsed.vector_dim == 3);
        // The serializer reports the float bytes it wrote into this fragment.
        uint64_t written = 0;
        std::vector<char> counted;
        CHECK(tide::serialize_membership_migration_fragment(plan, target, counted, error, &written) &&
              counted == bytes && written == carried * 3 * 4);
        for (size_t i = 0; i < plan.moves.size(); ++i) {
            const auto& got = parsed.moves[i];
            const auto& want = plan.moves[i];
            CHECK(got.global_id == want.global_id && got.from_partition == want.from_partition &&
                  got.to_partition == want.to_partition && got.adjacency == want.adjacency &&
                  got.adjacency_digest == want.adjacency_digest && got.vector_digest == want.vector_digest);
            CHECK(want.to_partition == target ? got.vector == want.vector : got.vector.empty());
        }
    }
    // Fragment of rank 2 (carries node 1's floats): retargeted to rank 1, its
    // floats are not where the loader reads a move header, so it fails;
    // a flipped float fails the digest; a target beyond the servers is refused.
    CHECK(tide::serialize_membership_migration_fragment(plan, 2, bytes, error));
    auto retargeted = bytes;
    const uint32_t one = 1;
    std::memcpy(retargeted.data() + 20, &one, 4);
    tide::MembershipMigrationPlan parsed;
    CHECK(!tide::load_membership_migration_plan_bytes(retargeted.data(), retargeted.size(), NODES, 4, parsed,
                                                        error));
    auto corrupted = bytes;
    // node 1's floats follow its row (2 neighbours) and its vector digest.
    const size_t header = 8 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 8;
    const size_t node1_floats = header + (4 + 4 + 4 + 4 + 8) + 2 * 4 + 8;
    corrupted[node1_floats + 1] ^= 0x40;
    CHECK(!tide::load_membership_migration_plan_bytes(corrupted.data(), corrupted.size(), NODES, 4, parsed, error) &&
          error.find("vector digest") != std::string::npos);
    auto beyond = bytes;
    const uint32_t nine = 9;
    std::memcpy(beyond.data() + 20, &nine, 4);
    CHECK(!tide::load_membership_migration_plan_bytes(beyond.data(), beyond.size(), NODES, 4, parsed, error) &&
          error.find("vector target") != std::string::npos);
    CHECK(!tide::serialize_membership_migration_fragment(plan, 4, bytes, error));
    // A stale digest (the coordinator changed a float after REFRESH) is refused.
    auto stale = plan;
    stale.moves[0].vector[1] = 9.0f;
    CHECK(!tide::serialize_membership_migration_fragment(stale, 2, bytes, error));
    return 0;
}

// A shipped vector must have the server's dimension: PREPARE refuses a plan
// whose vector is longer or shorter, on the target and elsewhere, under both
// storage modes, before anything changes; and a digest that does not match.
int test_prepare_refuses_vector_dimension_mismatch() {
    Cluster cluster;
    initialize(cluster);
    make_own_only(cluster);
    tide::MembershipMigrationPlan plan;
    plan.n_servers = 4; plan.epoch_from = 7; plan.epoch_to = 8;
    plan.moves = moves();
    CHECK(refresh(cluster, plan) == 0);
    auto wide = plan;
    for (auto& move : wide.moves) {
        move.vector.push_back(1.0f);  // dim 3 against the servers' 2
        move.vector_digest = tide::migration_vector_digest(move.vector);
    }
    for (uint32_t rank = 0; rank < 4; ++rank) {
        const uint32_t before = tide::n_local_load(cluster[rank]);
        tide::MembershipMigrationJournal journal;
        std::string error;
        CHECK(!tide::prepare_membership_migration(cluster[rank], wide.moves, 7, 8, journal, error));
        CHECK(error.find("dimension differs") != std::string::npos && !journal.prepared);
        CHECK(tide::n_local_load(cluster[rank]) == before && tide::mem_at(cluster[rank], 1) == 0);
    }
    CHECK(cluster[2].global_to_local.at(1) == UINT32_MAX);
    auto stale = plan;
    stale.moves[0].vector[0] += 0.5f;  // digest of the original vector
    tide::MembershipMigrationJournal journal;
    std::string error;
    CHECK(!tide::prepare_membership_migration(cluster[2], stale.moves, 7, 8, journal, error) &&
          error.find("vector digest") != std::string::npos);
    // Full replication: the same refusal for a vector of another dimension.
    Cluster full;
    initialize(full);
    CHECK(!tide::prepare_membership_migration(full[2], wide.moves, 7, 8, journal, error) &&
          error.find("dimension differs") != std::string::npos);
    return 0;
}

// The per-target MOVE on an own-only cluster, as the coordinator and the
// servers run it: REFRESH (the sources ship rows and vectors), one v4 fragment
// per server, PREPARE from the server's own fragment, COMMIT and ACCEPT.
//  - Bystanders and sources receive no float of a vector they do not install
//    and never hold a moved node's vector; every server sees every move.
//  - The target holds the vector, byte for byte, at the slot its mapping names
//    as soon as PREPARE published that mapping (before COMMIT).
//  - After COMMIT the target holds the vector and the source returns null.
//  - ROLLBACK of a committed fragment-prepared MOVE restores the source vector
//    byte for byte.
//  - ACCEPT frees the source row and own-vector slot into the pool (generation
//    bumped, row at zero capacity, vector zeroed); a copy of the accepted
//    journal can never roll back; the next PREPARE onto that server reuses the
//    pooled slot for the incoming node, vector included.
//  - membership_check agrees with the cluster after every phase.
int test_per_target_move_accept_frees_and_reuses() {
    Cluster cluster;
    initialize(cluster);
    make_own_only(cluster);
    auto check_all = [&](const std::array<uint32_t, NODES>& owners) -> int {
        for (uint32_t rank = 0; rank < 4; ++rank) {
            const auto c = tide::membership_check(cluster[rank], owners.data(), NODES);
            CHECK(c.membership_mismatches == 0 && c.mapping_mismatches == 0 && c.vector_mismatches == 0 &&
                  c.missing_codes == 0 && c.pool_ok && c.first_mismatch == UINT32_MAX && c.owned == c.mapped &&
                  c.vectors == c.owned);
            // The digest of the own vectors is the digest of the originals of the owned nodes.
            uint64_t expected = 0;
            for (uint32_t gid = 0; gid < NODES; ++gid)
                if (owners[gid] == rank) expected += tide::node_vector_hash(gid, own_vector(gid).data(), 2);
            CHECK(c.vector_digest == expected);
        }
        return 0;
    };
    auto fragments_prepare = [&](tide::MembershipMigrationPlan plan,
                                 std::array<tide::MembershipMigrationPlan, 4>& plans,
                                 std::array<tide::MembershipMigrationJournal, 4>& journals) -> int {
        CHECK(refresh(cluster, plan) == 0);
        for (uint32_t rank = 0; rank < 4; ++rank) {
            std::vector<char> bytes;
            std::string error;
            CHECK(tide::serialize_membership_migration_fragment(plan, rank, bytes, error));
            CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, plans[rank],
                                                               error));
            CHECK(plans[rank].vector_target == rank && plans[rank].moves.size() == plan.moves.size());
            for (const auto& move : plans[rank].moves)
                CHECK(move.vector.empty() == (move.to_partition != rank));
            CHECK(tide::prepare_membership_migration(cluster[rank], plans[rank].moves, plan.epoch_from,
                                                        plan.epoch_to, journals[rank], error));
            tide::strip_membership_migration_payload(plans[rank]);
        }
        return 0;
    };
    const auto start = membership();
    CHECK(check_all(start) == 0);
    tide::MembershipMigrationPlan forward;
    forward.n_servers = 4; forward.epoch_from = 7; forward.epoch_to = 8;
    forward.moves = moves();  // node 1: 0 -> 2, node 4: 2 -> 3
    for (auto& move : forward.moves) { move.adjacency.clear(); move.adjacency_digest = tide::migration_row_digest({}); }

    // 1. Forward MOVE, committed, then rolled back: the source vector is restored.
    {
        std::array<tide::MembershipMigrationPlan, 4> plans;
        std::array<tide::MembershipMigrationJournal, 4> journals;
        CHECK(fragments_prepare(forward, plans, journals) == 0);
        const uint32_t slot = cluster[2].global_to_local.at(1);
        CHECK(slot != UINT32_MAX && cluster[2].local_to_global.at(slot) == 1);
        const auto v1 = own_vector(1);
        CHECK(std::memcmp(cluster[2].full_own.vec(slot), v1.data(), sizeof(float) * 2) == 0);
        CHECK(tide::mem_at(cluster[2], 1) == 0);  // ownership not yet published
        for (uint32_t rank : {1u, 3u}) CHECK(tide::own_vec(cluster[rank], 1) == nullptr);
        for (uint32_t rank = 0; rank < 4; ++rank) {
            std::string error;
            CHECK(tide::commit_membership_migration(cluster[rank], plans[rank].moves, journals[rank], error));
        }
        CHECK(tide::own_vec(cluster[0], 1) == nullptr && tide::own_vec(cluster[2], 1) != nullptr);
        for (uint32_t rank = 0; rank < 4; ++rank) {
            std::string error;
            CHECK(tide::rollback_membership_migration(cluster[rank], plans[rank].moves, journals[rank], error));
        }
        const float* restored = tide::own_vec(cluster[0], 1);
        CHECK(restored != nullptr && std::memcmp(restored, v1.data(), sizeof(float) * 2) == 0);
        const auto v4 = own_vector(4);
        CHECK(std::memcmp(tide::own_vec(cluster[2], 4), v4.data(), sizeof(float) * 2) == 0);
        CHECK(check_all(start) == 0);
    }
    // 2. Forward MOVE, committed and accepted: the sources' rows are freed.
    const uint32_t source1 = cluster[0].global_to_local.at(1), source4 = cluster[2].global_to_local.at(4);
    const uint64_t generation1 = tide::migration_slot_generation(cluster[0], source1);
    tide::MembershipMigrationJournal accepted_copy;
    {
        forward.epoch_from = 7; forward.epoch_to = 8;
        std::array<tide::MembershipMigrationPlan, 4> plans;
        std::array<tide::MembershipMigrationJournal, 4> journals;
        CHECK(fragments_prepare(forward, plans, journals) == 0);
        std::array<std::vector<uint64_t>, 4> epochs;
        for (uint32_t rank = 0; rank < 4; ++rank) {
            std::string error;
            CHECK(tide::commit_membership_migration(cluster[rank], plans[rank].moves, journals[rank], error));
        }
        accepted_copy = journals[0];
        for (uint32_t rank = 0; rank < 4; ++rank) {
            std::string error;
            CHECK(tide::accept_membership_migration_history(cluster[rank], plans[rank], journals[rank],
                                                              epochs[rank], error));
            CHECK(plans[rank].moves.empty() && journals[rank].entries.empty());
        }
    }
    auto moved = start;
    moved[1] = 2; moved[4] = 3;
    CHECK(check_all(moved) == 0);
    CHECK(cluster[0].free_slot_pool == std::vector<uint32_t>{source1} &&
          cluster[2].free_slot_pool == std::vector<uint32_t>{source4});
    CHECK(cluster[0].adj.at(source1).capacity() == 0 && cluster[0].local_to_global.at(source1) == UINT32_MAX &&
          tide::migration_slot_generation(cluster[0], source1) == generation1 + 1);
    CHECK(cluster[0].full_own.vec(source1)[0] == 0.0f && cluster[0].full_own.vec(source1)[1] == 0.0f);
    CHECK(tide::own_vec(cluster[0], 1) == nullptr && tide::own_vec(cluster[2], 4) == nullptr);
    const auto v1 = own_vector(1), v4 = own_vector(4);
    CHECK(std::memcmp(tide::own_vec(cluster[2], 1), v1.data(), sizeof(float) * 2) == 0 &&
          std::memcmp(tide::own_vec(cluster[3], 4), v4.data(), sizeof(float) * 2) == 0);
    // A retained copy of an accepted journal never rolls back; nothing changes.
    {
        std::string error;
        auto copy = accepted_copy;
        CHECK(!tide::rollback_membership_migration(cluster[0], moves(), copy, error) &&
              error.find("revoked") != std::string::npos);
        CHECK(cluster[0].free_slot_pool == std::vector<uint32_t>{source1} && tide::mem_at(cluster[0], 1) == 2);
    }
    // 3. The next PREPARE onto rank 0 reuses the pooled slot: node 3 (rank 1 -> 0).
    {
        tide::MembershipMigrationPlan next;
        next.n_servers = 4; next.epoch_from = 8; next.epoch_to = 9;
        tide::MembershipMove move;
        move.global_id = 3; move.from_partition = 1; move.to_partition = 0;
        move.adjacency_digest = tide::migration_row_digest({});
        next.moves.push_back(move);
        const uint32_t n_local_before = tide::n_local_load(cluster[0]);
        std::array<tide::MembershipMigrationPlan, 4> plans;
        std::array<tide::MembershipMigrationJournal, 4> journals;
        CHECK(fragments_prepare(next, plans, journals) == 0);
        CHECK(journals[0].entries[0].target_reused && journals[0].entries[0].target_local_idx == source1);
        CHECK(cluster[0].free_slot_pool.empty() && tide::n_local_load(cluster[0]) == n_local_before);
        const auto v3 = own_vector(3);
        CHECK(cluster[0].global_to_local.at(3) == source1 &&
              std::memcmp(cluster[0].full_own.vec(source1), v3.data(), sizeof(float) * 2) == 0);
        std::array<std::vector<uint64_t>, 4> epochs;
        for (uint32_t rank = 0; rank < 4; ++rank) {
            std::string error;
            CHECK(tide::commit_membership_migration(cluster[rank], plans[rank].moves, journals[rank], error));
        }
        for (uint32_t rank = 0; rank < 4; ++rank) {
            std::string error;
            CHECK(tide::accept_membership_migration_history(cluster[rank], plans[rank], journals[rank],
                                                              epochs[rank], error));
        }
        moved[3] = 0;
        CHECK(check_all(moved) == 0);
        CHECK(cluster[1].free_slot_pool.size() == 1);  // node 3's old slot on rank 1
    }
    // membership_check sees a broken state: a label, a mapping, a vector.
    {
        auto& server = cluster[1];
        tide::mem_set(server, 5, 0);
        auto c = tide::membership_check(server, moved.data(), NODES);
        CHECK(c.membership_mismatches == 1 && c.first_mismatch == 5);
        tide::mem_set(server, 5, moved[5]);
        auto wrong_owner = moved;
        wrong_owner[2] = 3;  // the coordinator says rank 3, rank 1 holds the row and vector
        c = tide::membership_check(server, wrong_owner.data(), NODES);
        CHECK(c.membership_mismatches == 1 && c.mapping_mismatches == 1 && c.vector_mismatches == 1);
        server.adj.at(server.free_slot_pool.front()) = {1};
        c = tide::membership_check(server, moved.data(), NODES);
        CHECK(!c.pool_ok);
        // One flipped bit of an own vector changes the digest.
        const uint64_t digest = c.vector_digest;
        float* held = const_cast<float*>(tide::own_vec(server, 2));
        uint32_t bits = 0;
        std::memcpy(&bits, held, sizeof(bits));
        bits ^= 1u;
        std::memcpy(held, &bits, sizeof(bits));
        CHECK(tide::membership_check(server, moved.data(), NODES).vector_digest != digest);
    }
    return 0;
}

// A plan carrying vector digests (wire version 3 or 4) whose vector is not the
// original: a v3 file built from a wrong vector, internally consistent (its
// digest is that of the wrong floats). The target cannot tell; the source
// compares the digest with its own vector at PREPARE and refuses the plan
// before any change, so it keeps its vector (the coordinator then rolls back
// the servers that prepared). The REFRESHed plan (digests from the sources)
// passes on every server, and a plan without digests (v2) is not compared.
int test_source_refuses_plan_with_wrong_vector() {
    Cluster cluster;
    initialize(cluster);
    make_own_only(cluster);
    tide::MembershipMigrationPlan plan;
    plan.n_servers = 4; plan.epoch_from = 7; plan.epoch_to = 8;
    plan.moves = moves();  // node 1: 0 -> 2, node 4: 2 -> 3
    std::string error;
    std::vector<char> bytes;
    tide::MembershipMigrationPlan parsed;
    CHECK(tide::serialize_membership_migration_plan(plan, bytes, error));
    CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, parsed, error) &&
          parsed.vector_dim == 0);
    CHECK(refresh(cluster, plan) == 0);
    auto wrong = plan;
    wrong.moves[0].vector[0] += 1.0f;  // node 1: not its original vector
    wrong.moves[0].vector_digest = tide::migration_vector_digest(wrong.moves[0].vector);
    CHECK(tide::serialize_membership_migration_plan(wrong, bytes, error));
    CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, parsed, error) &&
          parsed.vector_dim == 2 && parsed.moves[0].vector == wrong.moves[0].vector);
    // The v3 plan as the coordinator sends it: one v4 fragment per server.
    std::array<tide::MembershipMigrationJournal, 4> journals;
    std::array<bool, 4> prepared{};
    for (uint32_t rank = 0; rank < 4; ++rank) {
        tide::MembershipMigrationPlan fragment;
        CHECK(tide::serialize_membership_migration_fragment(parsed, rank, bytes, error));
        CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, fragment, error) &&
              fragment.vector_dim == 2);
        prepared[rank] = tide::prepare_membership_migration(cluster[rank], fragment.moves, 7, 8, journals[rank],
                                                              error, fragment.vector_dim > 0);
        if (rank == 0) CHECK(!prepared[rank] && error.find("source vector differs") != std::string::npos);
        else CHECK(prepared[rank]);
    }
    // The source refused before any change and keeps its vector, row and mapping.
    const float* kept = tide::own_vec(cluster[0], 1);
    CHECK(kept != nullptr && std::vector<float>(kept, kept + 2) == own_vector(1));
    CHECK(!journals[0].prepared && tide::mem_at(cluster[0], 1) == 0 &&
          tide::n_local_load(cluster[0]) == 2 && cluster[0].global_to_local.at(1) == 1);
    // The coordinator's rollback undoes the servers that prepared: the target
    // drops the wrong vector it installed.
    for (uint32_t rank = 1; rank < 4; ++rank)
        CHECK(tide::rollback_membership_migration(cluster[rank], parsed.moves, journals[rank], error));
    CHECK(cluster[2].global_to_local.at(1) == UINT32_MAX && tide::own_vec(cluster[2], 1) == nullptr);
    // The REFRESHed plan (digests of the sources' own vectors) passes everywhere.
    for (uint32_t rank = 0; rank < 4; ++rank) {
        tide::MembershipMigrationJournal journal;
        tide::MembershipMigrationPlan fragment;
        CHECK(tide::serialize_membership_migration_fragment(plan, rank, bytes, error));
        CHECK(tide::load_membership_migration_plan_bytes(bytes.data(), bytes.size(), NODES, 4, fragment, error));
        CHECK(tide::prepare_membership_migration(cluster[rank], fragment.moves, 7, 8, journal, error, true));
        CHECK(tide::rollback_membership_migration(cluster[rank], fragment.moves, journal, error));
    }
    // A v4 fragment as the source receives it (node 1's digest, no floats)
    // naming the digest of another node's vector is refused at the source.
    auto swapped = plan;
    swapped.moves[0].vector.clear();
    swapped.moves[0].vector_digest = tide::migration_vector_digest(own_vector(0));
    tide::MembershipMigrationJournal journal;
    CHECK(!tide::prepare_membership_migration(cluster[0], swapped.moves, 7, 8, journal, error, true) &&
          error.find("source vector differs") != std::string::npos);
    return 0;
}

// A MOVE of an inserted id whose global_to_local chunk a
// bystander never allocated. Id 100,005 lies in mapping chunk 1 (ids 100,000 and
// up); its owner (rank 0) holds its row and, after PREPARE, its target (rank 2)
// allocates the chunk; ranks 1 and 3 know the id (its membership label and
// n_base, as its broadcast left them) but hold no mapping chunk for it.
// PREPARE, COMMIT and ACCEPT of the MOVE and of its inverse run on every
// server without indexing an absent chunk; the bystanders still have none, and
// membership-check on every server is consistent after each MOVE.
int test_inserted_id_move_on_bystander_without_chunk() {
    Cluster cluster;
    initialize(cluster);
    constexpr uint32_t kInserted = 100005;
    for (auto& data : cluster) {
        tide::server_ensure_mem_capacity(data, kInserted);
        tide::mem_set(data, kInserted, 0);
        tide::n_base_set(data, kInserted + 1);
    }
    auto& home = cluster[0];
    const uint32_t slot = tide::n_local_load(home);
    home.global_to_local.ensure_capacity_for(kInserted, UINT32_MAX);
    home.adj.ensure_capacity_for(slot);
    home.local_to_global.ensure_capacity_for(slot, UINT32_MAX);
    home.adj.at(slot) = {1, 2};
    home.local_to_global.at(slot) = kInserted;
    home.global_to_local.at(kInserted) = slot;
    tide::n_local_set(home, slot + 1);
    tide::ensure_node_locks(home, slot + 1);
    for (uint32_t rank : {1u, 2u, 3u})
        CHECK(cluster[rank].global_to_local.segment(1) == nullptr &&
              tide::g2l_checked(cluster[rank], kInserted) == UINT32_MAX);
    std::vector<uint32_t> owners(kInserted + 1, UINT32_MAX);
    const auto base_owners = membership();
    std::copy(base_owners.begin(), base_owners.end(), owners.begin());
    auto move_once = [&](uint32_t from, uint32_t to, uint64_t epoch) -> int {
        tide::MembershipMigrationPlan plan;
        plan.n_servers = 4; plan.epoch_from = epoch; plan.epoch_to = epoch + 1;
        tide::MembershipMove move{};
        move.global_id = kInserted; move.from_partition = from; move.to_partition = to;
        move.adjacency = {1, 2}; move.adjacency_digest = tide::migration_row_digest(move.adjacency);
        plan.moves.push_back(move);
        std::array<tide::MembershipMigrationJournal, 4> journals;
        std::array<std::vector<uint64_t>, 4> accepted;
        std::string error;
        for (uint32_t rank = 0; rank < 4; ++rank)
            CHECK(tide::prepare_membership_migration(cluster[rank], plan.moves, epoch, epoch + 1, journals[rank],
                                                       error));
        for (uint32_t rank = 0; rank < 4; ++rank)
            CHECK(tide::commit_membership_migration(cluster[rank], plan.moves, journals[rank], error));
        for (uint32_t rank = 0; rank < 4; ++rank) {
            auto copy = plan;
            CHECK(tide::accept_membership_migration_history(cluster[rank], copy, journals[rank], accepted[rank],
                                                              error));
        }
        owners[kInserted] = to;
        for (uint32_t rank = 0; rank < 4; ++rank) {
            const auto counts = tide::membership_check(cluster[rank], owners.data(), kInserted + 1);
            CHECK(counts.pool_ok && counts.membership_mismatches == 0 && counts.mapping_mismatches == 0);
            CHECK(tide::mem_at(cluster[rank], kInserted) == to);
        }
        const uint32_t target_slot = tide::g2l_checked(cluster[to], kInserted);
        CHECK(target_slot != UINT32_MAX && cluster[to].adj.at(target_slot) == std::vector<uint32_t>({1, 2}));
        return 0;
    };
    CHECK(move_once(0, 2, 0) == 0);
    for (uint32_t rank : {1u, 3u}) CHECK(cluster[rank].global_to_local.segment(1) == nullptr);
    CHECK(tide::g2l_checked(cluster[0], kInserted) == UINT32_MAX && home.free_slot_pool.size() == 1);
    CHECK(move_once(2, 0, 1) == 0);
    for (uint32_t rank : {1u, 3u}) CHECK(cluster[rank].global_to_local.segment(1) == nullptr);
    CHECK(tide::g2l_checked(home, kInserted) == slot && home.free_slot_pool.empty());
    return 0;
}

int main() {
    if (test_plan_round_trip_and_trailing_rejection() != 0) return 1;
    if (test_plan_mode_refuses_copy()) return 1;
    if (test_plan_round_trip_k16() != 0) return 1;
    if (test_plan_v3_vectors() != 0) return 1;
    if (test_own_only_move_ships_vector() != 0) return 1;
    if (test_own_only_inverse_move_and_accept() != 0) return 1;
    if (test_plan_v4_per_target_fragments() != 0) return 1;
    if (test_prepare_refuses_vector_dimension_mismatch() != 0) return 1;
    if (test_per_target_move_accept_frees_and_reuses() != 0) return 1;
    if (test_prepare_commit_and_rollback() != 0) return 1;
    if (test_accept_revokes_rollback() != 0) return 1;
    if (test_accept_history_keeps_epochs_only() != 0) return 1;
    if (test_partial_accept_completes_by_retry() != 0) return 1;
    if (test_inverse_move_restores_state() != 0) return 1;
    if (test_reset_clears_free_slot_pool() != 0) return 1;
    if (test_free_slot_pool_reuse_and_rollback_return() != 0) return 1;
    if (test_commit_requires_target_prepare() != 0) return 1;
    if (test_bad_payload_fails_before_mutation() != 0) return 1;
    if (test_target_capacity_failure_fails_before_mapping_publication() != 0)
        return 1;
    if (test_refresh_uses_current_post_maintenance_row() != 0) return 1;
    if (test_current_post_maintenance_row_survives_commit_and_rollback() != 0)
        return 1;
    if (test_interleaved_insert_survives_rollback() != 0) return 1;
    if (test_source_refuses_plan_with_wrong_vector() != 0) return 1;
    if (test_inserted_id_move_on_bystander_without_chunk() != 0) return 1;
    std::cout << "membership migration tests passed" << std::endl;
    return 0;
}
