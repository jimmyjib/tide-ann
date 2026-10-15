#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tide {

// Coordinator-side membership migration lifecycle. PREPARE installs shadow
// rows while retaining the old membership. COMMIT publishes the new
// membership only after the caller has fenced admission and drained search.
// ACCEPT makes the commit final (the plan can no longer roll back); a later
// change of mind is a new forward plan (the inverse MOVE).
//
// PREPARE is per target: every server receives the whole move list (ids,
// owners, rows and their digests, every vector digest), so bystanders check
// mem_at == from and update membership and every ACK names the global move
// count; each moved node's original vector (own-only storage: dim x 4 bytes)
// goes only to its target server (plan wire version 4). The coordinator keeps
// no payload after the PREPARE fanout.
//
// The two path entries are the admin command's (migration-apply [--refresh]):
// both are refused while a controller phase is active. A
// manual MOVE between two search commands of a phase would move the ownership
// epoch away from the controller's history, so every later plan would be
// refused and a burst awaiting its verdict could neither accept nor revert.
bool coordinator_membership_migration_prepare(const std::string& path);
// PREPARE after a refresh exchange replaced the plan's rows with the source
// servers' current rows, and under own-only storage added each moved node's
// original vector from its source (the controller's skeleton plans carry none).
bool coordinator_membership_migration_prepare_current(const std::string& path);
// The controller's own fence: the same refreshed PREPARE of a skeleton plan
// the controller holds in memory (its forward burst or the inverse MOVE of a
// rejected one). Holding the plan in memory means a burst can always be
// reverted, even when the window's dump directory cannot be written; the plan
// files are records only.
bool coordinator_membership_migration_prepare_controller(const std::vector<char>& plan_bytes);
bool coordinator_membership_migration_commit();
bool coordinator_membership_migration_rollback();
bool coordinator_membership_migration_accept();
bool coordinator_membership_migration_is_prepared();
uint64_t coordinator_membership_ownership_epoch();

// Fence payload of a migration: the bytes that pass through the coordinator.
// Under own-only storage each moved node's vector crosses it twice (in its
// source's REFRESH reply and in its target's PREPARE fragment).
struct MigrationPayloadStats {
    uint64_t moves = 0;
    uint64_t refresh_request_bytes = 0;   // the move template, summed over servers
    uint64_t refresh_response_bytes = 0;  // the sources' rows (and vectors)
    uint64_t prepare_request_bytes = 0;   // the PREPARE fragments, summed over servers
    uint64_t vector_bytes = 0;            // vector floats in the plan (moves x dim x 4)
    uint64_t prepare_vector_bytes = 0;    // vector floats sent in PREPARE, summed over servers
    // Per server (index = rank): the PREPARE request bytes (type byte + fragment)
    // and the vector float bytes the serializer wrote into that server's
    // fragment (counted by serialize_membership_migration_fragment, not derived
    // from the move count). prepare_request_bytes and prepare_vector_bytes are
    // their sums.
    std::vector<uint64_t> prepare_request_bytes_by_server;
    std::vector<uint64_t> prepare_vector_bytes_by_server;
    double refresh_deadline_s = 0.0;      // the deadlines applied
    double exchange_deadline_s = 0.0;     // PREPARE, COMMIT, ROLLBACK and each ACCEPT attempt
    double refresh_s = 0.0;               // wall time of the refresh exchange
    double prepare_s = 0.0;               // wall time of the PREPARE fanout
    uint64_t total_bytes() const {
        return refresh_request_bytes + refresh_response_bytes + prepare_request_bytes;
    }
};
// The payload of the active migration, or of the last one after it resolved.
const MigrationPayloadStats& coordinator_membership_migration_payload();

// ,"prepare_request_bytes_by_server":[...],"prepare_vector_bytes_by_server":[...]
// (the migration-apply reply and the controller journal's fence fields).
inline std::string migration_payload_by_server_fields(const MigrationPayloadStats& p) {
    std::string out;
    for (int list = 0; list < 2; ++list) {
        const auto& values = list == 0 ? p.prepare_request_bytes_by_server : p.prepare_vector_bytes_by_server;
        out += list == 0 ? ",\"prepare_request_bytes_by_server\":[" : ",\"prepare_vector_bytes_by_server\":[";
        for (size_t i = 0; i < values.size(); ++i) out += (i ? "," : "") + std::to_string(values[i]);
        out += ']';
    }
    return out;
}

// Base deadline of one migration fanout (PREPARE, COMMIT, ROLLBACK, one ACCEPT
// attempt) in seconds; 30 s. Tests shorten it to drive the timeout path.
extern double g_migration_exchange_deadline_sec;
// Base deadline of the refresh exchange; 60 s.
extern double g_migration_refresh_deadline_sec;
// The slowest payload throughput the deadlines allow for (bytes per second;
// 8 MiB/s): each exchange of a plan gets its base deadline plus the plan's
// payload bytes divided by this rate, so the fence is bounded by its payload.
extern double g_migration_payload_bytes_per_sec;
// base_sec + payload_bytes / g_migration_payload_bytes_per_sec.
double migration_deadline_sec(double base_sec, uint64_t payload_bytes);

}  // namespace tide
