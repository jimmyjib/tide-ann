#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tide {

// Coordinator side of PREPARE / COMMIT / ACCEPT / ROLLBACK.

// migration-apply [--refresh]. Rejected during a controller phase.
bool coordinator_membership_migration_prepare(const std::string& path);
// PREPARE with the source servers' current rows (and vectors in PQ mode).
bool coordinator_membership_migration_prepare_current(const std::string& path);
// The controller's own batches, planned in memory without rows.
bool coordinator_membership_migration_prepare_controller(const std::vector<char>& plan_bytes);
bool coordinator_membership_migration_commit();
bool coordinator_membership_migration_rollback();
bool coordinator_membership_migration_accept();
bool coordinator_membership_migration_is_prepared();
uint64_t coordinator_membership_ownership_epoch();

// Bytes through the coordinator. In PQ mode each vector passes twice (REFRESH reply, PREPARE).
struct MigrationPayloadStats {
    uint64_t moves = 0;
    uint64_t refresh_request_bytes = 0;   // the move template, summed over servers
    uint64_t refresh_response_bytes = 0;  // the sources' rows (and vectors)
    uint64_t prepare_request_bytes = 0;   // the PREPARE fragments, summed over servers
    uint64_t vector_bytes = 0;            // vector floats in the plan (moves x dim x 4)
    uint64_t prepare_vector_bytes = 0;    // vector floats sent in PREPARE, summed over servers
    // per server: PREPARE request bytes and the vector bytes written into its fragment
    std::vector<uint64_t> prepare_request_bytes_by_server;
    std::vector<uint64_t> prepare_vector_bytes_by_server;
    double refresh_deadline_s = 0.0;
    double exchange_deadline_s = 0.0;     // PREPARE, COMMIT, ROLLBACK, each ACCEPT
    double refresh_s = 0.0;               // wall time of the refresh exchange
    double prepare_s = 0.0;               // wall time of the PREPARE fanout
    uint64_t total_bytes() const {
        return refresh_request_bytes + refresh_response_bytes + prepare_request_bytes;
    }
};
// payload of the active migration, or of the last one
const MigrationPayloadStats& coordinator_membership_migration_payload();

// per-server payload JSON fields (migration-apply reply, controller journal)
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

// Base fanout deadline (s). Tests lower it.
extern double g_migration_exchange_deadline_sec;
// base deadline of the refresh exchange (s)
extern double g_migration_refresh_deadline_sec;
// slowest payload rate allowed for (bytes per second): deadline = base + payload / rate
extern double g_migration_payload_bytes_per_sec;
double migration_deadline_sec(double base_sec, uint64_t payload_bytes);

}  // namespace tide
