#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace tide {

// One same-graph ownership move. Adjacency entries remain global node IDs.
// `vector` is the moved node's original vector (empty when the plan carries
// none): under own-only storage the source holds it and the target installs it
// before it publishes the row, so the target scores and re-ranks the node
// exactly. Wire version 3 carries it with its digest; versions 1 and 2 never do.
struct MembershipMove {
    uint32_t global_id = UINT32_MAX;
    uint32_t from_partition = UINT32_MAX;
    uint32_t to_partition = UINT32_MAX;
    std::vector<uint32_t> adjacency;
    uint64_t adjacency_digest = 0;
    std::vector<float> vector;
    uint64_t vector_digest = 0;
};

// The wire's plan mode: 0, an ownership MOVE, is the only mode; the loader
// refuses any other value.
inline constexpr uint32_t kMigrationPlanModeMove = 0;

// Wire version 4 names the one server whose moves carry their vectors.
inline constexpr uint32_t kMigrationNoVectorTarget = UINT32_MAX;

struct MembershipMigrationPlan {
    uint32_t n_servers = 0;
    uint32_t mode = kMigrationPlanModeMove;
    uint64_t epoch_from = 0;
    uint64_t epoch_to = 0;
    // A per-target PREPARE fragment (wire version 4): every move carries its
    // vector digest, and only the moves whose target is this server carry the
    // vector. kMigrationNoVectorTarget for versions 1-3.
    uint32_t vector_target = kMigrationNoVectorTarget;
    // The loader's header field: the vector dimension of a version-3 or -4 plan,
    // whose every move carries its vector digest (a source compares it with its
    // own vector at PREPARE); 0 for versions 1 and 2 (no digest). Kept by
    // strip_membership_migration_payload.
    uint32_t vector_dim = 0;
    std::vector<MembershipMove> moves;
};

uint64_t migration_row_digest(const std::vector<uint32_t>& adjacency) noexcept;
// FNV-1a 64 over the length (u32) and the float bytes of a shipped vector.
uint64_t migration_vector_digest(const std::vector<float>& vector) noexcept;
// The same digest of `dim` floats in place (a server's own vector).
uint64_t migration_vector_digest(const float* vector, size_t dim) noexcept;
// Largest vector dimension a plan may carry.
inline constexpr uint32_t kMigrationMaxVectorDim = 65536;

// FNV-1a 64 over a node id (u32 little-endian) and the bits of its vector: the
// per-node term of membership-check's vector digest (summed mod 2^64 over the
// nodes a server owns, so the coordinator can compare each server's own vectors
// with the original vectors byte for byte, in any order).
inline uint64_t node_vector_hash(uint32_t id, const float* vector, uint32_t dim) noexcept {
    uint64_t value = 1469598103934665603ULL;
    auto mix = [&value](uint32_t word) {
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            value ^= static_cast<uint8_t>(word >> shift);
            value *= 1099511628211ULL;
        }
    };
    mix(id);
    for (uint32_t d = 0; d < dim; ++d) {
        uint32_t bits = 0;
        std::memcpy(&bits, vector + d, sizeof(bits));
        mix(bits);
    }
    return value;
}

// Portable little-endian plan. The file contains only same-graph ownership
// moves and the exact outgoing adjacency row for each moved global node; from
// version 3 also each moved node's vector (header field vector_dim, every move
// the same length). The serializer writes version 2 when no move carries a
// vector and version 3 when every move does; a mix is refused. Version 4 is
// the per-target PREPARE fragment (serialize_membership_migration_fragment).
bool load_membership_migration_plan(
    const std::string& path,
    uint32_t expected_n_base,
    uint32_t expected_n_servers,
    MembershipMigrationPlan& plan,
    std::string& error);

bool load_membership_migration_plan_bytes(
    const char* bytes,
    size_t size,
    uint32_t expected_n_base,
    uint32_t expected_n_servers,
    MembershipMigrationPlan& plan,
    std::string& error);

bool serialize_membership_migration_plan(
    const MembershipMigrationPlan& plan,
    std::vector<char>& bytes,
    std::string& error);

// Wire version 4: the PREPARE fragment of server `target`. Every server gets the
// same plan identity (epochs, the whole move list with ids, owners, rows and
// row digests, and every move's vector digest, so bystanders check and update
// membership and every ACK names the global move count); the vector floats go
// only with the moves whose target is `target`, so each original vector crosses
// to exactly one server. Every move of `plan` must carry a vector of one length
// with its digest. The loader enforces the same rule: a version-4 move carries
// its vector exactly when its target is the fragment's target, and the floats
// must match the digest the source computed at REFRESH. `vector_bytes`, when
// given, receives the float bytes the serializer actually wrote into this
// fragment (the coordinator reports them per server, so a test can check that
// each vector went to its target only).
bool serialize_membership_migration_fragment(
    const MembershipMigrationPlan& plan,
    uint32_t target,
    std::vector<char>& bytes,
    std::string& error,
    uint64_t* vector_bytes = nullptr);

// Drops the rows and the vectors of a prepared plan and keeps every digest:
// COMMIT, ROLLBACK and ACCEPT read only epochs, ids, owners and digests, so a
// prepared plan holds no payload while it waits for its verdict. A stripped
// plan is never serialized or prepared again (its rows no longer match their
// digests, and the serializer and PREPARE refuse it).
void strip_membership_migration_payload(MembershipMigrationPlan& plan) noexcept;

// Bytes of the vector floats a plan carries (every move's vector).
uint64_t membership_migration_vector_bytes(const MembershipMigrationPlan& plan) noexcept;

}  // namespace tide
