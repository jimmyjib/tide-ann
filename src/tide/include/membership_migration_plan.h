#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace tide {

// One ownership move with its row (global ids) and, in PQ mode, the node's original vector.
struct MembershipMove {
    uint32_t global_id = UINT32_MAX;
    uint32_t from_partition = UINT32_MAX;
    uint32_t to_partition = UINT32_MAX;
    std::vector<uint32_t> adjacency;
    uint64_t adjacency_digest = 0;
    std::vector<float> vector;
    uint64_t vector_digest = 0;
};

// the only plan mode: an ownership MOVE
inline constexpr uint32_t kMigrationPlanModeMove = 0;

// version 4: the server whose moves carry their vectors
inline constexpr uint32_t kMigrationNoVectorTarget = UINT32_MAX;

struct MembershipMigrationPlan {
    uint32_t n_servers = 0;
    uint32_t mode = kMigrationPlanModeMove;
    uint64_t epoch_from = 0;
    uint64_t epoch_to = 0;
    // version 4: the server whose moves carry vectors, otherwise kMigrationNoVectorTarget
    uint32_t vector_target = kMigrationNoVectorTarget;
    // vector dimension of a version 3 or 4 plan (every move has a vector digest), else 0
    uint32_t vector_dim = 0;
    std::vector<MembershipMove> moves;
};

uint64_t migration_row_digest(const std::vector<uint32_t>& adjacency) noexcept;
// FNV-1a 64 over the length (u32) and the float bytes of a vector
uint64_t migration_vector_digest(const std::vector<float>& vector) noexcept;
// the same over `dim` floats in place
uint64_t migration_vector_digest(const float* vector, size_t dim) noexcept;
inline constexpr uint32_t kMigrationMaxVectorDim = 65536;

// FNV-1a 64 over a node id and its vector bits, summed per server by membership-check
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

// Plan file: version 2, or 3 with every move's vector. Version 4 is the per-server PREPARE message.
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

// PREPARE message (version 4): the version 3 header with a u32 target server after the dimension, then every
// move's row and digests, with floats only for moves to that target.
bool serialize_membership_migration_fragment(
    const MembershipMigrationPlan& plan,
    uint32_t target,
    std::vector<char>& bytes,
    std::string& error,
    uint64_t* vector_bytes = nullptr);

// Drops rows and vectors and keeps the digests. A stripped plan is never sent again.
void strip_membership_migration_payload(MembershipMigrationPlan& plan) noexcept;

uint64_t membership_migration_vector_bytes(const MembershipMigrationPlan& plan) noexcept;

}  // namespace tide
