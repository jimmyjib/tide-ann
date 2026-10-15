#include "membership_migration.h"
#include "membership_migration_storage.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace tide {

using membership_migration_detail::journal_matches;

MembershipMigrationJournal g_membership_migration_journal;

namespace {

bool fail(std::string& error, const std::string& message) {
    error = message;
    return false;
}

std::string node_error(uint32_t global_id, const char* message) {
    std::ostringstream out;
    out << "migration node " << global_id << ": " << message;
    return out.str();
}

bool read_exact(std::istream& input, char* data, std::streamsize size) {
    input.read(data, size);
    return input.gcount() == size;
}

bool read_u32_le(std::istream& input, uint32_t& value) {
    uint8_t bytes[4]{};
    if (!read_exact(input, reinterpret_cast<char*>(bytes), sizeof(bytes))) {
        return false;
    }
    value = static_cast<uint32_t>(bytes[0]) |
            (static_cast<uint32_t>(bytes[1]) << 8) |
            (static_cast<uint32_t>(bytes[2]) << 16) |
            (static_cast<uint32_t>(bytes[3]) << 24);
    return true;
}

bool read_u64_le(std::istream& input, uint64_t& value) {
    uint8_t bytes[8]{};
    if (!read_exact(input, reinterpret_cast<char*>(bytes), sizeof(bytes))) {
        return false;
    }
    value = 0;
    for (uint32_t index = 0; index < 8; ++index) {
        value |= static_cast<uint64_t>(bytes[index]) << (8 * index);
    }
    return true;
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

bool read_f32_le(std::istream& input, float& value) {
    uint32_t bits = 0;
    if (!read_u32_le(input, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

void write_f32_le(std::ostream& output, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32_le(output, bits);
}

bool parse_membership_migration_plan(
    std::istream& input,
    uint32_t expected_n_base,
    uint32_t expected_n_servers,
    MembershipMigrationPlan& plan,
    std::string& error) {
    char magic[8]{};
    if (!read_exact(input, magic, sizeof(magic)) ||
        std::memcmp(magic, "TIDEMIG\0", sizeof(magic)) != 0) {
        return fail(error, "migration plan magic mismatch");
    }
    // The mode field sits directly after the version (wire version 2), so
    // the version must be read and checked before any later header field.
    uint32_t version = 0;
    if (!read_u32_le(input, version)) {
        return fail(error, "migration plan header is truncated");
    }
    if (version < 1 || version > 4) {
        return fail(error, "unsupported migration plan version");
    }
    uint32_t mode = kMigrationPlanModeMove;
    if (version >= 2 && !read_u32_le(input, mode)) {
        return fail(error, "migration plan mode is truncated");
    }
    if (mode != kMigrationPlanModeMove) {
        return fail(error, "migration plan mode is invalid");
    }
    // Version 3: the dimension of the vector every move carries (> 0).
    // Version 4: the same, then the server whose moves carry their vectors.
    uint32_t vector_dim = 0;
    uint32_t vector_target = kMigrationNoVectorTarget;
    if (version >= 3) {
        if (!read_u32_le(input, vector_dim)) return fail(error, "migration plan vector dimension is truncated");
        if (vector_dim == 0 || vector_dim > kMigrationMaxVectorDim)
            return fail(error, "migration plan vector dimension is invalid");
    }
    if (version == 4 && !read_u32_le(input, vector_target)) {
        return fail(error, "migration plan vector target is truncated");
    }
    uint32_t n_servers = 0;
    uint64_t epoch_from = 0;
    uint64_t epoch_to = 0;
    uint64_t move_count = 0;
    if (!read_u32_le(input, n_servers) ||
        !read_u64_le(input, epoch_from) ||
        !read_u64_le(input, epoch_to) ||
        !read_u64_le(input, move_count)) {
        return fail(error, "migration plan header is truncated");
    }
    if (n_servers != expected_n_servers) {
        return fail(error, "migration plan server-count mismatch");
    }
    if (version == 4 && vector_target >= n_servers) {
        return fail(error, "migration plan vector target is invalid");
    }
    if (epoch_to != epoch_from + 1) {
        return fail(error, "migration plan epochs are not consecutive");
    }
    if (move_count == 0 || move_count > expected_n_base) {
        return fail(error, "migration plan move count is invalid");
    }

    std::unordered_set<uint32_t> seen;
    std::vector<MembershipMove> moves;
    moves.reserve(static_cast<size_t>(move_count));
    for (uint64_t index = 0; index < move_count; ++index) {
        MembershipMove move{};
        uint32_t degree = 0;
        if (!read_u32_le(input, move.global_id) ||
            !read_u32_le(input, move.from_partition) ||
            !read_u32_le(input, move.to_partition) ||
            !read_u32_le(input, degree) ||
            !read_u64_le(input, move.adjacency_digest)) {
            return fail(error, "migration plan move header is truncated");
        }
        if (move.global_id >= expected_n_base ||
            !seen.insert(move.global_id).second) {
            return fail(error, node_error(move.global_id,
                                          "invalid or duplicate global id"));
        }
        if (move.from_partition >= expected_n_servers ||
            move.to_partition >= expected_n_servers ||
            move.from_partition == move.to_partition) {
            return fail(error, node_error(move.global_id,
                                          "invalid owner transition"));
        }
        if (degree > expected_n_base) {
            return fail(error, node_error(move.global_id,
                                          "invalid adjacency degree"));
        }
        move.adjacency.resize(degree);
        for (uint32_t& neighbor : move.adjacency) {
            if (!read_u32_le(input, neighbor) || neighbor >= expected_n_base) {
                return fail(error, node_error(move.global_id,
                                              "invalid adjacency payload"));
            }
        }
        if (migration_row_digest(move.adjacency) != move.adjacency_digest) {
            return fail(error, node_error(move.global_id,
                                          "adjacency digest mismatch"));
        }
        if (vector_dim > 0) {
            if (!read_u64_le(input, move.vector_digest)) {
                return fail(error, node_error(move.global_id, "vector header is truncated"));
            }
            // Version 4 ships the floats only to the move's target.
            if (version == 4 && move.to_partition != vector_target) {
                moves.push_back(std::move(move));
                continue;
            }
            move.vector.resize(vector_dim);
            for (float& value : move.vector) {
                if (!read_f32_le(input, value)) {
                    return fail(error, node_error(move.global_id, "vector payload is truncated"));
                }
            }
            if (migration_vector_digest(move.vector) != move.vector_digest) {
                return fail(error, node_error(move.global_id, "vector digest mismatch"));
            }
        }
        moves.push_back(std::move(move));
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        return fail(error, "migration plan has trailing bytes");
    }
    if (!input.eof()) return fail(error, "migration plan read failed");
    plan.n_servers = n_servers;
    plan.mode = mode;
    plan.epoch_from = epoch_from;
    plan.epoch_to = epoch_to;
    plan.vector_target = vector_target;
    plan.vector_dim = vector_dim;
    plan.moves = std::move(moves);
    return true;
}

// The shared part of every serialized plan: header (version, mode, the vector
// dimension from version 3, the vector target in version 4, servers, epochs,
// move count) and per move its identity, row and row digest, and from version 3
// its vector digest, then its floats when `with_floats(move)`. `float_bytes`
// (when given) receives the bytes of the floats written.
template <class WithFloats>
bool write_membership_migration_plan(const MembershipMigrationPlan& plan, uint32_t version,
                                     uint32_t vector_dim, uint32_t vector_target, WithFloats&& with_floats,
                                     std::vector<char>& bytes, std::string& error,
                                     uint64_t* float_bytes = nullptr) {
    uint64_t floats_written = 0;
    std::ostringstream output(std::ios::binary | std::ios::out);
    output.write("TIDEMIG\0", 8);
    write_u32_le(output, version);
    write_u32_le(output, plan.mode);
    if (version >= 3) write_u32_le(output, vector_dim);
    if (version == 4) write_u32_le(output, vector_target);
    write_u32_le(output, plan.n_servers);
    write_u64_le(output, plan.epoch_from);
    write_u64_le(output, plan.epoch_to);
    write_u64_le(output, plan.moves.size());
    std::unordered_set<uint32_t> seen;
    for (const auto& move : plan.moves) {
        if (!seen.insert(move.global_id).second ||
            move.from_partition >= plan.n_servers ||
            move.to_partition >= plan.n_servers ||
            move.from_partition == move.to_partition ||
            move.adjacency.size() > std::numeric_limits<uint32_t>::max() ||
            migration_row_digest(move.adjacency) != move.adjacency_digest) {
            return fail(error, node_error(move.global_id,
                                          "invalid serialization input"));
        }
        write_u32_le(output, move.global_id);
        write_u32_le(output, move.from_partition);
        write_u32_le(output, move.to_partition);
        write_u32_le(output,
                     static_cast<uint32_t>(move.adjacency.size()));
        write_u64_le(output, move.adjacency_digest);
        for (uint32_t neighbor : move.adjacency) {
            write_u32_le(output, neighbor);
        }
        if (version >= 3) {
            write_u64_le(output, move.vector_digest);
            if (with_floats(move)) {
                for (float value : move.vector) write_f32_le(output, value);
                floats_written += static_cast<uint64_t>(move.vector.size()) * sizeof(float);
            }
        }
    }
    const std::string encoded = output.str();
    if (!output.good() || encoded.empty()) {
        return fail(error, "migration plan serialization failed");
    }
    bytes.assign(encoded.begin(), encoded.end());
    if (float_bytes) *float_bytes = floats_written;
    return true;
}

}  // namespace

uint64_t migration_row_digest(
    const std::vector<uint32_t>& adjacency) noexcept {
    // FNV-1a over the degree and little-endian uint32 payload. This is an
    // integrity checksum for the migration contract, not a cryptographic ID.
    uint64_t value = 1469598103934665603ULL;
    auto mix = [&](uint32_t word) {
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            value ^= static_cast<uint8_t>(word >> shift);
            value *= 1099511628211ULL;
        }
    };
    mix(static_cast<uint32_t>(adjacency.size()));
    for (uint32_t neighbor : adjacency) mix(neighbor);
    return value;
}

uint64_t migration_vector_digest(const float* vector, size_t dim) noexcept {
    uint64_t value = 1469598103934665603ULL;
    auto mix = [&](uint32_t word) {
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            value ^= static_cast<uint8_t>(word >> shift);
            value *= 1099511628211ULL;
        }
    };
    mix(static_cast<uint32_t>(dim));
    for (size_t d = 0; d < dim; ++d) {
        uint32_t bits = 0;
        std::memcpy(&bits, vector + d, sizeof(bits));
        mix(bits);
    }
    return value;
}

uint64_t migration_vector_digest(const std::vector<float>& vector) noexcept {
    return migration_vector_digest(vector.data(), vector.size());
}

bool load_membership_migration_plan(
    const std::string& path,
    uint32_t expected_n_base,
    uint32_t expected_n_servers,
    MembershipMigrationPlan& plan,
    std::string& error) {
    error.clear();
    plan = MembershipMigrationPlan{};
    std::ifstream input(path, std::ios::binary);
    if (!input) return fail(error, "cannot open migration plan: " + path);
    return parse_membership_migration_plan(
        input, expected_n_base, expected_n_servers, plan, error);
}

bool load_membership_migration_plan_bytes(
    const char* bytes, size_t size, uint32_t expected_n_base,
    uint32_t expected_n_servers, MembershipMigrationPlan& plan,
    std::string& error) {
    error.clear();
    plan = MembershipMigrationPlan{};
    if (bytes == nullptr || size == 0) {
        return fail(error, "migration plan bytes are empty");
    }
    std::istringstream input(
        std::string(bytes, size), std::ios::binary | std::ios::in);
    return parse_membership_migration_plan(
        input, expected_n_base, expected_n_servers, plan, error);
}

bool serialize_membership_migration_plan(
    const MembershipMigrationPlan& plan, std::vector<char>& bytes,
    std::string& error) {
    error.clear();
    bytes.clear();
    if (plan.n_servers == 0 || plan.epoch_to != plan.epoch_from + 1 ||
        plan.moves.empty()) {
        return fail(error, "migration plan cannot be serialized");
    }
    // Every move carries a vector of one length, or none does.
    const size_t vector_dim = plan.moves.front().vector.size();
    for (const auto& move : plan.moves) {
        if (move.vector.size() != vector_dim) {
            return fail(error, node_error(move.global_id, "moves carry vectors of different lengths"));
        }
        if (vector_dim > 0 && migration_vector_digest(move.vector) != move.vector_digest) {
            return fail(error, node_error(move.global_id, "vector digest mismatch"));
        }
    }
    if (vector_dim > kMigrationMaxVectorDim) {
        return fail(error, "migration plan vector dimension is invalid");
    }
    return write_membership_migration_plan(plan, vector_dim > 0 ? 3 : 2, static_cast<uint32_t>(vector_dim),
                                           kMigrationNoVectorTarget,
                                           [](const MembershipMove&) { return true; }, bytes, error);
}

bool serialize_membership_migration_fragment(
    const MembershipMigrationPlan& plan, uint32_t target, std::vector<char>& bytes,
    std::string& error, uint64_t* vector_bytes) {
    error.clear();
    bytes.clear();
    if (vector_bytes) *vector_bytes = 0;
    if (plan.n_servers == 0 || plan.epoch_to != plan.epoch_from + 1 ||
        plan.moves.empty() || target >= plan.n_servers) {
        return fail(error, "migration fragment cannot be serialized");
    }
    const size_t vector_dim = plan.moves.front().vector.size();
    if (vector_dim == 0 || vector_dim > kMigrationMaxVectorDim) {
        return fail(error, "migration fragment needs a vector on every move");
    }
    for (const auto& move : plan.moves) {
        if (move.vector.size() != vector_dim) {
            return fail(error, node_error(move.global_id, "moves carry vectors of different lengths"));
        }
        if (migration_vector_digest(move.vector) != move.vector_digest) {
            return fail(error, node_error(move.global_id, "vector digest mismatch"));
        }
    }
    return write_membership_migration_plan(
        plan, 4, static_cast<uint32_t>(vector_dim), target,
        [target](const MembershipMove& move) { return move.to_partition == target; }, bytes, error, vector_bytes);
}

void strip_membership_migration_payload(MembershipMigrationPlan& plan) noexcept {
    for (auto& move : plan.moves) {
        std::vector<uint32_t>().swap(move.adjacency);
        std::vector<float>().swap(move.vector);
    }
}

uint64_t membership_migration_vector_bytes(const MembershipMigrationPlan& plan) noexcept {
    uint64_t bytes = 0;
    for (const auto& move : plan.moves) bytes += static_cast<uint64_t>(move.vector.size()) * sizeof(float);
    return bytes;
}

bool snapshot_membership_migration_source_rows(
    const ServerData& data,
    const MembershipMigrationPlan& move_template,
    MembershipMigrationPlan& source_fragment,
    std::string& error) {
    error.clear();
    source_fragment = MembershipMigrationPlan{};
    if (move_template.n_servers != data.n_servers ||
        move_template.epoch_to != move_template.epoch_from + 1 ||
        move_template.moves.empty()) {
        return fail(error, "invalid migration source-row template");
    }
    source_fragment.n_servers = move_template.n_servers;
    source_fragment.epoch_from = move_template.epoch_from;
    source_fragment.epoch_to = move_template.epoch_to;
    const uint32_t n_base = n_base_load(data);
    const uint32_t n_local = n_local_load(data);
    for (const auto& template_move : move_template.moves) {
        if (template_move.global_id >= n_base ||
            template_move.from_partition >= data.n_servers ||
            template_move.to_partition >= data.n_servers ||
            template_move.from_partition == template_move.to_partition ||
            mem_at(data, template_move.global_id) !=
                template_move.from_partition) {
            return fail(error, node_error(template_move.global_id,
                                          "source-row template mismatch"));
        }
        if (template_move.from_partition != data.my_partition) continue;
        // Checked: never index a mapping chunk this server did not allocate.
        const uint32_t local_idx = g2l_checked(data, template_move.global_id);
        if (local_idx == UINT32_MAX || local_idx >= n_local ||
            data.local_to_global.at(local_idx) != template_move.global_id) {
            return fail(error, node_error(template_move.global_id,
                                          "current source mapping invalid"));
        }
        MembershipMove current = template_move;
        current.adjacency = data.adj.at(local_idx);
        current.adjacency_digest = migration_row_digest(current.adjacency);
        current.vector.clear();
        current.vector_digest = 0;
        if (data.own_only) {
            // The target holds no vector of this node: ship the original one.
            const float* vec = own_vec(data, template_move.global_id);
            if (vec == nullptr) {
                return fail(error, node_error(template_move.global_id, "source vector missing"));
            }
            current.vector.assign(vec, vec + data.dim);
            current.vector_digest = migration_vector_digest(current.vector);
        }
        source_fragment.moves.push_back(std::move(current));
    }
    return true;
}

MembershipCheckCounts membership_check(const ServerData& data, const uint32_t* owners, uint32_t ids) {
    MembershipCheckCounts c;
    c.ids = ids;
    c.server_ids = n_base_load(data);
    c.n_local = n_local_load(data);
    c.free_slots = static_cast<uint32_t>(data.free_slot_pool.size());
    using L2G = decltype(data.local_to_global);
    using Delta = decltype(data.membership_delta);
    auto note = [&c](uint32_t id) { if (c.first_mismatch == UINT32_MAX) c.first_mismatch = id; };
    for (uint32_t id = 0; id < ids; ++id) {
        const uint32_t owner = owners[id];
        const bool here = owner == data.my_partition;
        c.owned += here ? 1 : 0;
        // mem_at, checked: an id this server never learned has no delta chunk.
        uint32_t label = UINT32_MAX;
        if (id < c.server_ids) {
            if (id < data.base_static_capacity) {
                label = id < data.membership.size() ? data.membership[id] : UINT32_MAX;
            } else {
                const uint32_t delta = id - data.base_static_capacity;
                const uint32_t* segment = data.membership_delta.segment(delta / Delta::chunk_size);
                label = segment ? segment[delta % Delta::chunk_size] : UINT32_MAX;
            }
        }
        if (label != owner) { ++c.membership_mismatches; note(id); }
        const uint32_t slot = g2l_checked(data, id);
        const bool row = slot != UINT32_MAX && slot < c.n_local && data.local_to_global.segment(slot / L2G::chunk_size) &&
            data.local_to_global.at(slot) == id;
        c.mapped += row ? 1 : 0;
        if (row != here || (slot != UINT32_MAX && !row)) { ++c.mapping_mismatches; note(id); }
        if (data.own_only) {
            const float* vector = own_vec(data, id);
            const bool vector_here = vector != nullptr;
            c.vectors += vector_here ? 1 : 0;
            if (vector_here) c.vector_digest += node_vector_hash(id, vector, data.dim);
            if (vector_here != here) { ++c.vector_mismatches; note(id); }
            if (pq_code(data, id) == nullptr) { ++c.missing_codes; note(id); }
        }
    }
    for (uint32_t slot : data.free_slot_pool) {
        if (slot >= c.n_local || !data.local_to_global.segment(slot / L2G::chunk_size) ||
            data.local_to_global.at(slot) != UINT32_MAX ||
            !data.adj.segment(slot / decltype(data.adj)::chunk_size) || !data.adj.at(slot).empty() ||
            data.adj.at(slot).capacity() != 0)
            c.pool_ok = false;
    }
    return c;
}

bool commit_membership_migration(
    ServerData& data,
    const std::vector<MembershipMove>& moves,
    MembershipMigrationJournal& journal,
    std::string& error) {
    error.clear();
    if (!journal.prepared || journal.committed || journal.rolled_back) {
        return fail(error, "migration journal is not commit-ready");
    }
    if (!journal_matches(moves, journal, error) ||
        !validate_membership_migration_generations(data, journal, error)) return false;

    // Validate target-install-before-membership and source authority for the
    // complete batch before publishing any ownership change.
    for (size_t index = 0; index < moves.size(); ++index) {
        const auto& move = moves[index];
        const auto& entry = journal.entries[index];
        if (mem_at(data, move.global_id) != move.from_partition) {
            return fail(error, node_error(move.global_id,
                                          "membership changed before commit"));
        }
        if (data.my_partition == move.to_partition) {
            if (entry.target_local_idx == UINT32_MAX ||
                data.global_to_local.at(move.global_id) !=
                    entry.target_local_idx ||
                data.local_to_global.at(entry.target_local_idx) !=
                    move.global_id ||
                migration_row_digest(data.adj.at(entry.target_local_idx)) !=
                    move.adjacency_digest) {
                return fail(error, node_error(move.global_id,
                                              "target shadow row missing"));
            }
            if (data.own_only && own_vec(data, move.global_id) == nullptr) {
                return fail(error, node_error(move.global_id, "target vector missing"));
            }
        }
        if (data.my_partition == move.from_partition &&
            (entry.source_local_idx == UINT32_MAX ||
             data.global_to_local.at(move.global_id) !=
                 entry.source_local_idx)) {
            return fail(error, node_error(move.global_id,
                                          "source authority changed"));
        }
    }

    for (const auto& move : moves) {
        mem_set(data, move.global_id, move.to_partition);
    }
    for (const auto& entry : journal.entries) {
        if (data.my_partition != entry.from_partition) continue;
        data.global_to_local.at(entry.global_id) = UINT32_MAX;
        data.local_to_global.at(entry.source_local_idx) = UINT32_MAX;
        // The source adjacency row is deliberately retained for rollback and
        // is not counted as active after both mappings are tombstoned.
    }
    journal.committed = true;
    return true;
}

bool accept_membership_migration(
    const ServerData& data,
    const std::vector<MembershipMove>& moves,
    const MembershipMigrationJournal& journal,
    std::string& error) {
    error.clear();
    if (!journal.prepared || !journal.committed || journal.rolled_back) {
        return fail(error, "migration journal is not accept-ready");
    }
    if (!journal_matches(moves, journal, error) ||
        !validate_membership_migration_generations(data, journal, error)) return false;

    for (size_t index = 0; index < moves.size(); ++index) {
        const auto& move = moves[index];
        const auto& entry = journal.entries[index];
        if (mem_at(data, move.global_id) != move.to_partition) {
            return fail(error, node_error(
                move.global_id, "membership changed before accept"));
        }
        if (data.my_partition == move.to_partition &&
            (entry.target_local_idx == UINT32_MAX ||
             data.global_to_local.at(move.global_id) !=
                 entry.target_local_idx ||
             data.local_to_global.at(entry.target_local_idx) !=
                 move.global_id)) {
            return fail(error, node_error(
                move.global_id, "target authority missing at accept"));
        }
        if (data.my_partition == move.from_partition &&
            (entry.source_local_idx == UINT32_MAX ||
             data.global_to_local.at(move.global_id) != UINT32_MAX ||
             data.local_to_global.at(entry.source_local_idx) != UINT32_MAX)) {
            return fail(error, node_error(
                move.global_id, "source rollback row changed before accept"));
        }
    }
    // From here on the transaction is final: no copy of this journal can roll
    // back, so the source rows are never restored (the server's ACCEPT,
    // accept_membership_migration_history, frees them next).
    journal.lifetime->accepted = true;
    journal.lifetime->rollback_revoked = true;
    return true;
}

bool membership_migration_accept_is_repeat(const MembershipMigrationPlan& active,
    const std::vector<uint64_t>& accepted_epochs, uint64_t ownership_epoch) {
    return active.moves.empty() && ownership_epoch > 0 && !accepted_epochs.empty() &&
           accepted_epochs.back() == ownership_epoch;
}

MigrationRollbackDisposition membership_migration_rollback_disposition(
    const MembershipMigrationPlan& active, const std::vector<uint64_t>& accepted_epochs,
    uint64_t ownership_epoch, uint64_t epoch_from, uint64_t epoch_to) {
    if (!active.moves.empty())
        return active.epoch_from == epoch_from && active.epoch_to == epoch_to
            ? MigrationRollbackDisposition::ROLL_BACK : MigrationRollbackDisposition::REFUSE_MISMATCH;
    if (std::find(accepted_epochs.begin(), accepted_epochs.end(), epoch_to) != accepted_epochs.end())
        return MigrationRollbackDisposition::REFUSE_ACCEPTED;
    return epoch_from == ownership_epoch && epoch_to == epoch_from + 1
        ? MigrationRollbackDisposition::NO_OP : MigrationRollbackDisposition::REFUSE_MISMATCH;
}

bool accept_membership_migration_history(ServerData& data, MembershipMigrationPlan& plan,
    MembershipMigrationJournal& journal, std::vector<uint64_t>& accepted_epochs,
    std::string& error) {
    try {
        accepted_epochs.reserve(accepted_epochs.size() + 1);
    } catch (const std::exception& ex) {
        return fail(error, std::string("ACCEPT history allocation failed: ") + ex.what());
    }
    if (plan.moves.empty() || plan.epoch_from != journal.epoch_from || plan.epoch_to != journal.epoch_to)
        return fail(error, "migration ACCEPT plan/journal mismatch");
    // Everything that can fail comes first: the pool capacity and the slot
    // generations of the source rows this server frees.
    if (!journal_matches(plan.moves, journal, error) ||
        !validate_membership_migration_generations(data, journal, error) ||
        !reserve_membership_source_release(data, journal, error)) return false;
    // Validation, then rollback_revoked (shared by every copy of the journal)
    // before any row is freed: no copy can roll back onto a freed slot.
    if (!accept_membership_migration(data, plan.moves, journal, error)) return false;
    release_accepted_source_rows(data, journal);
    accepted_epochs.push_back(plan.epoch_to);  // capacity reserved above: cannot throw
    // The accepted plan (with every moved row) and its journal are never read
    // again; keeping them would grow memory with every burst.
    plan = MembershipMigrationPlan{};
    journal = MembershipMigrationJournal{};
    return true;
}

}  // namespace tide
