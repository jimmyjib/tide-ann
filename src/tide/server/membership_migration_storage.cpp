#include "membership_migration_storage.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace tide {
namespace {
bool fail(std::string& error, const char* message) {
    error = message;
    return false;
}
bool slot_addressable(const ServerData& data, uint32_t slot) {
    return slot != UINT32_MAX && slot < n_local_load(data) &&
        data.adj.segment(slot / decltype(data.adj)::chunk_size) &&
        data.local_to_global.segment(slot / decltype(data.local_to_global)::chunk_size);
}
}  // namespace

bool validate_membership_migration_generations(
    const ServerData& data, const MembershipMigrationJournal& journal,
    std::string& error) {
    if (!journal.lifetime || journal.lifetime->rollback_revoked)
        return fail(error, "migration journal storage capability revoked");
    for (const auto& entry : journal.entries) {
        if (data.my_partition == entry.from_partition &&
            (!slot_addressable(data, entry.source_local_idx) ||
             migration_slot_generation(data, entry.source_local_idx) != entry.source_generation))
            return fail(error, "migration source slot generation mismatch");
        if (data.my_partition == entry.to_partition && entry.target_local_idx != UINT32_MAX &&
            (!slot_addressable(data, entry.target_local_idx) ||
             migration_slot_generation(data, entry.target_local_idx) != entry.target_generation))
            return fail(error, "migration target slot generation mismatch");
    }
    return true;
}

bool prepare_membership_migration(
    ServerData& data, const std::vector<MembershipMove>& moves,
    uint64_t epoch_from, uint64_t epoch_to,
    MembershipMigrationJournal& journal, std::string& error, bool vector_digests) {
    error.clear();
    if (journal.prepared || journal.committed || journal.rolled_back ||
        !journal.entries.empty() || journal.lifetime)
        return fail(error, "migration journal is not fresh");
    if (moves.empty() || epoch_from == UINT64_MAX || epoch_to != epoch_from + 1)
        return fail(error, "invalid migration epoch or empty move set");
    const uint32_t before = n_local_load(data), n_base = n_base_load(data);
    MembershipMigrationJournal staged;
    staged.epoch_from = epoch_from; staged.epoch_to = epoch_to;
    staged.n_local_before = before;
    std::vector<std::vector<uint32_t>> rows;
    size_t target_count = 0;
    try {
        std::unordered_set<uint32_t> seen;
        staged.entries.reserve(moves.size());
        rows.resize(moves.size());
        staged.lifetime = std::make_shared<MembershipMigrationLifetime>();
        for (size_t i = 0; i < moves.size(); ++i) {
            const auto& move = moves[i];
            if (!seen.insert(move.global_id).second || move.global_id >= n_base ||
                move.from_partition >= data.n_servers || move.to_partition >= data.n_servers ||
                move.from_partition == move.to_partition ||
                mem_at(data, move.global_id) != move.from_partition ||
                migration_row_digest(move.adjacency) != move.adjacency_digest ||
                std::any_of(move.adjacency.begin(), move.adjacency.end(),
                            [n_base](uint32_t id) { return id >= n_base; }))
                return fail(error, "migration move identity/payload invalid");
            // A shipped vector must have this server's dimension and match the
            // digest its source computed at REFRESH (wire v4 sends it only to
            // the target; a v3 file plan carries it on every move).
            if (!move.vector.empty() &&
                (move.vector.size() != data.dim || migration_vector_digest(move.vector) != move.vector_digest))
                return fail(error, move.vector.size() != data.dim
                    ? "migration move vector dimension differs from the server's"
                    : "migration move vector digest mismatch");
            // Own-only storage: the target installs the shipped vector, so a
            // target move without the node's original vector is refused before
            // anything changes. Sources and bystanders need none.
            if (data.own_only && data.my_partition == move.to_partition && move.vector.size() != data.dim)
                return fail(error, "own-only migration move lacks the node's vector");
            MembershipMoveJournalEntry entry;
            entry.global_id = move.global_id; entry.from_partition = move.from_partition;
            entry.to_partition = move.to_partition;
            // Checked: an inserted id's global_to_local chunk may be absent here.
            const uint32_t local = g2l_checked(data, move.global_id);
            if (data.my_partition == move.from_partition) {
                if (!slot_addressable(data, local) ||
                    data.local_to_global.at(local) != move.global_id)
                    return fail(error, "migration source mapping is invalid");
                if (migration_row_digest(data.adj.at(local)) != move.adjacency_digest)
                    return fail(error, "migration source adjacency differs");
                // Own-only: the plan's vector digest must be that of the
                // source's own vector (the original), or the target would
                // install another vector and ACCEPT free the only good copy.
                if (vector_digests && data.own_only) {
                    const float* own = own_vec(data, move.global_id);
                    if (own == nullptr || migration_vector_digest(own, data.dim) != move.vector_digest)
                        return fail(error, "migration source vector differs from the plan's vector digest");
                }
                entry.source_local_idx = local;
                entry.source_generation = migration_slot_generation(data, local);
            } else if (local != UINT32_MAX) {
                return fail(error, "migration non-source already has active row");
            }
            if (data.my_partition == move.to_partition) {
                ++target_count;
                rows[i] = move.adjacency; // every throwing copy precedes publication
                if (!data.global_to_local.ensure_capacity_for(move.global_id, UINT32_MAX))
                    return fail(error, "migration target id mapping capacity exhausted");
            }
            staged.entries.push_back(entry);
        }
        std::unordered_set<uint32_t> free_seen;
        for (uint32_t slot : data.free_slot_pool) {
            if (!free_seen.insert(slot).second || !slot_addressable(data, slot) ||
                data.local_to_global.at(slot) != UINT32_MAX ||
                !data.adj.at(slot).empty() || data.adj.at(slot).capacity() != 0)
                return fail(error, "free row-slot pool invariant violated");
        }
        const size_t reuse_count = std::min(target_count, data.free_slot_pool.size());
        const uint64_t end = uint64_t{before} + target_count - reuse_count;
        if (end > UINT32_MAX)
            return fail(error, "migration target local-index overflow");
        if (end > decltype(data.adj)::max_capacity ||
            end > decltype(data.local_to_global)::max_capacity)
            return fail(error, "migration target local-index capacity exhausted");
        staged.n_local_after_prepare = static_cast<uint32_t>(end);
        if (end > before &&
            (!data.adj.ensure_capacity_for(end - 1) ||
             !data.local_to_global.ensure_capacity_for(end - 1, UINT32_MAX)))
            return fail(error, "migration target capacity allocation failed");
        if (end > before && !data.node_locks.grow_to(end))
            return fail(error, "migration target node-lock allocation failed");
        if (data.own_only && target_count > 0) {
            // Own vectors by local slot: the largest slot is end - 1 or a reused
            // pool slot below it. An inserted node's code is stored now if this
            // server has not got it yet (the codes of load-time ids are static).
            if (end > 0 && !data.full_own.ensure_capacity_for(end - 1))
                return fail(error, "migration target vector capacity allocation failed");
            for (const auto& move : moves) {
                if (data.my_partition == move.to_partition && pq_code(data, move.global_id) == nullptr)
                    store_pq_code(data, move.global_id, move.vector.data());
            }
        }
        size_t reused = 0;
        uint32_t append = before;
        for (auto& entry : staged.entries) {
            if (data.my_partition != entry.to_partition) continue;
            entry.target_reused = reused < reuse_count;
            const uint32_t slot = entry.target_reused
                ? data.free_slot_pool[data.free_slot_pool.size() - 1 - reused++]
                : append++;
            entry.target_local_idx = slot;
            if (!data.migration_slot_generations.ensure_capacity_for(slot))
                return fail(error, "migration generation capacity allocation failed");
            const uint64_t generation = migration_slot_generation(data, slot);
            // Keep one transition available for mandatory rollback/retirement.
            if (generation >= UINT64_MAX - 1)
                return fail(error, "migration target generation exhausted");
            entry.target_generation = generation + 1;
        }
        // Reserve rollback returns now, not after any published mapping.
        data.free_slot_pool.reserve(data.free_slot_pool.size() + target_count);
        // No allocation or throwing row copy remains below this boundary.
        for (size_t i = 0; i < staged.entries.size(); ++i) {
            const auto& entry = staged.entries[i];
            if (data.my_partition != entry.to_partition) continue;
            const uint32_t slot = entry.target_local_idx;
            data.adj.at(slot).swap(rows[i]);
            data.migration_slot_generations.at(slot) = entry.target_generation;
            // The vector is in place before the mapping that makes it reachable.
            if (data.own_only)
                std::copy(moves[i].vector.begin(), moves[i].vector.end(), data.full_own.vec(slot));
            data.local_to_global.at(slot) = entry.global_id;
            data.global_to_local.at(entry.global_id) = slot;
        }
        data.free_slot_pool.resize(data.free_slot_pool.size() - reuse_count);
        n_local_set(data, staged.n_local_after_prepare);
        staged.prepared = true;
        journal = std::move(staged);
        return true;
    } catch (const std::exception& ex) {
        error = std::string("migration allocation failed before publication: ") + ex.what();
        return false;
    }
}

bool reserve_membership_rollback_slots(
    ServerData& data, const MembershipMigrationJournal& journal, bool pool_appended, std::string& error) {
    size_t returns = 0;
    try {
        for (const auto& entry : journal.entries) {
            if (data.my_partition != entry.to_partition || entry.target_local_idx == UINT32_MAX)
                continue;
            if (!data.migration_slot_generations.ensure_capacity_for(entry.target_local_idx) ||
                migration_slot_generation(data, entry.target_local_idx) == UINT64_MAX)
                return fail(error, "rollback target generation exhausted");
            if (entry.target_reused || pool_appended) {
                if (std::find(data.free_slot_pool.begin(), data.free_slot_pool.end(),
                              entry.target_local_idx) != data.free_slot_pool.end())
                    return fail(error, "rollback target already in the free row-slot pool");
                ++returns;
            }
        }
        data.free_slot_pool.reserve(data.free_slot_pool.size() + returns);
        return true;
    } catch (const std::exception& ex) {
        error = std::string("rollback allocation failed before mutation: ") + ex.what();
        return false;
    }
}

// Zeroes the own vector of a slot that left service (own-only), so a free
// slot keeps no node's vector.
static void clear_own_vector_slot(ServerData& data, uint32_t slot) noexcept {
    if (!data.own_only || !data.full_own.segment_allocated(slot / ChunkedVectorStore::CHUNK_NODES)) return;
    float* vec = data.full_own.vec(slot);
    std::fill(vec, vec + data.dim, 0.0f);
}

void release_membership_target_slot(
    ServerData& data, const MembershipMoveJournalEntry& entry, bool to_pool) {
    data.global_to_local.at(entry.global_id) = UINT32_MAX;
    data.local_to_global.at(entry.target_local_idx) = UINT32_MAX;
    std::vector<uint32_t>().swap(data.adj.at(entry.target_local_idx));
    clear_own_vector_slot(data, entry.target_local_idx);
    ++data.migration_slot_generations.at(entry.target_local_idx);
    if (to_pool) data.free_slot_pool.push_back(entry.target_local_idx);
}

bool reserve_membership_source_release(
    ServerData& data, const MembershipMigrationJournal& journal, std::string& error) {
    size_t sources = 0;
    try {
        std::unordered_set<uint32_t> pooled(data.free_slot_pool.begin(), data.free_slot_pool.end());
        for (const auto& entry : journal.entries) {
            if (data.my_partition != entry.from_partition) continue;
            if (!slot_addressable(data, entry.source_local_idx) || pooled.count(entry.source_local_idx))
                return fail(error, "accepted source slot is not addressable or already free");
            if (!data.migration_slot_generations.ensure_capacity_for(entry.source_local_idx) ||
                migration_slot_generation(data, entry.source_local_idx) >= UINT64_MAX - 1)
                return fail(error, "accepted source slot generation exhausted");
            ++sources;
        }
        data.free_slot_pool.reserve(data.free_slot_pool.size() + sources);
        return true;
    } catch (const std::exception& ex) {
        error = std::string("ACCEPT allocation failed before release: ") + ex.what();
        return false;
    }
}

void release_accepted_source_rows(ServerData& data, const MembershipMigrationJournal& journal) noexcept {
    for (const auto& entry : journal.entries) {
        if (data.my_partition != entry.from_partition) continue;
        const uint32_t slot = entry.source_local_idx;
        // Zero capacity: the pool invariant (PREPARE checks it before reuse).
        std::vector<uint32_t>().swap(data.adj.at(slot));
        clear_own_vector_slot(data, slot);
        ++data.migration_slot_generations.at(slot);
        data.free_slot_pool.push_back(slot);  // capacity reserved: cannot throw
    }
}

bool rollback_membership_migration(
    ServerData& data, const std::vector<MembershipMove>& moves,
    MembershipMigrationJournal& journal, std::string& error) {
    error.clear();
    if (!journal.prepared || journal.rolled_back)
        return fail(error, "migration journal is not rollback-ready");
    if (!membership_migration_detail::journal_matches(moves, journal, error) ||
        !validate_membership_migration_generations(data, journal, error)) return false;
    // Validate ALL ownership/mappings before clearing flags or restoring a source.
    for (const auto& entry : journal.entries) {
        const bool moved = journal.committed;
        if (entry.global_id >= n_base_load(data) ||
            mem_at(data, entry.global_id) != (moved ? entry.to_partition : entry.from_partition))
            return fail(error, "rollback membership changed");
        if (data.my_partition == entry.from_partition) {
            const uint32_t expected = moved ? UINT32_MAX : entry.source_local_idx;
            if (data.global_to_local.at(entry.global_id) != expected ||
                data.local_to_global.at(entry.source_local_idx) !=
                    (moved ? UINT32_MAX : entry.global_id))
                return fail(error, "rollback source mapping changed");
        }
        if (data.my_partition == entry.to_partition &&
            (entry.target_local_idx == UINT32_MAX ||
             data.global_to_local.at(entry.global_id) != entry.target_local_idx ||
             data.local_to_global.at(entry.target_local_idx) != entry.global_id))
            return fail(error, "rollback target mapping changed");
    }
    // PREPARE is quiescent, so an unchanged allocator still ends at this
    // journal's suffix: n_local shrinks back over the appended target slots.
    // If it changed since (a slot appended after PREPARE), the appended slots
    // stay below n_local, unmapped with an empty row: they join the free pool.
    const bool shrink = n_local_load(data) == journal.n_local_after_prepare;
    if (!reserve_membership_rollback_slots(data, journal, !shrink, error)) return false;
    for (const auto& entry : journal.entries) {
        if (journal.committed) mem_set(data, entry.global_id, entry.from_partition);
        if (journal.committed && data.my_partition == entry.from_partition) {
            data.global_to_local.at(entry.global_id) = entry.source_local_idx;
            data.local_to_global.at(entry.source_local_idx) = entry.global_id;
        }
        if (data.my_partition != entry.to_partition || entry.target_local_idx == UINT32_MAX)
            continue;
        release_membership_target_slot(data, entry, entry.target_reused || !shrink);
    }
    if (shrink) n_local_set(data, journal.n_local_before);
    journal.lifetime->rollback_revoked = true;
    journal.committed = false; journal.prepared = false; journal.rolled_back = true;
    return true;
}

}  // namespace tide
