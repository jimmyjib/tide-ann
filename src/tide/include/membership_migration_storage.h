#pragma once

#include "membership_migration.h"

namespace tide {

namespace membership_migration_detail {
inline bool journal_matches(const std::vector<MembershipMove>& moves,
                            const MembershipMigrationJournal& journal,
                            std::string& error) {
    if (moves.size() != journal.entries.size()) {
        error = "migration move/journal size mismatch";
        return false;
    }
    for (size_t index = 0; index < moves.size(); ++index) {
        const auto& move = moves[index];
        const auto& entry = journal.entries[index];
        if (move.global_id != entry.global_id ||
            move.from_partition != entry.from_partition ||
            move.to_partition != entry.to_partition) {
            error = "migration node " + std::to_string(move.global_id) +
                    ": journal identity mismatch";
            return false;
        }
    }
    return true;
}
}  // namespace membership_migration_detail

// Internal lifecycle guards. All callers obey the migration admin/quiescence
// contract; the live installer alone writes its newly appended target slots.
bool validate_membership_migration_generations(
    const ServerData& data, const MembershipMigrationJournal& journal,
    std::string& error);
// `pool_appended`: the rollback cannot shrink n_local over the slots PREPARE
// appended (n_local changed since), so those slots join the free pool too.
bool reserve_membership_rollback_slots(
    ServerData& data, const MembershipMigrationJournal& journal,
    bool pool_appended, std::string& error);
// Unmaps a target slot, drops its row (zero capacity) and own vector, bumps its
// generation and, when `to_pool`, pushes it to the free row-slot pool (capacity
// reserved by reserve_membership_rollback_slots).
void release_membership_target_slot(
    ServerData& data, const MembershipMoveJournalEntry& entry, bool to_pool);
// ACCEPT frees the source rows. The reservation (pool capacity, slot
// generations) is made before anything changes and may fail; the release
// cannot: each source slot loses its row (swapped to zero capacity), its own
// vector (zeroed, own-only) and its generation (bumped), and joins the free
// row-slot pool that the next PREPARE reuses.
bool reserve_membership_source_release(
    ServerData& data, const MembershipMigrationJournal& journal, std::string& error);
void release_accepted_source_rows(ServerData& data, const MembershipMigrationJournal& journal) noexcept;

}  // namespace tide
