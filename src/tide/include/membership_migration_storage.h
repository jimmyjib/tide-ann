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
                    ": journal entry does not match the move";
            return false;
        }
    }
    return true;
}
}  // namespace membership_migration_detail

// Callers hold the admin lock exclusively with the server quiescent.
bool validate_membership_migration_generations(
    const ServerData& data, const MembershipMigrationJournal& journal,
    std::string& error);
// pool_appended: PREPARE's appended slots join the free pool too
bool reserve_membership_rollback_slots(
    ServerData& data, const MembershipMigrationJournal& journal,
    bool pool_appended, std::string& error);
// Unmaps a target slot, clears its row and vector, bumps its generation and optionally pools it.
void release_membership_target_slot(
    ServerData& data, const MembershipMoveJournalEntry& entry, bool to_pool);
// ACCEPT: the reservation may fail but the release cannot (rows cleared, slots pooled).
bool reserve_membership_source_release(
    ServerData& data, const MembershipMigrationJournal& journal, std::string& error);
void release_accepted_source_rows(ServerData& data, const MembershipMigrationJournal& journal) noexcept;

}  // namespace tide
