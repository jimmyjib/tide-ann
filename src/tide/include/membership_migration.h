#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "membership_migration_plan.h"
#include "server_types.h"

namespace tide {

struct MembershipMoveJournalEntry {
    uint32_t global_id = UINT32_MAX;
    uint32_t from_partition = UINT32_MAX;
    uint32_t to_partition = UINT32_MAX;
    uint32_t source_local_idx = UINT32_MAX;
    uint32_t target_local_idx = UINT32_MAX;
    uint64_t source_generation = 0;
    uint64_t target_generation = 0;
    bool target_reused = false;  // target slot came from the free pool
};

// shared by every copy of one journal
struct MembershipMigrationLifetime {
    bool accepted = false;
    // set by ROLLBACK and ACCEPT: no rollback after that
    bool rollback_revoked = false;
};

// Server side of a MOVE. Callers hold the admin lock exclusively with the server quiescent.
// PREPARE installs the target rows, COMMIT publishes the membership, ACCEPT makes it final and
// ROLLBACK (before ACCEPT) restores it.
struct MembershipMigrationJournal {
    uint64_t epoch_from = 0;
    uint64_t epoch_to = 0;
    uint32_t n_local_before = 0;
    uint32_t n_local_after_prepare = 0;
    bool prepared = false;
    bool committed = false;
    bool rolled_back = false;
    std::vector<MembershipMoveJournalEntry> entries;
    std::shared_ptr<MembershipMigrationLifetime> lifetime;
};

extern MembershipMigrationJournal g_membership_migration_journal;

// vector_digests: in PQ mode the source's own vector must match each move's vector digest.
bool prepare_membership_migration(
    ServerData& data,
    const std::vector<MembershipMove>& moves,
    uint64_t epoch_from,
    uint64_t epoch_to,
    MembershipMigrationJournal& journal,
    std::string& error,
    bool vector_digests = false);

bool commit_membership_migration(
    ServerData& data,
    const std::vector<MembershipMove>& moves,
    MembershipMigrationJournal& journal,
    std::string& error);

// Rejected once rolled back or accepted. PREPARE's appended target slots are returned by shrinking
// n_local, or join the free pool if n_local grew since.
bool rollback_membership_migration(
    ServerData& data,
    const std::vector<MembershipMove>& moves,
    MembershipMigrationJournal& journal,
    std::string& error);

// Revokes rollback. The source rows are freed by accept_membership_migration_history.
bool accept_membership_migration(
    const ServerData& data,
    const std::vector<MembershipMove>& moves,
    const MembershipMigrationJournal& journal,
    std::string& error);

// ACCEPT: free this server's source rows into the pool, then drop the plan and journal and keep the
// accepted epoch. All checks and allocations come first.
bool accept_membership_migration_history(ServerData& data, MembershipMigrationPlan& plan,
    MembershipMigrationJournal& journal, std::vector<uint64_t>& accepted_epochs,
    std::string& error);

// A retried ACCEPT of the epoch this server already accepted (answered with move_count 0).
bool membership_migration_accept_is_repeat(const MembershipMigrationPlan& active,
    const std::vector<uint64_t>& accepted_epochs, uint64_t ownership_epoch);

enum class MigrationRollbackDisposition : uint8_t {
    ROLL_BACK,        // the request names the active plan: roll it back
    NO_OP,            // never prepared here
    REFUSE_ACCEPTED,  // the request names an accepted epoch: ACCEPT is final
    REFUSE_MISMATCH,  // the request names another plan or epoch
};
// what a ROLLBACK of epoch_from -> epoch_to does on this server
MigrationRollbackDisposition membership_migration_rollback_disposition(
    const MembershipMigrationPlan& active, const std::vector<uint64_t>& accepted_epochs,
    uint64_t ownership_epoch, uint64_t epoch_from, uint64_t epoch_to);

// membership-check on one server: owner labels, rows, vectors (PQ mode) and the free pool against
// the coordinator's owners. The caller holds the admin lock with the server quiescent.
struct MembershipCheckCounts {
    uint32_t ids = 0, server_ids = 0, n_local = 0, free_slots = 0;
    uint32_t first_mismatch = UINT32_MAX;
    bool pool_ok = true;
    uint64_t owned = 0, mapped = 0, vectors = 0;
    uint64_t membership_mismatches = 0, mapping_mismatches = 0, vector_mismatches = 0, missing_codes = 0;
    uint64_t vector_digest = 0;
};
MembershipCheckCounts membership_check(const ServerData& data, const uint32_t* owners, uint32_t ids);

// Current rows of the moved nodes this server owns (after the drain, before PREPARE).
bool snapshot_membership_migration_source_rows(
    const ServerData& data,
    const MembershipMigrationPlan& move_template,
    MembershipMigrationPlan& source_fragment,
    std::string& error);

}  // namespace tide
