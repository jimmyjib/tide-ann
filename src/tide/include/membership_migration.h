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
    bool target_reused = false;  // the target slot came from the free row-slot pool
};

// Shared by every copy of one journal.
struct MembershipMigrationLifetime {
    bool accepted = false;
    // Set by ROLLBACK and by ACCEPT: the transaction can never roll back again.
    bool rollback_revoked = false;
};

// Migration is stop-the-world: callers must hold the server admin lock
// exclusively and must have established queue/handler quiescence before any
// operation. PREPARE installs the target rows, COMMIT publishes the new
// membership, ACCEPT makes it final, ROLLBACK (before ACCEPT) restores it.
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

// `vector_digests`: the plan carries every move's vector digest (wire version
// 3 or 4, MembershipMigrationPlan::vector_dim > 0). Under own-only storage the
// source then requires the digest of its own vector to equal the move's, so a
// plan whose vector differs from the original (e.g. a file built from stale
// insert vectors) is refused before anything changes: the target would install
// it and ACCEPT would then free the only correct copy.
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

// Refused once the journal's lifetime is revoked (rolled back or accepted).
// The target slots PREPARE appended are given back by shrinking n_local when
// n_local is still where PREPARE left it; otherwise (it grew since) they stay
// below n_local and join the free row-slot pool, like the reused slots.
bool rollback_membership_migration(
    ServerData& data,
    const std::vector<MembershipMove>& moves,
    MembershipMigrationJournal& journal,
    std::string& error);

// Validates a committed migration and revokes its rollback (every copy of the
// journal shares the lifetime). It frees nothing: the source rows stay
// allocated but unreachable until accept_membership_migration_history frees them.
bool accept_membership_migration(
    const ServerData& data,
    const std::vector<MembershipMove>& moves,
    const MembershipMigrationJournal& journal,
    std::string& error);

// ACCEPT as the server handles it: reserve, accept (rollback_revoked is set
// first), free this server's source rows and own-vector slots into the free
// row-slot pool (each slot's generation bumped, its row swapped to zero
// capacity, its own vector zeroed), then drop the plan and the journal and
// record only the accepted epoch. Every check and allocation precedes the first
// change, so a rejected ACCEPT leaves everything unchanged. The next PREPARE
// reuses the pooled slots before it appends; RESET empties the pool.
bool accept_membership_migration_history(ServerData& data, MembershipMigrationPlan& plan,
    MembershipMigrationJournal& journal, std::vector<uint64_t>& accepted_epochs,
    std::string& error);

// A retried ACCEPT on a server that already accepted the epoch it is at: no
// active plan, and the last accepted epoch is the current ownership epoch. The
// server answers it with success (epoch_from = epoch - 1, epoch_to = epoch,
// move_count 0), so the coordinator can complete a partial ACCEPT by retrying.
bool membership_migration_accept_is_repeat(const MembershipMigrationPlan& active,
    const std::vector<uint64_t>& accepted_epochs, uint64_t ownership_epoch);

enum class MigrationRollbackDisposition : uint8_t {
    ROLL_BACK,        // the request names the active plan: roll it back
    NO_OP,            // no active plan and the request starts at this epoch: never prepared here
    REFUSE_ACCEPTED,  // the request names an accepted epoch: ACCEPT is final
    REFUSE_MISMATCH,  // the request names another plan or epoch
};
// What a ROLLBACK request for the plan epoch_from -> epoch_to does on a server
// with the active plan `active` (no moves when none) at `ownership_epoch`.
MigrationRollbackDisposition membership_migration_rollback_disposition(
    const MembershipMigrationPlan& active, const std::vector<uint64_t>& accepted_epochs,
    uint64_t ownership_epoch, uint64_t epoch_from, uint64_t epoch_to);

// membership-check on one server: its state against the coordinator's owner
// of every id in [0, ids). Per id: mem_at equals the owner (an id the server
// does not know counts as a mismatch); a valid local row (global_to_local and
// local_to_global agree) exactly where the owner is this server; under own-only
// storage the original vector exactly there and a PQ code for every id. Also
// the free row-slot pool: every slot below n_local, unmapped, with a
// zero-capacity row. vector_digest (own-only) is the sum mod 2^64 of
// node_vector_hash over every id whose own vector this server holds, so the
// coordinator can check the bytes against the original vectors. The caller
// holds the admin lock with the server quiescent.
struct MembershipCheckCounts {
    uint32_t ids = 0, server_ids = 0, n_local = 0, free_slots = 0;
    uint32_t first_mismatch = UINT32_MAX;
    bool pool_ok = true;
    uint64_t owned = 0, mapped = 0, vectors = 0;
    uint64_t membership_mismatches = 0, mapping_mismatches = 0, vector_mismatches = 0, missing_codes = 0;
    uint64_t vector_digest = 0;
};
MembershipCheckCounts membership_check(const ServerData& data, const uint32_t* owners, uint32_t ids);

// Snapshot the authoritative current rows owned by this server for a frozen
// identity/from/to move template. This is called only after insert/backlink
// maintenance is drained and before target shadow installation.
bool snapshot_membership_migration_source_rows(
    const ServerData& data,
    const MembershipMigrationPlan& move_template,
    MembershipMigrationPlan& source_fragment,
    std::string& error);

}  // namespace tide
