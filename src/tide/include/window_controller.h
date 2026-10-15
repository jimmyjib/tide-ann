// Shared types of the runtime repair controller: the window tables merged from
// the servers' snapshot frames, and one planned MOVE.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "window_snapshot_wire.h"

namespace tide {

struct WindowTables {
    // work[v]: distances computed while expanding v during the window, on v's
    // owner (the only server that expands it).
    std::vector<uint64_t> work;
    size_t nodes = 0;
    std::vector<uint64_t> per_rank_work;  // [rank] = sum of that rank's window work
};

// The structure sections of every rank, scattered to global ids.
struct WindowStructure {
    bool valid = false;
    uint32_t sections = 0, owners = 0;
    std::vector<uint32_t> owner_hist;  // [nodes * owners], row v at v * owners (bit kWindowStructureOwnerHist)
    std::vector<uint64_t> nb_work;     // [nodes], summed over ranks (bit kWindowStructureNbWork)
    uint64_t cut_edges = 0, nb_work_total = 0;
    std::string error;
};

// One ownership MOVE of node `node` from `source_owner` to `target_owner`.
// adjacency_bytes is the node's row bytes (4 + 4 * degree), the unit of every
// byte budget and cap.
struct WindowBurstAction {
    uint32_t node = 0;
    uint32_t source_owner = 0;
    uint32_t target_owner = 0;
    uint64_t adjacency_bytes = 0;
};

// Merges one decoded frame per rank (frames[r] from rank r, all K = owners of
// them) in memory against the coordinator's membership: every rank's owned
// count and owned-id digest must equal those of `membership`, then each rank's
// rows are scattered back to their ascending global ids. Fills `tables` and
// `structure` (both left empty with an error on any mismatch).
bool window_merge_frames(const std::vector<std::shared_ptr<const WindowRankFrame>>& frames,
                         const std::vector<uint32_t>& membership, uint32_t owners, uint32_t sections,
                         WindowTables* tables, WindowStructure* structure, std::string* error);

}  // namespace tide
