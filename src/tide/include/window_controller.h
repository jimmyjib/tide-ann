// Window tables merged from the servers' snapshots, and one planned MOVE.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "window_snapshot_wire.h"

namespace tide {

struct WindowTables {
    // work[v]: distance evaluations of v's expansions in the window (on v's owner)
    std::vector<uint64_t> work;
    size_t nodes = 0;
    std::vector<uint64_t> per_rank_work;  // sum per rank
};

// structure sections of every rank, by global id
struct WindowStructure {
    bool valid = false;
    uint32_t sections = 0, owners = 0;
    std::vector<uint32_t> owner_hist;  // [nodes * owners], row v at v * owners (bit kWindowStructureOwnerHist)
    std::vector<uint64_t> nb_work;     // [nodes], summed over ranks (bit kWindowStructureNbWork)
    uint64_t cut_edges = 0, nb_work_total = 0;
    std::string error;
};

// adjacency_bytes: the node's row bytes (4 + 4 * degree), the unit of every budget and cap.
struct WindowBurstAction {
    uint32_t node = 0;
    uint32_t source_owner = 0;
    uint32_t target_owner = 0;
    uint64_t adjacency_bytes = 0;
};

// Merges one decoded frame per rank against the membership. A mismatch gives empty tables and an error.
bool window_merge_frames(const std::vector<std::shared_ptr<const WindowRankFrame>>& frames,
                         const std::vector<uint32_t>& membership, uint32_t owners, uint32_t sections,
                         WindowTables* tables, WindowStructure* structure, std::string* error);

}  // namespace tide
