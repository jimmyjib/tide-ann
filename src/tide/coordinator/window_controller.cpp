#include "window_controller.h"

namespace tide {

bool window_merge_frames(const std::vector<std::shared_ptr<const WindowRankFrame>>& frames,
                         const std::vector<uint32_t>& membership, uint32_t owners, uint32_t sections,
                         WindowTables* tables, WindowStructure* structure, std::string* error) {
    auto fail = [&](const std::string& why) {
        if (tables) *tables = WindowTables{};
        if (structure) { *structure = WindowStructure{}; structure->error = why; }
        if (error) *error = why;
        return false;
    };
    if (!tables || !structure || !window_owners_valid(owners) || !window_structure_sections_valid(sections) ||
        frames.size() != owners || membership.empty())
        return fail("invalid merge arguments");
    const bool want_hist = sections & kWindowStructureOwnerHist;
    const bool want_nbwork = sections & kWindowStructureNbWork;
    const size_t nodes = membership.size();
    std::vector<uint64_t> counts, digests;
    if (!window_owned_digests(membership, owners, counts, digests)) return fail("membership owner out of range");
    for (uint32_t rank = 0; rank < owners; ++rank) {
        const auto& frame = frames[rank];
        if (!frame) return fail("snapshot missing at rank " + std::to_string(rank));
        const auto& h = frame->header;
        if (h.ack.server_rank != rank || h.owners != owners || h.nodes != nodes || h.sections != sections ||
            frame->w.size() != h.owned_count ||
            frame->hist.size() != (want_hist ? size_t(h.owned_count) * owners : 0) ||
            frame->nbwork.size() != (want_nbwork ? nodes : 0))
            return fail("snapshot shape differs at rank " + std::to_string(rank));
        if (h.owned_count != counts[rank] || h.owned_digest != digests[rank])
            return fail("owned ids differ from the membership at rank " + std::to_string(rank));
    }
    WindowTables merged;
    WindowStructure out;
    merged.nodes = nodes;
    merged.work.assign(nodes, 0);
    merged.per_rank_work.assign(owners, 0);
    out.sections = sections; out.owners = owners;
    if (want_hist) out.owner_hist.assign(nodes * owners, 0);
    if (want_nbwork) out.nb_work.assign(nodes, 0);
    std::vector<size_t> row(owners, 0);
    for (size_t v = 0; v < nodes; ++v) {
        const uint32_t rank = membership[v];
        const auto& frame = *frames[rank];
        const size_t r = row[rank]++;
        merged.work[v] = frame.w[r];
        merged.per_rank_work[rank] += frame.w[r];
        if (want_hist) {
            uint32_t* to = out.owner_hist.data() + v * owners;
            const uint16_t* from = frame.hist.data() + r * owners;
            for (uint32_t p = 0; p < owners; ++p) {
                to[p] = from[p];
                if (p != rank) out.cut_edges += from[p];
            }
        }
    }
    if (want_nbwork) {
        for (uint32_t rank = 0; rank < owners; ++rank) {
            const auto& nb = frames[rank]->nbwork;
            for (size_t v = 0; v < nodes; ++v) {
                if (nb[v] > UINT64_MAX - out.nb_work[v]) return fail("structure nbwork overflow");
                out.nb_work[v] += nb[v];
            }
        }
        for (const uint64_t value : out.nb_work) out.nb_work_total += value;
    }
    out.valid = true;
    *tables = std::move(merged);
    *structure = std::move(out);
    if (error) error->clear();
    return true;
}

}  // namespace tide
