// Stats-window snapshot frame (version 3), one per rank. After the header (K, N, owned count and digest):
//   w       [owned_count]      u32  work of each owned id, ascending id
//   hist    [owned_count * K]  u16  out-neighbors of each owned id by owner (section OwnerHist)
//   nbwork  [N]                u64  w(u) of each owned row u added to its out-neighbors (section NbWork)
// The SHA-256 covers the frame with its own 64 bytes zeroed.
#pragma once

#include "protocol/message_types.h"
#include "sha256.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace tide {
inline constexpr uint32_t kWindowSnapshotMagic = 0x33574E53u;  // "SNW3"
inline constexpr uint8_t kWindowSnapshotVersion = 3;
// Owner counts the wire carries: 2 <= K <= 64 (the planner's target mask is one uint64).
inline constexpr uint32_t kWindowMaxOwners = 64;
inline constexpr uint32_t kWindowStructureOwnerHist = 1u << 1;
inline constexpr uint32_t kWindowStructureNbWork = 1u << 2;
inline constexpr uint32_t kWindowStructureSectionsMask = kWindowStructureOwnerHist | kWindowStructureNbWork;
// ann::StatsWindowBeginRequest::version understood by this build.
inline constexpr uint8_t kStatsWindowBeginVersion = 1;

inline bool window_owners_valid(uint32_t owners) { return owners >= 2 && owners <= kWindowMaxOwners; }
inline bool window_structure_sections_valid(uint32_t sections) {
    return sections && !(sections & ~kWindowStructureSectionsMask);
}

// Digest of a rank's owned ids: splitmix64 folded over the ids in ascending order.
inline constexpr uint64_t kWindowOwnedDigestSeed = 0x6F776E6564763331ull;  // fixed seed
inline uint64_t window_owned_digest_add(uint64_t digest, uint32_t id) {
    uint64_t z = (digest ^ id) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// Owned count and digest per rank. False when a label is not below `owners`.
inline bool window_owned_digests(const std::vector<uint32_t>& membership, uint32_t owners,
                                 std::vector<uint64_t>& counts, std::vector<uint64_t>& digests) {
    counts.assign(owners, 0);
    digests.assign(owners, kWindowOwnedDigestSeed);
    for (size_t v = 0; v < membership.size(); ++v) {
        const uint32_t owner = membership[v];
        if (owner >= owners || v > UINT32_MAX) return false;
        ++counts[owner];
        digests[owner] = window_owned_digest_add(digests[owner], static_cast<uint32_t>(v));
    }
    return true;
}

#pragma pack(push, 1)
struct WindowSnapshotRequest {
    ann::MessageType msg_type = ann::MessageType::STATS_WINDOW_SNAPSHOT_REQUEST;
    uint8_t version = kWindowSnapshotVersion;
    uint16_t sections = 0;  // requested structure sections
    uint32_t nodes = 0;     // the window's node count N
    uint64_t epoch = 0;
};
struct WindowSnapshotHeader {
    ann::StatsWindowAck ack{};
    uint32_t magic = kWindowSnapshotMagic;
    uint32_t version = kWindowSnapshotVersion;
    uint32_t header_bytes = sizeof(WindowSnapshotHeader);
    uint32_t owners = 0;       // K
    uint32_t nodes = 0;        // N
    uint32_t owned_count = 0;
    uint32_t sections = 0;
    uint32_t reserved = 0;
    uint64_t owned_digest = 0;
    uint64_t non_owned_w = 0;  // window work on ids this rank does not own (must be 0)
    uint64_t w_words = 0, hist_words = 0, nbwork_words = 0;
    uint64_t payload_bytes = 0;
    char sha256[64]{};
};
struct WindowStopRequest {
    ann::MessageType msg_type = ann::MessageType::STATS_WINDOW_STOP_REQUEST;
    uint8_t version = 1;
    uint16_t reserved = 0;
    uint32_t ranks = 0;  // the sender's server count, which must equal the server's n_servers
    uint64_t epoch = 0;
};
struct WindowStopAck {
    ann::MessageType msg_type = ann::MessageType::STATS_WINDOW_STOP_ACK;
    uint8_t status = 1, version = 1, active = 0;
    uint32_t rank = 0;
    uint64_t requested_epoch = 0, actual_epoch = 0;
};
#pragma pack(pop)
static_assert(sizeof(WindowSnapshotRequest) == 16 && sizeof(WindowSnapshotHeader) == 240);
static_assert(sizeof(WindowStopRequest) == 16 && sizeof(WindowStopAck) == 24);

// Exact payload size of a frame with these section lengths.
inline bool window_snapshot_payload_bytes(uint64_t w_words, uint64_t hist_words, uint64_t nbwork_words,
                                          uint64_t& bytes) {
    if (w_words > UINT32_MAX || hist_words > uint64_t{UINT32_MAX} * kWindowMaxOwners ||
        nbwork_words > UINT32_MAX) return false;
    bytes = w_words * 4 + hist_words * 2 + nbwork_words * 8;
    return true;
}
// The section lengths a frame must carry for (owners, nodes, owned_count, sections).
inline bool window_snapshot_shape_valid(const WindowSnapshotHeader& h) {
    return h.magic == kWindowSnapshotMagic && h.version == kWindowSnapshotVersion &&
        h.header_bytes == sizeof(WindowSnapshotHeader) && window_owners_valid(h.owners) && h.nodes &&
        window_structure_sections_valid(h.sections) && !h.reserved && h.owned_count <= h.nodes &&
        h.w_words == h.owned_count &&
        h.hist_words == ((h.sections & kWindowStructureOwnerHist) ? uint64_t{h.owned_count} * h.owners : 0) &&
        h.nbwork_words == ((h.sections & kWindowStructureNbWork) ? uint64_t{h.nodes} : 0);
}
// Largest frame worth queuing (every node owned, every section). A larger one keeps only its header.
inline size_t window_snapshot_queue_bytes(const void* data, size_t size, uint32_t nodes, uint32_t owners) {
    if (!data || !size || *static_cast<const uint8_t*>(data) !=
        static_cast<uint8_t>(ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE)) return size;
    uint64_t payload = 0;
    if (!window_owners_valid(owners) ||
        !window_snapshot_payload_bytes(nodes, uint64_t{nodes} * owners, nodes, payload) ||
        size > sizeof(WindowSnapshotHeader) + payload) return std::min(size, sizeof(WindowSnapshotHeader));
    return size;
}

// One rank's decoded frame.
struct WindowRankFrame {
    WindowSnapshotHeader header{};
    std::vector<uint32_t> w;       // [owned_count]
    std::vector<uint16_t> hist;    // [owned_count * owners] or empty
    std::vector<uint64_t> nbwork;  // [nodes] or empty
};

// Builds a frame. An inconsistent combination yields no frame (empty).
inline std::vector<char> window_snapshot_encode(ann::StatsWindowAck ack, uint32_t owners, uint32_t nodes,
        uint32_t sections, uint64_t owned_digest, uint64_t non_owned_w, const std::vector<uint32_t>& w,
        const std::vector<uint16_t>& hist, const std::vector<uint64_t>& nbwork) {
    WindowSnapshotHeader header;
    ack.msg_type = ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE;
    header.ack = ack;
    header.owners = owners; header.nodes = nodes; header.sections = sections;
    if (w.size() > UINT32_MAX) return {};
    header.owned_count = static_cast<uint32_t>(w.size());
    header.owned_digest = owned_digest; header.non_owned_w = non_owned_w;
    header.w_words = w.size(); header.hist_words = hist.size(); header.nbwork_words = nbwork.size();
    uint64_t payload = 0;
    if (!window_snapshot_shape_valid(header) ||
        !window_snapshot_payload_bytes(header.w_words, header.hist_words, header.nbwork_words, payload) ||
        sizeof(header) + payload > std::numeric_limits<size_t>::max()) return {};
    header.payload_bytes = payload;
    std::vector<char> bytes(sizeof(header) + payload);
    char* out = bytes.data();
    std::memcpy(out, &header, sizeof(header)); out += sizeof(header);
    if (!w.empty()) { std::memcpy(out, w.data(), w.size() * 4); out += w.size() * 4; }
    if (!hist.empty()) { std::memcpy(out, hist.data(), hist.size() * 2); out += hist.size() * 2; }
    if (!nbwork.empty()) std::memcpy(out, nbwork.data(), nbwork.size() * 8);
    const auto digest = sha256_hex(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    std::memcpy(bytes.data() + offsetof(WindowSnapshotHeader, sha256), digest.data(), 64);
    return bytes;
}

// Validates a frame against the expected rank, epoch, owner count, node count
// and sections (header fields, exact size, digest, work totals) and fills `out`.
inline bool window_snapshot_decode(const void* data, size_t size, uint32_t rank, uint64_t epoch,
        uint32_t nodes, uint32_t owners, uint32_t sections, WindowRankFrame& out, std::string& error) {
    auto fail = [&](const char* why) { error = why; return false; };
    out = WindowRankFrame{};
    WindowSnapshotHeader header;
    if (!data || size < sizeof(header)) return fail("snapshot truncated header");
    std::memcpy(&header, data, sizeof(header));
    if (!window_owners_valid(owners) || rank >= owners || !epoch ||
        header.ack.msg_type != ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE || header.ack.status ||
        header.ack.reserved[0] || header.ack.reserved[1] || header.ack.reserved_total ||
        header.ack.server_rank != rank || header.ack.window_epoch != epoch)
        return fail("snapshot header type, status, rank or epoch mismatch");
    if (!window_snapshot_shape_valid(header) || header.owners != owners || header.nodes != nodes)
        return fail("snapshot owners/nodes/shape");
    if (header.sections != sections) return fail("snapshot structure sections mismatch");
    uint64_t payload = 0;
    if (!window_snapshot_payload_bytes(header.w_words, header.hist_words, header.nbwork_words, payload) ||
        header.payload_bytes != payload || size != sizeof(header) + payload)
        return fail("snapshot size");
    std::vector<uint8_t> frame(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    std::memset(frame.data() + offsetof(WindowSnapshotHeader, sha256), 0, 64);
    if (sha256_hex(frame.data(), frame.size()) != std::string(header.sha256, 64))
        return fail("snapshot digest");
    out.header = header;
    out.w.resize(header.w_words); out.hist.resize(header.hist_words); out.nbwork.resize(header.nbwork_words);
    const char* in = static_cast<const char*>(data) + sizeof(header);
    if (!out.w.empty()) { std::memcpy(out.w.data(), in, out.w.size() * 4); in += out.w.size() * 4; }
    if (!out.hist.empty()) { std::memcpy(out.hist.data(), in, out.hist.size() * 2); in += out.hist.size() * 2; }
    if (!out.nbwork.empty()) std::memcpy(out.nbwork.data(), in, out.nbwork.size() * 8);
    uint64_t total = 0;
    for (const auto value : out.w) total += value;
    if (header.non_owned_w) return fail("snapshot work on ids this rank does not own");
    if (total != header.ack.total_w) return fail("snapshot W total");
    return true;
}
} // namespace tide
