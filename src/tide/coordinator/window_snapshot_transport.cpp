#include "window_snapshot_transport.h"
#include "window_native_control.h"
#include <cstring>
#include <mutex>
#include <sstream>

namespace tide {
namespace {
void snapshot_event(WindowControllerRuntime& r, const char* event, const std::string& fields) {
    r.journal << "{\"event\":\"" << event << "\",\"window_epoch\":" << r.window_epoch
        << ",\"cycle\":" << r.cycles << ",\"mono_ns\":" << window_journal_now_ns()
        << ',' << fields << "}\n";
    r.journal.flush();
}
std::string json_text(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out + "\"";
}
}
bool window_snapshot_is_response(const void* data, size_t size) {
    return data && size && *static_cast<const uint8_t*>(data) ==
        static_cast<uint8_t>(ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE);
}

namespace {
// Decodes one rank's frame. Frames are merged in memory once all are valid.
WindowSnapshotResult decode_frame(uint32_t server, uint64_t epoch, uint32_t nodes, uint32_t owners,
    const std::vector<char>& bytes, bool duplicate, const std::shared_ptr<WindowControlGeneration>& generation,
    uint32_t expected_sections) {
    const auto start = std::chrono::steady_clock::now();
    WindowSnapshotResult result;
    result.server = server; result.bytes = bytes.size(); result.duplicate = duplicate;
    { std::lock_guard<std::mutex> lock(generation->publish_mutex);
      if (generation->cancelled) { result.error = "cancelled"; return result; } }
    auto frame = std::make_shared<WindowRankFrame>();
    if (!window_snapshot_decode(bytes.data(), bytes.size(), server, epoch, nodes, owners, expected_sections,
                                *frame, result.error)) {
        if (bytes.size() >= sizeof(WindowSnapshotHeader))
            std::memcpy(&result.header, bytes.data(), sizeof(WindowSnapshotHeader));
        return result;
    }
    result.header = frame->header;
    if (!duplicate) result.frame = std::move(frame);
    result.valid = true;
    result.work_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    return result;
}
void apply_receipt(WindowControllerRuntime& r, const WindowSnapshotResult& result) {
    if (r.exchange != WindowExchangeKind::HARVEST || result.header.ack.window_epoch != r.window_epoch) return;
    if (!result.valid) {
        r.exchange_rejected = true;
        snapshot_event(r, "snapshot_rejected", "\"server_rank\":" + std::to_string(result.server) +
                       ",\"reason\":" + json_text(result.error));
        return;
    }
    const auto server = result.server;
    const auto& header = result.header;
    const std::string digest(header.sha256, 64);
    if (server >= r.config.ranks || r.snapshot_digests.size() != r.config.ranks ||
        r.snapshot_frames.size() != r.config.ranks) { r.exchange_rejected = true; return; }
    if (!r.snapshot_digests[server].empty()) {
        if (r.snapshot_digests[server] != digest) r.exchange_rejected = true;
        return;
    }
    if (!result.frame) { r.exchange_rejected = true; return; }
    const auto& sums = r.exchange_counter_sums;
    if (header.ack.search_request_count > UINT64_MAX - sums.requests ||
        header.ack.search_distance_count > UINT64_MAX - sums.distances ||
        header.ack.search_distance_us > UINT64_MAX - sums.distance_us ||
        header.ack.search_total_us > UINT64_MAX - sums.total_us ||
        header.ack.total_w > UINT64_MAX - r.harvest_total_w) {
        r.exchange_rejected = true; return;
    }
    r.snapshot_digests[server] = digest;
    r.snapshot_frames[server] = result.frame;
    auto ack = header.ack; ack.msg_type = ann::MessageType::STATS_WINDOW_HARVEST_ACK;
    window_controller_offer_ack(r, server, &ack, sizeof(ack));
    std::ostringstream fields;
    fields << "\"server_rank\":" << server << ",\"snapshot_sha256\":\"" << digest
        << "\",\"snapshot_bytes\":" << result.bytes << ",\"nodes\":" << header.nodes
        << ",\"owners\":" << header.owners << ",\"owned_count\":" << header.owned_count
        << ",\"total_w\":" << header.ack.total_w
        << ",\"search_request_count\":" << header.ack.search_request_count
        << ",\"search_distance_count\":" << header.ack.search_distance_count
        << ",\"search_distance_us\":" << header.ack.search_distance_us
        << ",\"search_total_us\":" << header.ack.search_total_us
        << ",\"search_exact_count\":" << header.ack.search_exact_count
        << ",\"search_adc_count\":" << header.ack.search_adc_count
        << ",\"search_rerank_count\":" << header.ack.search_rerank_count
        << ",\"search_adc_table_count\":" << header.ack.search_adc_table_count
        << ",\"control_work_us\":" << result.work_us
        << ",\"structure_sections\":" << header.sections
        << ",\"structure_bytes\":" << (header.hist_words * 2 + header.nbwork_words * 8);
    snapshot_event(r, "snapshot_received", fields.str());
}
bool current_frame(WindowControllerRuntime& r, const void* data, size_t size) {
    if (r.exchange != WindowExchangeKind::HARVEST) return false;
    if (size >= sizeof(ann::StatsWindowAck)) {
        ann::StatsWindowAck ack; std::memcpy(&ack, data, sizeof(ack));
        if (ack.window_epoch != r.window_epoch) return false;
    }
    return true;
}
// node count of the window, fixed at controller-start
uint32_t window_nodes(const WindowControllerRuntime& r) { return r.nodes; }
}
bool window_snapshot_offer(WindowControllerRuntime& r, uint32_t server, const void* data, size_t size) {
    if (!window_snapshot_is_response(data, size)) return false;
    if (!current_frame(r, data, size)) return true;
    const auto* begin = static_cast<const char*>(data);
    const bool duplicate = server < r.snapshot_digests.size() && !r.snapshot_digests[server].empty();
    auto result = decode_frame(server, r.window_epoch, window_nodes(r), r.config.ranks,
        std::vector<char>(begin, begin + size), duplicate, std::make_shared<WindowControlGeneration>(),
        window_repair_structure_sections(r.config.repair));
    if (!result.valid) {
        r.exchange_rejected = true;
        snapshot_event(r, "snapshot_rejected", "\"server_rank\":" + std::to_string(server) +
                       ",\"reason\":" + json_text(result.error));
        return true;
    }
    apply_receipt(r, result); return true;
}
bool window_snapshot_offer_owned(WindowControllerRuntime& r, uint32_t server, std::vector<char> bytes) {
    if (!window_snapshot_is_response(bytes.data(), bytes.size())) return false;
    if (!current_frame(r, bytes.data(), bytes.size())) return true;
    if (!r.native_control) window_native_arm(r);
    auto& work = *r.native_control;
    if (server >= r.config.ranks || work.snapshots.size() >= 2 * size_t(r.config.ranks)) {
        r.exchange_rejected = true; return true;
    }
    const bool duplicate = work.arrived[server];
    auto job = work.worker.submit([server, epoch=r.window_epoch, nodes=window_nodes(r), owners=r.config.ranks,
            generation=work.generation, bytes=std::move(bytes), duplicate,
            sections=window_repair_structure_sections(r.config.repair)] {
        return decode_frame(server, epoch, nodes, owners, bytes, duplicate, generation, sections);
    });
    if (!job.valid()) { r.exchange_rejected = true; return true; }
    work.snapshots.push_back(std::move(job));
    if (!duplicate) { work.arrived[server] = 1; ++work.arrival_count; }
    snapshot_event(r, "snapshot_frame_arrived", "\"server_rank\":" + std::to_string(server) +
        ",\"duplicate\":" + (duplicate ? "true" : "false"));
    return true;
}
void window_native_poll_snapshots(WindowControllerRuntime& r) {
    if (!r.native_control) return;
    auto& work = *r.native_control;
    if (work.epoch != r.window_epoch || r.exchange != WindowExchangeKind::HARVEST) return;
    for (auto& job : work.snapshots) {
        if (!job.valid()) continue;
        if (!window_control_ready(job)) break;  // keep arrival order
        try {
            auto result = job.get();
            if (!result.valid) {
                r.exchange_rejected = true;
                snapshot_event(r, "snapshot_rejected", "\"server_rank\":" + std::to_string(result.server) +
                               ",\"reason\":" + json_text(result.error));
            } else apply_receipt(r, result);
        } catch (const std::exception&) { r.exchange_rejected = true; }
    }
}

bool WindowStopCollector::offer(uint32_t server, const void* data, size_t size) {
    if (!data || size != sizeof(WindowStopAck)) { rejected = true; return false; }
    WindowStopAck ack; std::memcpy(&ack, data, sizeof(ack));
    if (server >= ranks || ack.rank != server || ack.msg_type != ann::MessageType::STATS_WINDOW_STOP_ACK ||
        ack.version != 1 || ack.status || ack.active || ack.requested_epoch != epoch || ack.actual_epoch > epoch) {
        rejected = true; return false;
    }
    for (const auto& previous : receipts) if (previous.rank == server) {
        if (std::memcmp(&previous, &ack, sizeof(ack))) rejected = true;
        return !rejected;
    }
    receipts.push_back(ack);
    return !rejected;
}

std::string WindowStopCollector::json() const {
    std::ostringstream out;
    out << ",\"stats_off_confirmed\":" << (complete() ? "true" : "false")
        << ",\"stats_off_epoch\":" << epoch << ",\"stats_off_ranks\":[";
    bool first = true;
    for (uint32_t rank = 0; rank < ranks; ++rank) for (const auto& ack : receipts) if (ack.rank == rank) {
        if (!first) out << ','; first = false; out << rank;
    }
    out << "],\"stats_off_acks\":["; first = true;
    for (uint32_t rank = 0; rank < ranks; ++rank) for (const auto& ack : receipts) if (ack.rank == rank) {
        if (!first) out << ','; first = false;
        out << "{\"rank\":" << rank << ",\"actual_epoch\":" << ack.actual_epoch << ",\"active\":false}";
    }
    return out.str() + "]";
}
} // namespace tide
