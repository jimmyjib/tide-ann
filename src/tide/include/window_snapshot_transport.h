#pragma once
#include "window_controller_runtime.h"
#include "window_snapshot_wire.h"

namespace tide {
bool window_snapshot_is_response(const void* data, size_t size);
bool window_snapshot_offer(WindowControllerRuntime& runtime, uint32_t server,
                           const void* data, size_t size);
// collects one STOP result from each of `ranks` servers
struct WindowStopCollector {
    uint64_t epoch = 0;
    uint32_t ranks = 0;
    bool rejected = false;
    std::vector<WindowStopAck> receipts;
    WindowStopCollector(uint64_t expected, uint32_t server_count) : epoch(expected), ranks(server_count) {}
    bool offer(uint32_t server, const void* data, size_t size);
    bool complete() const { return !rejected && ranks && receipts.size() == ranks; }
    std::string json() const;
};
} // namespace tide
