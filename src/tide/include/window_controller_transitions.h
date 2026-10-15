#pragma once
#include "window_native_control.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
namespace tide { namespace window_control_detail {
using Clock = std::chrono::steady_clock;
// BEGIN/HARVEST exchange deadline at K <= 4 servers, counted from arming the
// exchange (the tick journals "harvest_requested" with armed_mono_ns and
// deadline_s) to the merged structure. The frames of one harvest are decoded
// and merged in memory on one control worker, so the deadline grows linearly
// with K (window_controller_exchange_deadline_sec: 25 s x max(1, K / 4)).
constexpr double kExchangeDeadlineSec = 25.0;
// Pre-prepare drain timeout is recoverable; ownership has not changed.
constexpr double kFenceDrainDeadlineSec = 120.0;
constexpr double kFenceHeartbeatSec = 5.0;
inline double seconds_since(const Clock::time_point& start, const Clock::time_point& now) {
    return std::chrono::duration<double>(now - start).count();
}

inline std::string json_quote(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Flush every transition so a crash leaves the last decision readable. Every
// event carries "mono_ns" (coordinator steady clock), like the direct writers.
inline void journal_event(WindowControllerRuntime& runtime, const char* event, const std::string& fields) {
    runtime.journal << "{\"event\":\"" << event << "\",\"cycle\":" << runtime.cycles
                    << ",\"window_epoch\":" << runtime.window_epoch << ",\"commits\":" << runtime.commits
                    << ",\"moved_bytes\":" << runtime.moved_bytes
                    << ",\"mono_ns\":" << window_journal_now_ns();
    if (!fields.empty()) runtime.journal << "," << fields;
    runtime.journal << "}\n";
    runtime.journal.flush();
}

inline void enter_cooldown(WindowControllerRuntime& runtime, const Clock::time_point& now) {
    window_native_cancel(runtime);
    window_history_cancel(runtime, now);  // Only an uncommitted burst releases its reservation.
    runtime.pending_actions.clear();
    runtime.pending_plan.clear();
    runtime.pending_bytes = 0;
    runtime.reverting = false;
    runtime.state = WindowControllerState::COOLDOWN;
    runtime.cooldown_until = now + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(runtime.config.cooldown_seconds));
}

inline void stop_on_error(WindowControllerRuntime& runtime, const char* event,
                          const std::string& fields, bool ownership_ambiguous = false) {
    window_native_cancel(runtime);
    runtime.ownership_ambiguous |= ownership_ambiguous;
    journal_event(runtime, event,
                  "\"ownership_ambiguous\":" + std::string(runtime.ownership_ambiguous ? "true" : "false") +
                      (fields.empty() ? "" : "," + fields));
    runtime.stopped_on_error = true;
    runtime.proposals_disabled = true;
    runtime.terminal_reason = event;
    runtime.state = WindowControllerState::RECOVERY_FAILED;
}

inline std::string exact_number(double value) {
    if (!std::isfinite(value)) return "null";  // Missing measurement stays valid JSON, never an ACCEPT.
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return out.str();
}

inline std::string window_dump_dir(const WindowControllerRuntime& runtime) {
    return runtime.config.journal_dir + "/window-" + std::to_string(runtime.window_epoch);
}
} } // namespace tide::window_control_detail
