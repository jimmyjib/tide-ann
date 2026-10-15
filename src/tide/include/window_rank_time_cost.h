#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace tide {
// One immutable BEGIN_ACK-to-HARVEST_ACK window observation per rank: D distances,
// R requests, Td_us distance time and Ttot_us total search time. Prices have units
// us/distance and us/request; all rank costs have units us/completed-query.
// Zero denominators (including 0/0), missing ranks, negative residuals and
// nonfinite prices are undefined: no common-price or neighboring-rank fallback.
struct WindowRankTimeRow {
    uint64_t D = 0, R = 0, Td_us = 0, Ttot_us = 0;
    double c_d_us = std::numeric_limits<double>::quiet_NaN();
    double c_r_us = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};
struct WindowRankTimeContext {
    uint64_t Q = 0;
    std::vector<WindowRankTimeRow> ranks;  // one row per server rank (K rows)
    bool valid = false;
    std::string reason = "missing-rank-time-context";
};

inline bool window_rank_time_valid(const WindowRankTimeContext& c) {
    if (!c.valid || !c.Q || c.ranks.empty()) return false;
    for (const auto& r : c.ranks) {
        if (!r.valid || !r.D || !r.R || r.Ttot_us < r.Td_us ||
            !std::isfinite(r.c_d_us) || r.c_d_us < 0 ||
            !std::isfinite(r.c_r_us) || r.c_r_us < 0 ||
            r.c_d_us != double(r.Td_us) / r.D ||
            r.c_r_us != double(r.Ttot_us - r.Td_us) / r.R) return false;
    }
    return true;
}

inline bool window_rank_time_seal(WindowRankTimeContext* c) {
    if (!c) return false;
    c->valid = false;
    c->reason = c->Q ? (c->ranks.empty() ? "no-ranks" : "") : "zero-query-denominator";
    for (size_t rank = 0; rank < c->ranks.size(); ++rank) {
        auto& r = c->ranks[rank];
        r.valid = r.D && r.R && r.Ttot_us >= r.Td_us;
        r.c_d_us = r.D ? double(r.Td_us) / r.D : std::numeric_limits<double>::quiet_NaN();
        r.c_r_us = r.R && r.Ttot_us >= r.Td_us ? double(r.Ttot_us - r.Td_us) / r.R
            : std::numeric_limits<double>::quiet_NaN();
        r.valid = r.valid && std::isfinite(r.c_d_us) && std::isfinite(r.c_r_us);
        if (!r.valid && c->reason.empty()) c->reason = "rank-" + std::to_string(rank) +
            (!r.D ? "-zero-D" : !r.R ? "-zero-R" : r.Ttot_us < r.Td_us ? "-negative-residual" : "-nonfinite-price");
    }
    c->valid = c->reason.empty();
    return window_rank_time_valid(*c);
}

inline std::string window_rank_time_json(const WindowRankTimeContext& c) {
    std::ostringstream out; out << std::setprecision(17);
    out << "{\"Q\":" << c.Q << ",\"valid\":" << (window_rank_time_valid(c) ? "true" : "false")
        << ",\"reason\":\"" << c.reason << "\",\"ranks\":[";
    for (size_t rank = 0; rank < c.ranks.size(); ++rank) {
        const auto& r = c.ranks[rank]; if (rank) out << ',';
        out << "{\"rank\":" << rank << ",\"D\":" << r.D << ",\"R\":" << r.R
            << ",\"Td_us\":" << r.Td_us << ",\"Ttot_us\":" << r.Ttot_us << ",\"c_d_us\":";
        if (std::isfinite(r.c_d_us)) out << r.c_d_us; else out << "null";
        out << ",\"c_r_us\":";
        if (std::isfinite(r.c_r_us)) out << r.c_r_us; else out << "null";
        out << ",\"valid\":" << (r.valid ? "true" : "false") << '}';
    }
    return out.str() + "]}";
}
} // namespace tide
