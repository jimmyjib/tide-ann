// Compare the planner (K = 4, weight wnb) with testdata/repair_golden_k4.json (--write regenerates it).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "window_repair_planner.h"

namespace {
using namespace tide;

constexpr uint32_t kOwners = 4;
// The fixed repair token, parsed by the production parser.
constexpr const char* kRepairToken =
    "algo:local,refine:0.03,balance:0.02,tau:0.3,eps:0.03,skew:1.02,cooldown:1,cum:1.0,"
    "slack:0.10,hop_slack:0.05,ema:0.5,converge:3,guard:0.10";
// The default weight is W (pred = w / Q). This test uses W^nb, added only when parsing, so the token
// written into the golden file has no weight key.
constexpr const char* kWeightKey = "weight:wnb";

[[noreturn]] void fail(const std::string& why) {
    std::fprintf(stderr, "GOLDEN TEST FAILED: %s\n", why.c_str());
    std::abort();
}
void require(bool ok, const std::string& why) {
    if (!ok) fail(why);
}

// splitmix64: fully specified, so the inputs are identical on every platform.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t below(uint32_t n) { return uint32_t(next() % n); }
};

enum class Heat { HotOwner0, HotOwners12Rotating3, Uniform };

struct Spec {
    const char* name;
    uint64_t seed;
    uint32_t nodes, clusters;            // contiguous clusters, cluster c on home owner c % 4
    uint32_t deg_min, deg_span;          // out-degree in [deg_min, deg_min + deg_span)
    uint32_t primary_pct, secondary_pct; // out-neighbors in the own / one other-home cluster, the rest uniform
    uint32_t misplaced_per_mille;        // independent misplacement away from the home owner
    uint32_t swapped_pairs;              // symmetric swaps between home owners (owner sizes unchanged)
    uint64_t queries;                    // window queries Q, a power of two
    std::array<double, kOwners> price;   // fixed per-rank prices, dyadic
    uint32_t windows;
    Heat heat;
    uint32_t work_base, work_span, zero_work_pct;
    bool expect_balance;                 // true: refine and balance both move in window 1
                                         // false: balance never moves in any window
};

const Spec kSpecs[] = {
    // Owner 0 carries three times the work: refine repairs misplaced rows, balance sheds owner 0.
    {"A-hot-owner0", 101, 5000, 40, 12, 13, 60, 35, 60, 0, 1024, {1.0, 1.0, 1.0, 1.0}, 4,
     Heat::HotOwner0, 8, 9, 10, true},
    // Unequal prices, owners 1 and 2 hot, one very hot owner-3 cluster that moves every window.
    {"B-priced-hot-owners12-rotating3", 202, 5000, 40, 10, 21, 55, 35, 40, 0, 2048, {1.0, 0.75, 1.25, 1.5}, 4,
     Heat::HotOwners12Rotating3, 6, 11, 15, true},
    // Uniform work and degree, symmetric swaps: refine undoes the swaps, skew stays below the
    // trigger, so balance never runs. The quiet windows then converge the phase.
    {"C-balanced-swaps", 303, 5000, 40, 16, 1, 75, 25, 0, 40, 1024, {1.0, 1.0, 1.0, 1.0}, 5,
     Heat::Uniform, 10, 1, 0, false},
};

uint32_t home_of_cluster(uint32_t cluster) { return cluster % kOwners; }

struct Graph {
    std::vector<std::vector<uint32_t>> adj;
    std::vector<uint32_t> cluster, home, membership;
    std::vector<uint64_t> row_bytes;  // 4 + 4 * out-degree
};

Graph build_graph(const Spec& s) {
    const uint32_t n = s.nodes, per = s.nodes / s.clusters;
    require(per * s.clusters == n && s.clusters % kOwners == 0 && s.deg_span > 0 && s.deg_min > 0, "spec shape");
    Rng rng(s.seed);
    Graph g;
    g.adj.resize(n);
    g.cluster.resize(n);
    g.home.resize(n);
    g.row_bytes.resize(n);
    for (uint32_t v = 0; v < n; ++v) {
        g.cluster[v] = v / per;
        g.home[v] = home_of_cluster(g.cluster[v]);
    }
    auto add_unique = [](std::vector<uint32_t>& row, uint32_t v, uint32_t u) {
        if (u == v) return;
        for (const uint32_t x : row) if (x == u) return;
        row.push_back(u);
    };
    for (uint32_t v = 0; v < n; ++v) {
        const uint32_t deg = s.deg_min + rng.below(s.deg_span);
        const uint32_t primary = deg * s.primary_pct / 100, secondary = deg * s.secondary_pct / 100;
        require(primary + secondary <= deg && primary < per && secondary < per, "spec degree mix");
        const uint32_t c = g.cluster[v];
        uint32_t sc = 0;
        do sc = rng.below(s.clusters); while (home_of_cluster(sc) == home_of_cluster(c));
        auto& row = g.adj[v];
        while (row.size() < primary) add_unique(row, v, c * per + rng.below(per));
        while (row.size() < primary + secondary) add_unique(row, v, sc * per + rng.below(per));
        while (row.size() < deg) add_unique(row, v, rng.below(n));
        g.row_bytes[v] = 4 + 4 * uint64_t(deg);
    }
    g.membership = g.home;
    if (s.misplaced_per_mille)
        for (uint32_t v = 0; v < n; ++v)
            if (rng.below(1000) < s.misplaced_per_mille) g.membership[v] = (g.home[v] + 1 + rng.below(3)) % kOwners;
    for (uint32_t i = 0; i < s.swapped_pairs; ++i) {
        uint32_t a = 0, b = 0;
        do a = rng.below(n); while (g.membership[a] != g.home[a]);
        do b = rng.below(n); while (g.membership[b] != g.home[b] || g.home[b] == g.home[a]);
        g.membership[a] = g.home[b];
        g.membership[b] = g.home[a];
    }
    return g;
}

uint32_t heat_of(const Spec& s, uint32_t cluster, uint32_t window_index) {
    switch (s.heat) {
        case Heat::HotOwner0:
            return home_of_cluster(cluster) == 0 ? 3 : 1;
        case Heat::HotOwners12Rotating3: {
            if (cluster == (3 + 4 * window_index) % s.clusters) return 6;  // an owner-3 cluster
            const uint32_t h = home_of_cluster(cluster);
            return h == 1 || h == 2 ? 2 : 1;
        }
        case Heat::Uniform:
            return 1;
    }
    return 1;
}

std::vector<uint64_t> window_work(const Spec& s, const Graph& g, uint32_t window_index) {
    Rng rng(s.seed ^ (0xD1B54A32D192ED03ull * (uint64_t(window_index) + 1)));
    std::vector<uint64_t> work(s.nodes);
    for (uint32_t v = 0; v < s.nodes; ++v) {
        const bool zero = s.zero_work_pct && rng.below(100) < s.zero_work_pct;
        const uint64_t base = s.work_base + (s.work_span > 1 ? rng.below(s.work_span) : 0);
        work[v] = zero ? 0 : heat_of(s, g.cluster[v], window_index) * base;
    }
    return work;
}

// The servers' structure capture on the full graph: hist[u][p] counts u's
// out-neighbors owned by p, and nbwork[v] sums the window work of v's in-neighbors.
void capture_structure(const Graph& g, const std::vector<uint32_t>& membership, const std::vector<uint64_t>& work,
                       std::vector<uint32_t>& hist, std::vector<uint64_t>& nb_work) {
    const size_t n = membership.size();
    hist.assign(n * kOwners, 0);
    nb_work.assign(n, 0);
    for (size_t u = 0; u < n; ++u)
        for (const uint32_t v : g.adj[u]) {
            nb_work[v] += work[u];
            ++hist[u * kOwners + membership[v]];
        }
}

std::string num(double value) {
    require(std::isfinite(value), "golden numbers are finite");
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", value);
    return buf;
}
std::string u64(uint64_t value) { return std::to_string(value); }
template <class Array> std::string loads(const Array& W) {
    std::string out = "[";
    for (uint32_t p = 0; p < kOwners; ++p) out += (p ? "," : "") + num(W[p]);
    return out + "]";
}

struct InputSummary {
    std::vector<uint64_t> refine_moved, balance_moved;
    std::vector<bool> balance_triggered;
};

InputSummary run_input(const Spec& s, const WindowRepairConfig& config, std::string& out) {
    const Graph g = build_graph(s);
    std::vector<uint32_t> membership = g.membership;
    uint64_t total_bytes = 0;
    for (const uint64_t b : g.row_bytes) total_bytes += b;
    const std::vector<double> price(s.price.begin(), s.price.end());
    std::array<uint64_t, kOwners> owner_nodes{};
    uint64_t misplaced = 0;
    for (uint32_t v = 0; v < s.nodes; ++v) {
        ++owner_nodes[membership[v]];
        misplaced += membership[v] != g.home[v];
    }
    // The controller runtime's start state: no smoothed weights, no cooldowns, IDLE.
    WindowRepairState state;
    state.price = price;
    state.price_frozen = true;
    state.cooldown_until.assign(s.nodes, 0);

    out += "{\"name\":\"" + std::string(s.name) + "\",\"seed\":" + u64(s.seed) + ",\"nodes\":" + u64(s.nodes) +
        ",\"clusters\":" + u64(s.clusters) + ",\"queries\":" + u64(s.queries) + ",\"price\":" + loads(s.price) +
        ",\"total_bytes\":" + u64(total_bytes) + ",\"initial_owner_nodes\":[" + u64(owner_nodes[0]) + "," +
        u64(owner_nodes[1]) + "," + u64(owner_nodes[2]) + "," + u64(owner_nodes[3]) + "],\"initial_misplaced\":" +
        u64(misplaced) + ",\n\"windows\":[\n";

    InputSummary summary;
    for (uint32_t r = 0; r < s.windows; ++r) {
        const std::vector<uint64_t> work = window_work(s, g, r);
        std::vector<uint32_t> hist;
        std::vector<uint64_t> nb_work;
        capture_structure(g, membership, work, hist, nb_work);
        const uint32_t window = state.windows + 1;
        const WindowRepairPhase phase_before = state.phase;
        const WindowRepairPlan plan = window_repair_plan(config, kOwners, state.phase, window, s.queries, total_bytes, membership,
                                                         g.row_bytes, work, nb_work, hist, price, state.pred,
                                                         state.cooldown_until);
        const std::string where = std::string(s.name) + " window " + std::to_string(window) + ": ";
        require(plan.valid, where + "plan invalid: " + plan.error);
        const WindowRepairBurstStats& st = plan.stats;
        require(plan.actions.size() == st.refine_moved + st.balance_moved, where + "actions = refine + balance moves");
        require(plan.planned || plan.actions.empty(), where + "an unplanned window has no actions");
        require(plan.pred.size() == s.nodes && plan.cooldown_until.size() == s.nodes, where + "state shapes");

        // Every action is applied in order to a private copy: each node moves at most once, from its
        // current owner, and carries its row bytes. The first refine_moved actions are refine's.
        std::vector<uint32_t> after = membership;
        std::vector<uint8_t> seen(s.nodes, 0);
        uint64_t refine_bytes = 0, balance_bytes = 0;
        for (size_t i = 0; i < plan.actions.size(); ++i) {
            const WindowBurstAction& a = plan.actions[i];
            require(a.node < s.nodes && !seen[a.node], where + "a node moves at most once per window");
            seen[a.node] = 1;
            require(a.source_owner == after[a.node] && a.target_owner < kOwners && a.target_owner != a.source_owner,
                    where + "action endpoints");
            require(a.adjacency_bytes == g.row_bytes[a.node], where + "action bytes are the row bytes");
            require(plan.cooldown_until[a.node] == window + config.cooldown_windows, where + "a moved node cools down");
            (i < st.refine_moved ? refine_bytes : balance_bytes) += a.adjacency_bytes;
            after[a.node] = a.target_owner;
        }
        require(refine_bytes == st.refine_bytes && balance_bytes == st.balance_bytes, where + "stats bytes");
        require(refine_bytes <= st.refine_budget_bytes, where + "refine stays within its budget");

        // The predicted per-owner loads equal the fixed-price sums of pred over the new membership,
        // exactly (all loads are dyadic, see the header).
        std::array<double, kOwners> sums{};
        for (uint32_t v = 0; v < s.nodes; ++v) sums[after[v]] += plan.pred[v];
        for (uint32_t p = 0; p < kOwners; ++p)
            require(price[p] * sums[p] == plan.W[p], where + "predicted W is exact for owner " + std::to_string(p));

        state.pred = plan.pred;
        state.cooldown_until = plan.cooldown_until;
        const WindowRepairTransition transition =
            window_repair_account_window(config, state, !plan.actions.empty(), plan.calibration.skew);
        membership = after;

        out += std::string(r ? ",\n" : "") + "{\"window\":" + u64(window) + ",\"phase_before\":\"" +
            window_repair_phase_name(phase_before) + "\",\"planned\":" + (plan.planned ? "true" : "false") +
            ",\"calibrated_W\":" + loads(plan.calibration.W) + ",\"calibrated_skew\":" + num(plan.calibration.skew) +
            ",\n \"refine\":{\"candidates\":" + u64(st.refine_candidates) + ",\"moved\":" + u64(st.refine_moved) +
            ",\"bytes\":" + u64(st.refine_bytes) + ",\"rejected_balance\":" + u64(st.rejected_balance) +
            ",\"budget_bytes\":" + u64(st.refine_budget_bytes) + "},\"balance\":{\"triggered\":" +
            (st.balance_triggered ? "true" : "false") + ",\"moved\":" + u64(st.balance_moved) +
            ",\"bytes\":" + u64(st.balance_bytes) + ",\"budget_bytes\":" + u64(st.balance_budget_bytes) +
            "},\"skew_before\":" + num(st.skew_before) + ",\"skew_after\":" + num(st.skew_after) +
            ",\n \"predicted_W\":" + loads(plan.W) + ",\"phase_after\":\"" + window_repair_phase_name(state.phase) +
            "\",\"quiet_windows\":" + u64(state.quiet_windows) + ",\"converged\":" +
            (transition.converged ? "true" : "false") + ",\"reactivated\":" + (transition.reactivated ? "true" : "false") +
            ",\n \"actions\":[";
        for (size_t i = 0; i < plan.actions.size(); ++i) {
            const WindowBurstAction& a = plan.actions[i];
            out += std::string(i ? "," : "") + "\n  [" + u64(a.node) + "," + u64(a.source_owner) + "," +
                u64(a.target_owner) + ",\"" + (i < st.refine_moved ? "refine" : "balance") + "\"," +
                u64(a.adjacency_bytes) + "]";
        }
        out += "]}";
        summary.refine_moved.push_back(st.refine_moved);
        summary.balance_moved.push_back(st.balance_moved);
        summary.balance_triggered.push_back(st.balance_triggered);
        std::printf("%-32s window %u phase %-9s -> %-9s skew %.6f -> %.6f refine %3llu (cand %4llu, rej %3llu) "
                    "balance %3llu%s\n",
                    s.name, window, window_repair_phase_name(phase_before), window_repair_phase_name(state.phase),
                    st.skew_before, st.skew_after, (unsigned long long)st.refine_moved,
                    (unsigned long long)st.refine_candidates, (unsigned long long)st.rejected_balance,
                    (unsigned long long)st.balance_moved, st.balance_triggered ? " (triggered)" : "");
    }
    out += "\n]}";
    return summary;
}

std::string generate(const WindowRepairConfig& config) {
    std::string out = std::string("{\"owners\":4,\"weight\":\"work+nbwork\",") +
        "\"token\":\"" + kRepairToken + "\",\n\"inputs\":[\n";
    bool first = true;
    for (const Spec& s : kSpecs) {
        if (!first) out += ",\n";
        first = false;
        const InputSummary summary = run_input(s, config, out);
        require(summary.refine_moved.size() >= 3, std::string(s.name) + ": at least three planning windows");
        if (s.expect_balance) {
            require(summary.refine_moved[0] > 0 && summary.balance_moved[0] > 0,
                    std::string(s.name) + ": refine and balance both move in window 1");
        } else {
            require(summary.refine_moved[0] > 0, std::string(s.name) + ": refine moves in window 1");
            for (size_t i = 0; i < summary.balance_moved.size(); ++i)
                require(summary.balance_moved[i] == 0 && !summary.balance_triggered[i],
                        std::string(s.name) + ": balance never runs");
        }
    }
    return out + "\n]}\n";
}

uint64_t fnv1a(const std::string& text) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const unsigned char c : text) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        if (end == std::string::npos) { lines.push_back(text.substr(start)); break; }
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return lines;
}
}  // namespace

int main(int argc, char** argv) {
    bool write = false;
    std::string path;
    if (argc == 3 && std::strcmp(argv[1], "--write") == 0) {
        write = true;
        path = argv[2];
    } else if (argc == 2 && std::strcmp(argv[1], "--write") != 0) {
        path = argv[1];
    } else {
        std::fprintf(stderr, "usage: %s <golden.json> | --write <golden.json>\n", argv[0]);
        return 2;
    }

    WindowRepairConfig config;
    std::string error;
    require(window_repair_parse_token(std::string(kRepairToken) + "," + kWeightKey, &config, &error),
            "repair token: " + error);
    require(config.enabled, "the repair token enables the planner");
    require(config.weight == WindowRepairWeight::WNB, "the golden file uses weight W^nb");
    {
        WindowRepairConfig default_weight;
        require(window_repair_parse_token(kRepairToken, &default_weight, &error) &&
                    default_weight.weight == WindowRepairWeight::W,
                "without the weight key the token parses to weight W");
    }
    require(config.refine_budget == 0.03 && config.balance_budget == 0.02 && config.tau == 0.3 && config.eps == 0.03 &&
                config.skew_trigger == 1.02 && config.cooldown_windows == 1 && config.cumulative_cap == 1.0 &&
                config.slack == 0.10 && config.hop_slack == 0.05 && config.ema == 0.5 && config.converge_windows == 3 &&
                config.guard == 0.10,
            "the parsed token carries the reference configuration");

    const std::string text = generate(config);
    std::printf("golden text: %zu bytes, %zu lines, fnv1a64 %016llx\n", text.size(), lines_of(text).size(),
                (unsigned long long)fnv1a(text));

    if (write) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(text.data(), std::streamsize(text.size()));
        file.close();
        require(file.good(), "cannot write " + path);
        std::printf("WROTE %s\n", path.c_str());
        return 0;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::fprintf(stderr, "GOLDEN TEST FAILED: cannot read %s\n", path.c_str());
        return 1;
    }
    const std::string golden((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (golden == text) {
        std::printf("GOLDEN MATCH %s\n", path.c_str());
        return 0;
    }
    const std::vector<std::string> want = lines_of(golden), got = lines_of(text);
    size_t differing = 0, first = SIZE_MAX;
    for (size_t i = 0; i < std::max(want.size(), got.size()); ++i) {
        const bool same = i < want.size() && i < got.size() && want[i] == got[i];
        if (!same) {
            ++differing;
            if (first == SIZE_MAX) first = i;
        }
    }
    std::fprintf(stderr, "GOLDEN TEST FAILED: %s differs (%zu vs %zu bytes, %zu vs %zu lines, %zu differing lines)\n",
                 path.c_str(), golden.size(), text.size(), want.size(), got.size(), differing);
    if (first != SIZE_MAX)
        std::fprintf(stderr, "first difference at line %zu\n  golden: %s\n  actual: %s\n", first + 1,
                     first < want.size() ? want[first].c_str() : "<end of file>",
                     first < got.size() ? got[first].c_str() : "<end of output>");
    return 1;
}
