// The serving object of a recall result.
#include "metrics/json_output.h"

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

static bool has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

static const char* kServing =
    "{\n"
    "    \"elapsed_ms\": 1.0000,\n"
    "    \"qps\": 80000.0000,\n"
    "    \"workload_setup_ms\": 0.1000,\n"
    "    \"head_index_ms\": 0.3000\n"
    "  }";

static const char* kIncomplete =
    "{\n"
    "    \"elapsed_ms\": null,\n"
    "    \"qps\": null,\n"
    "    \"workload_setup_ms\": null,\n"
    "    \"head_index_ms\": null\n"
    "  }";

// The top-level keys of a to_json result, in order: the lines that start with two spaces and a quote.
static std::vector<std::string> top_level_keys(const std::string& json) {
    std::vector<std::string> keys;
    std::istringstream lines(json);
    for (std::string line; std::getline(lines, line);) {
        if (line.rfind("  \"", 0) != 0) continue;
        const auto end = line.find('"', 3);
        assert(end != std::string::npos);
        keys.push_back(line.substr(3, end - 3));
    }
    return keys;
}

static void expect_incomplete(const ann::ServingAccounting& a) {
    assert(!a.complete());
    assert(ann::serving_json(a) == kIncomplete);
}

int main() {
    ann::ServingAccounting a;
    a.enabled = true;
    a.begin_ns = 1000000; a.end_ns = 2000000; a.owner_search_begin_ns = 1800000;
    a.declared_queries = a.completed_queries = 80;
    a.materialize = {1300000, 1400000};
    a.head_bf = {1400000, 1700000};
    assert(a.complete());
    assert(a.elapsed_ns() == 1000000);
    const auto json = ann::serving_json(a);
    assert(json == kServing);

    auto bad = a; bad.head_bf = {};
    expect_incomplete(bad);  // missing span
    bad = a; bad.materialize.begin_ns = 1450000; bad.materialize.end_ns = 1500000;
    expect_incomplete(bad);  // overlapping spans
    bad = a; bad.head_bf.end_ns = 1900000;
    expect_incomplete(bad);  // span ends after the first search starts
    bad = a; bad.head_bf = {1700000, 1400000};
    expect_incomplete(bad);  // reversed span
    bad = a; bad.owner_search_begin_ns = 900000;
    expect_incomplete(bad);  // first search before the interval begins
    bad = a; bad.enabled = false;
    expect_incomplete(bad);
    bad = a; bad.completed_queries = 60;
    expect_incomplete(bad);
    bad = a; bad.end_ns = 0;
    expect_incomplete(bad);

    assert(ann::ServingAccounting::ns_at(std::chrono::steady_clock::time_point{}) == 0);
    assert(ann::ServingAccounting::now_ns() > 0);

    // The top-level keys of a recall result and of a result without the serving object, in order.
    const std::vector<std::string> common = {
        "timestamp", "dataset", "groundtruth_file", "mode", "p2p_mode", "observability", "parameters", "search",
        "hop_distribution", "latency", "query_latency", "per_query_breakdown", "per_hop_breakdown", "hop_rtt",
        "server_timing_us", "coord_timing_per_hop_us", "coord_timing_totals", "bottleneck", "expansion", "distance",
        "per_hop_distance", "server_load", "lock_metrics", "insert", "unified_timing", "resize", "merge_search"};
    ann::WorkloadResult r; r.qps = 123; r.serving = a;
    const auto result = ann::to_json(r);
    assert(has(result, "  },\n  \"serving\": " + json + ",\n"));
    assert(has(result, "\"qps\": 123.0000"));
    std::vector<std::string> with_serving = common;
    with_serving.insert(with_serving.begin() + 8, "serving");
    assert(top_level_keys(result) == with_serving);
    ann::WorkloadResult plain; plain.qps = 7;
    const auto plain_json = ann::to_json(plain);
    assert(!has(plain_json, "\"serving\""));
    assert(has(plain_json, "\"qps\": 7.0000"));
    assert(top_level_keys(plain_json) == common);
    // P2P: the route total stays, the split by kind and visit is null
    ann::WorkloadResult p2p; p2p.p2p_mode = true; p2p.avg_dist_comp = 18;
    const auto p2p_json = ann::to_json(p2p);
    assert(has(p2p_json, "\"avg_dist_comp\": 18.0000"));
    assert(has(p2p_json, "  \"distance\": null,\n  \"per_hop_distance\": null,\n"));
    assert(top_level_keys(p2p_json) == common);
    return 0;
}
