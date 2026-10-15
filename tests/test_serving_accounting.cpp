// Whole-serving accounting of a search command (ann::ServingAccounting) and
// its JSON: one steady-clock envelope containing the materialization and
// head-BF spans, in order, with every declared query completed.
#include "metrics/json_output.h"

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

static bool has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// The whole serving-accounting object for the test's inputs: every key, the key order and every value, so a key
// added to or dropped from the block fails here.
static std::string expected_accounting(const std::string& status, const std::string& error, const std::string& end_ns,
                                       const std::string& raw) {
    return "{\"schema\":\"tide-serving-accounting-v2\",\"clock_scope\":\"coordinator-steady-ns\""
           ",\"status\":\"" + status + "\",\"error\":\"" + error + "\""
           ",\"begin_scope\":\"after-GT-load-before-query-and-context-preparation\""
           ",\"end_scope\":\"last-owner-search-completion\""
           ",\"owner_search_begin_scope\":\"first-owner-query-init-before-serialization-and-enqueue-not-transport-dispatch\""
           ",\"included\":[\"query-context-preparation\",\"workload-materialization\",\"parallel-head-BF-wall-once\","
           "\"native-search-and-controller-work\",\"rolling-Recall\",\"in-envelope-instrumentation\"]"
           ",\"outside_envelope\":[\"GT-load\",\"previous-resident-payload-load\",\"external-client-and-controller-startup\","
           "\"post-completion-controller-stop\",\"metric-thread-join\",\"final-Recall-digest-report\"]"
           ",\"declared_queries\":80,\"completed_queries\":80"
           ",\"begin_ns\":1000000,\"end_ns\":" + end_ns + ",\"owner_search_begin_ns\":1800000"
           ",\"materialize\":{\"begin_ns\":1300000,\"end_ns\":1400000}"
           ",\"head_bf\":{\"begin_ns\":1400000,\"end_ns\":1700000}" + raw + "}";
}

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

int main() {
    ann::ServingAccounting a;
    a.enabled = true;
    a.begin_ns = 1000000; a.end_ns = 2000000; a.owner_search_begin_ns = 1800000;
    a.declared_queries = a.completed_queries = 80;
    a.materialize = {1300000, 1400000};
    a.head_bf = {1400000, 1700000};
    assert(a.error().empty());
    assert(a.raw_ns() == 1000000);

    const auto json = ann::serving_accounting_json(a);
    assert(json == expected_accounting("VALID", "", "2000000", ",\"raw_elapsed_ns\":1000000,\"raw_envelope_qps\":80000"));

    auto bad = a; bad.head_bf = {};
    assert(!bad.error().empty());  // a required span is missing
    bad = a; bad.materialize.begin_ns = 1450000; bad.materialize.end_ns = 1500000;
    assert(!bad.error().empty());  // overlap: spans are ordered, never double counted
    bad = a; bad.head_bf.end_ns = 1900000;
    assert(!bad.error().empty());  // preparation after the first owner query-init
    bad = a; bad.head_bf = {1700000, 1400000};
    assert(!bad.error().empty());  // reversed span
    bad = a; bad.owner_search_begin_ns = 900000;
    assert(bad.error() == "missing_or_unordered_envelope");
    bad = a; bad.enabled = false;
    assert(bad.error() == "not_search_command");
    bad = a; bad.completed_queries = 60;
    assert(bad.error() == "unproven_completed_query_scope");  // a partial command is never accounted
    bad = a; bad.end_ns = 0;
    const auto invalid = ann::serving_accounting_json(bad);
    assert(invalid == expected_accounting("INVALID", "missing_or_unordered_envelope", "0",
                                          ",\"raw_elapsed_ns\":null,\"raw_envelope_qps\":null"));

    assert(ann::ServingAccounting::ns_at(std::chrono::steady_clock::time_point{}) == 0);
    assert(ann::ServingAccounting::now_ns() > 0);

    // to_json embeds the block only for search commands and keeps the native qps field. The top-level key list is
    // pinned for both results, so a field added to or dropped from the result JSON fails here.
    const std::vector<std::string> common = {
        "timestamp", "dataset", "groundtruth_file", "mode", "p2p_mode", "observability", "parameters", "search",
        "hop_distribution", "latency", "query_latency", "per_query_breakdown", "per_hop_breakdown", "hop_rtt",
        "server_timing_us", "coord_timing_per_hop_us", "coord_timing_totals", "bottleneck", "expansion", "distance",
        "per_hop_distance", "server_load", "lock_metrics", "insert", "unified_timing", "resize", "merge_search"};
    ann::ExperimentResult r; r.qps = 123; r.serving_accounting = a;
    const auto result = ann::to_json(r);
    assert(has(result, "  \"serving_accounting\": " + json + ",\n"));
    assert(has(result, "\"qps\": 123.0000"));
    std::vector<std::string> with_block = {"serving_accounting"};
    with_block.insert(with_block.end(), common.begin(), common.end());
    assert(top_level_keys(result) == with_block);
    ann::ExperimentResult plain; plain.qps = 7;
    const auto plain_json = ann::to_json(plain);
    assert(!has(plain_json, "\"serving_accounting\""));
    assert(has(plain_json, "\"qps\": 7.0000"));
    assert(top_level_keys(plain_json) == common);
    return 0;
}
