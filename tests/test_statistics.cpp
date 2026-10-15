// Nearest-rank percentile index.
#include "metrics/statistics.h"
#include "protocol/message_types.h"
#include "stats_response_collector.h"
#include "types/workload_types.h"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        // Nearest-rank p95 boundary cases. In particular, exact integer ranks
        // use rank - 1 rather than floor(0.95 * n).
        check(ann::nearest_rank_percentile_index(1, 95) == 0, "p95 n=1");
        check(ann::nearest_rank_percentile_index(2, 95) == 1, "p95 n=2");
        check(ann::nearest_rank_percentile_index(19, 95) == 18, "p95 n=19");
        check(ann::nearest_rank_percentile_index(20, 95) == 18, "p95 n=20");
        check(ann::nearest_rank_percentile_index(21, 95) == 19, "p95 n=21");
        check(ann::nearest_rank_percentile_index(100, 95) == 94, "p95 n=100");
        check(ann::nearest_rank_percentile_index(101, 95) == 95, "p95 n=101");

        bool rejected_empty = false;
        try {
            (void)ann::nearest_rank_percentile_index(0, 95);
        } catch (const std::invalid_argument&) {
            rejected_empty = true;
        }
        check(rejected_empty, "empty sample was accepted");

        // A consuming request (the admin stats command) takes and resets the
        // interval maxima. An observational read must leave them for the next one.
        std::atomic<uint64_t> interval_max{0};
        std::atomic<uint64_t> cumulative_sum{33};
        std::atomic<uint64_t> cumulative_count{3};
        ann::StatsRequest boundary_request{};
        boundary_request.msg_type = ann::MessageType::GET_STATS_REQUEST;
        boundary_request.consume_interval_max = 1;
        boundary_request.request_id = 2;
        ann::StatsRequest internal_request{};
        internal_request.msg_type = ann::MessageType::GET_STATS_REQUEST;
        internal_request.consume_interval_max = 0;
        internal_request.request_id = 1;

        // The first consuming snapshot takes every observation made before it.
        ann::update_interval_max(interval_max, 7);
        ann::update_interval_max(interval_max, 2);
        ann::update_interval_max(interval_max, 11);
        check(ann::snapshot_interval_max(
                  interval_max, boundary_request.consume_interval_max != 0) == 11,
              "first snapshot did not take the max");
        check(interval_max.load() == 0, "first snapshot did not reset the max");

        // An observational read sees the running max but does not take it, so
        // the next consuming snapshot takes the exact max of the interval.
        ann::update_interval_max(interval_max, 5);
        ann::update_interval_max(interval_max, 19);
        check(ann::snapshot_interval_max(
                  interval_max, internal_request.consume_interval_max != 0) == 19,
              "internal observation returned wrong max");
        check(interval_max.load() == 19, "internal observation consumed max");
        ann::update_interval_max(interval_max, 13);
        check(ann::snapshot_interval_max(
                  interval_max, boundary_request.consume_interval_max != 0) == 19,
              "next snapshot did not take the interval max");
        check(interval_max.load() == 0, "next snapshot did not reset the max");

        // Sum/count are absolute cumulative counters, independent of max mode.
        check(cumulative_sum.load() == 33, "cumulative sum changed");
        check(cumulative_count.load() == 3, "cumulative count changed");

        // Production sequence: an observational workload round may arrive
        // late, but cannot satisfy the next authoritative round. Responses
        // may arrive out of rank order and still form an exact four-rank set.
        auto response = [](uint32_t rank, uint64_t request_id) {
            ann::StatsResponseHeader value{};
            value.msg_type = ann::MessageType::GET_STATS_RESPONSE;
            value.partition_id = rank;
            value.request_id = request_id;
            return value;
        };
        tide::StatsResponseCollector observational(4, 101);
        auto old_rank3 = response(3, 101);
        check(observational.accept(3, reinterpret_cast<const char*>(&old_rank3),
                                   sizeof(old_rank3)) ==
                  tide::StatsResponseDisposition::ACCEPTED,
              "observational stats response rejected");

        tide::StatsResponseCollector authoritative(4, 102);
        ann::ThreadSafeQueue<tide::ServerResponse> shared_queue;
        auto enqueue = [&](uint32_t rank, const ann::StatsResponseHeader& value) {
            tide::ServerResponse queued;
            queued.server_idx = rank;
            const auto* begin = reinterpret_cast<const char*>(&value);
            queued.response_data.assign(begin, begin + sizeof(value));
            shared_queue.push(std::move(queued));
        };
        enqueue(3, old_rank3);
        for (uint32_t rank : {2u, 0u, 3u, 1u}) enqueue(rank, response(rank, 102));
        bool saw_stale = false;
        tide::ServerResponse queued;
        while (shared_queue.try_pop(queued)) {
            const auto disposition = authoritative.accept(
                queued.server_idx, queued.response_data.data(),
                queued.response_data.size());
            if (disposition == tide::StatsResponseDisposition::STALE_REQUEST) {
                saw_stale = true;
            } else {
                check(disposition == tide::StatsResponseDisposition::ACCEPTED,
                      "out-of-order authoritative response rejected");
            }
        }
        check(saw_stale, "late observational response was not classified stale");
        check(shared_queue.size() == 0, "stats collector left a queued response");
        check(authoritative.complete() && authoritative.count() == 4,
              "authoritative collector did not produce exact four ranks");
        auto duplicate = response(1, 102);
        check(authoritative.accept(1, reinterpret_cast<const char*>(&duplicate),
                                   sizeof(duplicate)) ==
                  tide::StatsResponseDisposition::DUPLICATE,
              "duplicate stats response accepted");

        tide::StatsResponseCollector missing(4, 103);
        for (uint32_t rank : {0u, 1u, 2u}) {
            auto partial = response(rank, 103);
            check(missing.accept(rank, reinterpret_cast<const char*>(&partial),
                                 sizeof(partial)) ==
                      tide::StatsResponseDisposition::ACCEPTED,
                  "partial response rejected unexpectedly");
        }
        check(!missing.complete(), "missing rank produced complete snapshot");
        auto wrong_id = response(3, 104);
        check(missing.accept(3, reinterpret_cast<const char*>(&wrong_id),
                             sizeof(wrong_id)) ==
                  tide::StatsResponseDisposition::STALE_REQUEST,
              "wrong request id accepted");

        uint8_t control_ack =
            static_cast<uint8_t>(ann::MessageType::DRAIN_ACK);
        check(missing.accept(3,
                             reinterpret_cast<const char*>(&control_ack),
                             sizeof(control_ack)) ==
                  tide::StatsResponseDisposition::INVALID,
              "control response accepted as stats");
    } catch (const std::exception& exc) {
        std::cerr << "statistics test failed: " << exc.what() << '\n';
        return 1;
    }

    std::cout << "statistics tests passed\n";
    return 0;
}
