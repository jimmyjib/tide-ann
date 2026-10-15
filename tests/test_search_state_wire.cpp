// Search state wire format: byte layout, round trips, the P2P forward state and the ERROR reply.
#include "state_protocol.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ \
                  << ": " #expr "\n"; \
        return EXIT_FAILURE; \
    } \
} while (false)

namespace {

uint32_t u32_at(const std::vector<char>& wire, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, wire.data() + offset, sizeof(value));
    return value;
}

}  // namespace

int main() {
    // StateHeader is 52 bytes. BeamEntry.padding[0] carries the exact-distance flag.
    static_assert(sizeof(ann::StateHeader) == 52, "state header size");
    static_assert(offsetof(ann::StateHeader, flags) == 28, "flags offset");
    static_assert(offsetof(ann::StateHeader, hop_count) == 36, "hop_count offset");
    static_assert(offsetof(ann::StateHeader, cumulative_distance_computations) == 48,
                  "cumulative_distance_computations is the last header field");
    CHECK(ann::STATE_WIRE_VERSION == 6);
    const std::string format = ann::wire_format_string();
    CHECK(format.rfind("tide-state-wire|magic=1414087749|version=6|state_header=52|", 0) == 0);
    CHECK(format.find("|beam=12|route=20|") != std::string::npos);

    const float query[] = {1.0f, -2.5f, 3.25f};
    std::vector<ann::BeamEntry> beam = {{11, 0.5f, 1, {0, 0, 0}}, {42, 1.75f, 0, {0, 0, 0}}};
    ann::set_beam_entry_exact(beam[0], true);
    CHECK(ann::beam_entry_exact(beam[0]) && beam[0].padding[0] == ann::kBeamEntryExactDistance);
    const std::vector<uint32_t> seen = {11, 42, 7};
    const std::vector<ann::RouteHop> route = {{2, 100, 150}};
    const uint32_t flags = ann::STATE_FLAG_P2P | ann::STATE_FLAG_TRACE_ROUTE;

    const auto wire = ann::serialize_state(
        7, 50, 10, 8, /*current_partition*/ 2, query, 3, beam, flags,
        /*origin_server*/ 1, /*hop_count*/ 3, /*queue_us*/ 11, /*processing_us*/ 12,
        seen, route, /*cumulative_distance_computations*/ 345);

    // Exact layout: type byte, header, query, beam, seen count + ids, route count + hops.
    CHECK(wire.size() == 1 + 52 + 3 * 4 + 2 * 12 + 4 + 3 * 4 + 4 + 1 * 20);
    CHECK(static_cast<uint8_t>(wire[0]) == static_cast<uint8_t>(ann::MessageType::SEARCH_REQUEST));
    CHECK(u32_at(wire, 1 + 0) == 7);     // query_id
    CHECK(u32_at(wire, 1 + 4) == 50);    // L
    CHECK(u32_at(wire, 1 + 16) == 2);    // current_partition
    CHECK(u32_at(wire, 1 + 20) == 2);    // beam_size
    CHECK(u32_at(wire, 1 + 24) == 3);    // dim
    CHECK(u32_at(wire, 1 + 28) == flags);
    CHECK(u32_at(wire, 1 + 32) == 1);    // origin_server
    CHECK(u32_at(wire, 1 + 36) == 3);    // hop_count
    CHECK(u32_at(wire, 1 + 48) == 345);  // cumulative_distance_computations
    const size_t seen_at = 1 + 52 + 3 * 4 + 2 * 12;
    CHECK(u32_at(wire, seen_at) == 3);
    CHECK(u32_at(wire, seen_at + 4 + 3 * 4) == 1);

    ann::StateHeader header{};
    std::vector<float> decoded_query;
    std::vector<ann::BeamEntry> decoded_beam;
    std::vector<uint32_t> decoded_seen;
    std::vector<ann::RouteHop> decoded_route;
    CHECK(ann::deserialize_state(wire.data() + 1, wire.size() - 1, header,
                                 decoded_query, decoded_beam, decoded_seen, decoded_route));
    CHECK(header.query_id == 7 && header.L == 50 && header.K == 10 && header.W == 8);
    CHECK(header.current_partition == 2 && header.beam_size == 2 && header.dim == 3);
    CHECK(header.flags == flags && header.origin_server == 1 && header.hop_count == 3);
    CHECK(header.cumulative_queue_us == 11 && header.cumulative_processing_us == 12);
    CHECK(header.cumulative_distance_computations == 345);
    CHECK(decoded_query.size() == 3 && decoded_query[1] == -2.5f && decoded_query[2] == 3.25f);
    CHECK(decoded_beam.size() == 2 && decoded_beam[1].node_id == 42 &&
          decoded_beam[1].distance == 1.75f && decoded_beam[0].expanded == 1);
    // The exact-distance flag travels with the entry across a hop.
    CHECK(ann::beam_entry_exact(decoded_beam[0]) && !ann::beam_entry_exact(decoded_beam[1]));
    CHECK(decoded_seen == seen);
    CHECK(decoded_route.size() == 1 && decoded_route[0].partition == 2 &&
          decoded_route[0].arrive_us == 100 && decoded_route[0].depart_us == 150);

    // P2P hop: the server forwards with the incoming flags, hop_count + 1 and the
    // advanced cumulative counters (serialize_forward_state, called by the server).
    // The coordinator only sees the final response.
    std::vector<uint32_t> hop_seen = decoded_seen;
    hop_seen.push_back(99);
    std::vector<ann::RouteHop> hop_route = decoded_route;
    hop_route.push_back({1, 160, 170});
    std::vector<ann::BeamEntry> hop_beam = decoded_beam;
    hop_beam.push_back({77, 2.5f, 0, {0, 0, 0}});
    const auto forwarded = ann::serialize_forward_state(
        header, /*next_partition*/ 0, decoded_query.data(), hop_beam, /*hop_queue_us*/ 5,
        /*hop_processing_us*/ 6, /*hop_distance_computations*/ 55, hop_seen, hop_route);
    CHECK(static_cast<uint8_t>(forwarded[0]) == static_cast<uint8_t>(ann::MessageType::SEARCH_REQUEST));
    ann::StateHeader second{};
    CHECK(ann::deserialize_state(forwarded.data() + 1, forwarded.size() - 1, second,
                                 decoded_query, decoded_beam, decoded_seen, decoded_route));
    CHECK(second.query_id == 7 && second.L == 50 && second.K == 10 && second.W == 8 && second.dim == 3);
    CHECK(second.flags == flags);
    CHECK((second.flags & ann::STATE_FLAG_P2P) != 0 && (second.flags & ann::STATE_FLAG_TRACE_ROUTE) != 0);
    CHECK(second.origin_server == 1 && second.current_partition == 0 && second.hop_count == 4);
    CHECK(second.cumulative_queue_us == 16 && second.cumulative_processing_us == 18);
    CHECK(second.cumulative_distance_computations == 400);
    CHECK(decoded_query.size() == 3 && decoded_query[0] == 1.0f && decoded_query[2] == 3.25f);
    CHECK(second.beam_size == 3 && decoded_beam.size() == 3 && decoded_beam[2].node_id == 77);
    CHECK(decoded_seen.size() == 4 && decoded_seen.back() == 99);
    CHECK(decoded_route.size() == 2 && decoded_route[1].partition == 1);

    // A second hop from the forwarded state: the counters keep accumulating, and the
    // 32-bit wire fields wrap as the server's arithmetic does.
    ann::StateHeader near_wrap = second;
    near_wrap.cumulative_distance_computations = 0xFFFFFFF0u;
    const auto again = ann::serialize_forward_state(
        near_wrap, /*next_partition*/ 3, decoded_query.data(), decoded_beam, 1, 2, 0x20, decoded_seen,
        decoded_route);
    ann::StateHeader third{};
    CHECK(ann::deserialize_state(again.data() + 1, again.size() - 1, third,
                                 decoded_query, decoded_beam, decoded_seen, decoded_route));
    CHECK(third.hop_count == 5 && third.current_partition == 3 && third.flags == flags);
    CHECK(third.cumulative_queue_us == 17 && third.cumulative_processing_us == 20);
    CHECK(third.cumulative_distance_computations == 0x10u);

    // Coordinator-originated state without P2P or route trace: flags stay 0 and the
    // empty seen/route tails still round-trip.
    const auto plain = ann::serialize_state(9, 50, 10, 8, 3, query, 3, {});
    CHECK(plain.size() == 1 + 52 + 3 * 4 + 4 + 4);
    CHECK(ann::deserialize_state(plain.data() + 1, plain.size() - 1, second,
                                 decoded_query, decoded_beam, decoded_seen, decoded_route));
    CHECK(second.flags == 0 && second.hop_count == 0 && second.cumulative_distance_computations == 0);
    CHECK(decoded_beam.empty() && decoded_seen.empty() && decoded_route.empty());

    // Truncated buffers are rejected: a short header, or a body shorter than the
    // query and beam the header announces.
    CHECK(!ann::deserialize_state(wire.data() + 1, 51, header, decoded_query, decoded_beam,
                                  decoded_seen, decoded_route));
    CHECK(!ann::deserialize_state(wire.data() + 1, 52 + 3 * 4 + 12, header, decoded_query,
                                  decoded_beam, decoded_seen, decoded_route));

    // The reply to a rejected state (a query of another dimension than the
    // server's): a SEARCH_RESPONSE with status ERROR for that query id, no beam,
    // no result, which the coordinator turns into a command error.
    const auto refused = ann::serialize_search_error(/*query_id*/ 77, /*hop_count*/ 2);
    CHECK(static_cast<uint8_t>(refused[0]) == static_cast<uint8_t>(ann::MessageType::SEARCH_RESPONSE));
    ann::StateResponseHeader response{};
    std::vector<uint32_t> neighbors;
    std::vector<float> distances;
    CHECK(ann::deserialize_response(refused.data() + 1, refused.size() - 1, response, decoded_beam, neighbors,
                                    distances, decoded_seen, decoded_route));
    CHECK(response.query_id == 77 && response.status == ann::StateStatus::ERROR &&
          response.inter_partition_hops == 3 && response.beam_size == 0 && response.result_count == 0);
    CHECK(decoded_beam.empty() && neighbors.empty() && distances.empty() && decoded_seen.empty());
    return EXIT_SUCCESS;
}
