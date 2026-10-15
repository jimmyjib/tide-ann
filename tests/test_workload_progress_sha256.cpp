// Header-only helpers that have no other test:
// - workload_progress_due (include/workload_progress.h), the progress-line throttle of the
//   pipeline (pipeline_core.cpp) and the workload generator;
// - sha256_hex (include/sha256.h), which authenticates snapshot frames
//   (window_snapshot_wire.h) and names the controller history context
//   (window_action_history.cpp): FIPS 180-4 known-answer vectors and the rules of the
//   optional progress callback.
// Progress cases plus the two-block and one-million-'a' SHA-256 vectors.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "sha256.h"
#include "workload_progress.h"

namespace {
using namespace tide;

void check(bool ok, const char* text) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", text); std::abort(); }
}

void workload_progress_cases() {
    check(!workload_progress_due(0, 0, 600000, 100, 180000) &&
          !workload_progress_due(123, 123, 600000, 100, 180000),
          "elapsed time never publishes unchanged completion progress");
    check(!workload_progress_due(124, 123, 600000, 100, 1999) &&
          workload_progress_due(124, 123, 600000, 100, 2000) &&
          workload_progress_due(223, 123, 600000, 100, 0),
          "actual advanced completions obey the count or elapsed-since-report throttle");
    check(!workload_progress_due(222, 123, 600000, 100, 0), "99 completions below a 100 interval wait");
    check(workload_progress_due(600000, 599999, 600000, 100, 0) &&
          !workload_progress_due(600000, 600000, 600000, 100, 180000) &&
          !workload_progress_due(4096, 8192, 600000, 4096, 180000) &&
          !workload_progress_due(600001, 600000, 600000, 100, 180000),
          "the final count logs once; stale concurrent tickets and counts past the total never log");
    check(workload_progress_due(150, 100, 1000, 100, 500, 500) &&
          !workload_progress_due(150, 100, 1000, 100, 499, 500),
          "the time interval is a parameter");
    std::printf("workload_progress_cases OK\n");
}

void sha256_known_answers() {
    check(sha256_hex(std::string()) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "FIPS 180-4: empty message");
    check(sha256_hex(std::string("abc")) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "FIPS 180-4: abc");
    check(sha256_hex(std::string("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "FIPS 180-4: 448-bit message (two padded blocks)");
    const std::string million(1000000, 'a');
    check(sha256_hex(million) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "FIPS 180-4: one million 'a'");
    check(sha256_hex(std::string("abd")) != sha256_hex(std::string("abc")), "one changed byte changes the digest");
    std::printf("sha256_known_answers OK\n");
}

void sha256_progress_rules() {
    struct Progress { std::vector<std::pair<size_t, size_t>> points; };
    for (size_t bytes : {size_t(0), size_t(55), size_t(56), size_t(64),
                        kSha256ProgressBytes, 2 * kSha256ProgressBytes + 17}) {
        const std::vector<uint8_t> input(bytes, 'a');
        Progress progress;
        const auto digest = sha256_hex(input.data(), input.size(),
            [](size_t completed, size_t total, void* context) {
                static_cast<Progress*>(context)->points.emplace_back(completed, total);
            }, &progress);
        check(digest == sha256_hex(input.data(), input.size()),
              "the progress callback cannot change input order, padding or digest");
        check(progress.points.size() == (bytes + kSha256ProgressBytes - 1) / kSha256ProgressBytes,
              "callback count is bounded by coarse byte intervals");
        size_t previous = 0;
        for (const auto& point : progress.points) {
            check(point.first > previous && point.first <= bytes && point.second == bytes,
                  "callbacks strictly advance real input bytes, never padding bytes");
            previous = point.first;
        }
        check(previous == bytes, "nonempty hashing reports the exact final byte count once");
    }
    std::printf("sha256_progress_rules OK\n");
}
}  // namespace

int main() {
    workload_progress_cases();
    sha256_known_answers();
    sha256_progress_rules();
    std::printf("workload progress and sha256 tests passed\n");
    return 0;
}
