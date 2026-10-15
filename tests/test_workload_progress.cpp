// workload_progress_due.
#include <cstdio>
#include <cstdlib>

#include "workload_progress.h"

namespace {
using namespace tide;

void check(bool ok, const char* text) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", text); std::abort(); }
}

void workload_progress_cases() {
    check(!workload_progress_due(0, 0, 600000, 100, 180000) &&
          !workload_progress_due(123, 123, 600000, 100, 180000),
          "no report without new completions");
    check(!workload_progress_due(124, 123, 600000, 100, 1999) &&
          workload_progress_due(124, 123, 600000, 100, 2000) &&
          workload_progress_due(223, 123, 600000, 100, 0),
          "count or time interval reached");
    check(!workload_progress_due(222, 123, 600000, 100, 0), "99 completions below an interval of 100");
    check(workload_progress_due(600000, 599999, 600000, 100, 0) &&
          !workload_progress_due(600000, 600000, 600000, 100, 180000),
          "final count reported once");
    check(!workload_progress_due(4096, 8192, 600000, 4096, 180000) &&
          !workload_progress_due(600001, 600000, 600000, 100, 180000),
          "no report below the last report or past the total");
    check(workload_progress_due(150, 100, 1000, 100, 500, 500) &&
          !workload_progress_due(150, 100, 1000, 100, 499, 500),
          "time interval parameter");
}
}  // namespace

int main() {
    workload_progress_cases();
    std::printf("workload_progress_cases OK\n");
    return 0;
}
