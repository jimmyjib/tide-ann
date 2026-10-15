#include "query_execution_plan.h"

#include <cassert>
#include <cstdint>

int main() {
    using tide::make_alternate_query_execution_plan;

    auto short_run = make_alternate_query_execution_plan(256, 4096);
    assert(short_run.loaded_rows == 256);
    assert(short_run.execution_count == 256);

    auto exact_run = make_alternate_query_execution_plan(4096, 4096);
    assert(exact_run.loaded_rows == 4096);
    assert(exact_run.execution_count == 4096);

    auto cycling_run = make_alternate_query_execution_plan(10000, 4096);
    assert(cycling_run.loaded_rows == 4096);
    assert(cycling_run.execution_count == 10000);

    auto empty_source = make_alternate_query_execution_plan(10000, 0);
    assert(empty_source.loaded_rows == 0);
    assert(empty_source.execution_count == 10000);
    return 0;
}
