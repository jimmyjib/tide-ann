cmake_minimum_required(VERSION 3.20)
project(tide_ann CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release is the default build type (single-configuration generators).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

enable_testing()

# ZeroMQ: the C library libzmq plus the C++ header zmq.hpp (cppzmq, header-only).
# libzmq comes from pkg-config when it is available, else from find_library. pkg-config leaves
# ZMQ_INCLUDE_DIRS empty when the headers sit in a default search path such as /usr/include, so
# zmq.hpp is always located on its own (the libzmq development package does not always ship it).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ QUIET libzmq)
endif()
if(NOT ZMQ_FOUND)
    find_library(ZMQ_LIBRARY NAMES zmq)
    if(NOT ZMQ_LIBRARY)
        message(FATAL_ERROR "libzmq not found. Install the ZeroMQ development package "
                            "(Debian/Ubuntu: libzmq3-dev; macOS: brew install zeromq) or set ZMQ_LIBRARY.")
    endif()
    set(ZMQ_LIBRARIES ${ZMQ_LIBRARY})
    set(ZMQ_INCLUDE_DIRS "")
    set(ZMQ_LIBRARY_DIRS "")
endif()
find_path(CPPZMQ_INCLUDE_DIR zmq.hpp HINTS ${ZMQ_INCLUDE_DIRS} ${ZMQ_INCLUDEDIR})
if(NOT CPPZMQ_INCLUDE_DIR)
    message(FATAL_ERROR "The ZeroMQ C++ header zmq.hpp (cppzmq) was not found. Install cppzmq "
                        "(Debian/Ubuntu: cppzmq-dev; macOS: brew install cppzmq), copy zmq.hpp from "
                        "https://github.com/zeromq/cppzmq into an include directory, or set CPPZMQ_INCLUDE_DIR.")
endif()
list(APPEND ZMQ_INCLUDE_DIRS ${CPPZMQ_INCLUDE_DIR})
list(REMOVE_DUPLICATES ZMQ_INCLUDE_DIRS)
message(STATUS "ZeroMQ: libraries ${ZMQ_LIBRARIES}, include ${ZMQ_INCLUDE_DIRS}")

# SIMD flags based on architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(SIMD_FLAGS -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(SIMD_FLAGS "")
endif()

# Distance metric: squared L2 only. Cosine data is normalised to unit length offline
# (tools/prepare/normalize_vectors.py); there is no cosine build.

# Common compile options
set(COMMON_FLAGS -Wall -Wextra ${SIMD_FLAGS})

# Common include directory
set(COMMON_INCLUDE src/common/include)

#=============================================================================
# Binaries (state-passing search between partitions)
#=============================================================================

# Index server: one partition of the graph, P2P state passing, inserts, migration.
add_executable(tide-server
    src/tide/tide_server.cpp
    src/tide/server/graph_ops.cpp
    src/tide/server/handlers.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    src/tide/server/stats_window.cpp
)
target_include_directories(tide-server PRIVATE ${COMMON_INCLUDE} src/tide/include ${ZMQ_INCLUDE_DIRS})
target_link_directories(tide-server PRIVATE ${ZMQ_LIBRARY_DIRS})
target_link_libraries(tide-server PRIVATE pthread ${ZMQ_LIBRARIES})
target_compile_options(tide-server PRIVATE ${COMMON_FLAGS})

# Coordinator: query dispatch, insert pipeline, admin commands, repair controller.
add_executable(tide-coordinator
    src/tide/tide_coordinator.cpp
    src/tide/coordinator/daemon_state.cpp
    src/tide/coordinator/metrics.cpp
    src/tide/coordinator/workload_runners.cpp
    src/tide/coordinator/admin_commands.cpp
    src/tide/coordinator/pipeline/debug_utils.cpp
    src/tide/coordinator/pipeline/workload_generator.cpp
    src/tide/coordinator/pipeline/pipeline_core.cpp
    src/tide/coordinator/pipeline/metric_thread.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    src/tide/coordinator/window_controller.cpp
    src/tide/coordinator/window_action_history.cpp
    src/tide/coordinator/window_controller_runtime.cpp
    src/tide/coordinator/window_snapshot_transport.cpp
    src/tide/coordinator/window_native_control.cpp
    src/tide/coordinator/window_controller_tick.cpp
    src/tide/coordinator/window_controller_apply_proposal.cpp
    src/tide/coordinator/window_repair_planner.cpp
)
target_include_directories(tide-coordinator PRIVATE ${COMMON_INCLUDE} src/tide/include ${ZMQ_INCLUDE_DIRS})
target_link_directories(tide-coordinator PRIVATE ${ZMQ_LIBRARY_DIRS})
target_link_libraries(tide-coordinator PRIVATE ${ZMQ_LIBRARIES})
target_compile_options(tide-coordinator PRIVATE ${COMMON_FLAGS})

# tide-build-pq: trains the PQ pivots of a base .npy and (--codes) encodes every row.
# OpenMP, when the compiler provides it, parallelises training and encoding over
# chunks / rows; the result does not depend on it (per-chunk seeded k-means).
add_executable(tide-build-pq src/tide/build_pq.cpp)
target_include_directories(tide-build-pq PRIVATE ${COMMON_INCLUDE} src/tide/include)
target_compile_options(tide-build-pq PRIVATE ${COMMON_FLAGS})
target_link_libraries(tide-build-pq PRIVATE pthread)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tide-build-pq PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "tide-build-pq: OpenMP ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "tide-build-pq: no OpenMP (single-threaded training)")
endif()

#=============================================================================
# Tests
#=============================================================================

# Directory of the test sources and their testdata/.
set(ANN_TEST_DIR tests)

# ann_add_test(<name> <source>... [ZMQ] [ARGS <arg>...])
# Builds test_<name> from the sources and registers it as ctest <name>.
# Every test compiles with -UNDEBUG: CMAKE_CXX_FLAGS_RELEASE carries -DNDEBUG,
# and the later -UNDEBUG keeps assert() checks active in Release builds.
# ZMQ adds the libzmq include directory and library.
function(ann_add_test name)
    cmake_parse_arguments(ARG "ZMQ" "" "ARGS" ${ARGN})
    set(target test_${name})
    add_executable(${target} ${ARG_UNPARSED_ARGUMENTS})
    target_include_directories(${target} PRIVATE ${COMMON_INCLUDE} src/tide/include src/tide)
    target_link_libraries(${target} PRIVATE pthread)
    target_compile_options(${target} PRIVATE ${COMMON_FLAGS} -UNDEBUG)
    if(ARG_ZMQ)
        target_include_directories(${target} PRIVATE ${ZMQ_INCLUDE_DIRS})
        target_link_directories(${target} PRIVATE ${ZMQ_LIBRARY_DIRS})
        target_link_libraries(${target} PRIVATE ${ZMQ_LIBRARIES})
    endif()
    add_test(NAME ${name} COMMAND ${target} ${ARG_ARGS})
    # A test that hangs fails after 10 minutes instead of blocking ctest (every test takes seconds).
    set_tests_properties(${name} PROPERTIES TIMEOUT 600)
endfunction()

ann_add_test(chunked_array ${ANN_TEST_DIR}/test_chunked_array.cpp)
ann_add_test(statistics ${ANN_TEST_DIR}/test_statistics.cpp)
ann_add_test(shutdown_protocol ${ANN_TEST_DIR}/test_shutdown_protocol.cpp)
ann_add_test(query_execution_plan ${ANN_TEST_DIR}/test_query_execution_plan.cpp)
ann_add_test(search_state_wire ${ANN_TEST_DIR}/test_search_state_wire.cpp)
ann_add_test(serving_accounting ${ANN_TEST_DIR}/test_serving_accounting.cpp)
# Header-only helpers: the progress throttle and SHA-256 (known answers, progress rules).
ann_add_test(workload_progress_sha256 ${ANN_TEST_DIR}/test_workload_progress_sha256.cpp)
ann_add_test(daemon_socket_ownership
    ${ANN_TEST_DIR}/test_daemon_socket_ownership.cpp
    src/tide/coordinator/daemon_state.cpp
    ZMQ)

# The server binary's start-up argument checks (--num-servers, --rank, --peers).
ann_add_test(server_args ${ANN_TEST_DIR}/test_server_args.cpp ARGS $<TARGET_FILE:tide-server>)
add_dependencies(test_server_args tide-server)

# Product quantization (pq.h) and the own-only storage helpers (server_types.h)
ann_add_test(pq ${ANN_TEST_DIR}/test_pq.cpp)
ann_add_test(own_only_store
    ${ANN_TEST_DIR}/test_own_only_store.cpp
    src/tide/server/graph_ops.cpp
    src/tide/server/handlers.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    src/tide/server/stats_window.cpp
    ZMQ)

# Server search path (links the real handlers)
ann_add_test(search_source_accounting
    ${ANN_TEST_DIR}/test_search_source_accounting.cpp
    src/tide/server/graph_ops.cpp
    src/tide/server/handlers.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    src/tide/server/stats_window.cpp
    ZMQ)

# The worker paths concurrently (searches while insert, backlink and broadcast
# batches apply; window controls while searches count), both storage modes.
# Built with -fsanitize=thread it is the race test of these paths.
ann_add_test(server_concurrency
    ${ANN_TEST_DIR}/test_server_concurrency.cpp
    src/tide/server/graph_ops.cpp
    src/tide/server/handlers.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    src/tide/server/stats_window.cpp
    ZMQ)

# Membership migration and window statistics
ann_add_test(membership_migration
    ${ANN_TEST_DIR}/test_membership_migration.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp)
ann_add_test(stats_window
    ${ANN_TEST_DIR}/test_stats_window.cpp
    src/tide/server/stats_window.cpp)

# Repair controller
# Pure pieces and the action history (no tick: nothing here needs the daemon).
ann_add_test(window_controller
    ${ANN_TEST_DIR}/test_window_controller.cpp
    ${ANN_TEST_DIR}/test_window_action_history.cpp
    src/tide/coordinator/window_controller.cpp
    src/tide/coordinator/window_action_history.cpp
    src/tide/coordinator/window_controller_runtime.cpp
    src/tide/coordinator/window_snapshot_transport.cpp
    src/tide/coordinator/window_native_control.cpp
    src/tide/coordinator/window_repair_planner.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp)
# The production tick with stubbed coordinator exchanges.
ann_add_test(window_controller_lifecycle
    ${ANN_TEST_DIR}/test_window_controller_lifecycle.cpp
    ${ANN_TEST_DIR}/test_window_snapshot_transport.cpp
    ${ANN_TEST_DIR}/test_window_native_control.cpp
    src/tide/coordinator/window_controller.cpp
    src/tide/coordinator/window_action_history.cpp
    src/tide/coordinator/window_controller_runtime.cpp
    src/tide/coordinator/window_snapshot_transport.cpp
    src/tide/coordinator/window_native_control.cpp
    src/tide/coordinator/window_controller_tick.cpp
    src/tide/coordinator/window_controller_apply_proposal.cpp
    src/tide/coordinator/window_repair_planner.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    ZMQ)
ann_add_test(window_repair_planner
    ${ANN_TEST_DIR}/test_window_repair_planner.cpp
    ${ANN_TEST_DIR}/test_window_repair_reference_exact.cpp
    src/tide/coordinator/window_repair_planner.cpp)
# The randomized tests compare the planner, compiled with the production flags
# (GCC contracts a multiply that feeds an add into one fused multiply-add within
# a basic block, as in tide-coordinator), with a separately written reference
# compiled WITHOUT contraction: only the reference's own source file carries
# -ffp-contract=off. Decisions must agree exactly and loads within rounding; a
# decision that differs only at a rounding boundary is shown and bounded
# (test_window_repair_reference.h).
set_source_files_properties(${ANN_TEST_DIR}/test_window_repair_reference_exact.cpp
    PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
# Golden output of the repair planner (K = 4, W^nb). ctest compares against the
# checked-in file; `test_window_repair_planner_golden --write <path>` regenerates it.
ann_add_test(window_repair_planner_golden
    ${ANN_TEST_DIR}/test_window_repair_planner_golden.cpp
    src/tide/coordinator/window_repair_planner.cpp
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/${ANN_TEST_DIR}/testdata/repair_golden_k4.json)
# Coordinator admin commands, the pipeline loop and the search runner against
# in-process fake servers (every coordinator source except main()).
ann_add_test(coordinator_admin
    ${ANN_TEST_DIR}/test_coordinator_admin.cpp
    src/tide/coordinator/daemon_state.cpp
    src/tide/coordinator/metrics.cpp
    src/tide/coordinator/workload_runners.cpp
    src/tide/coordinator/admin_commands.cpp
    src/tide/coordinator/pipeline/debug_utils.cpp
    src/tide/coordinator/pipeline/workload_generator.cpp
    src/tide/coordinator/pipeline/pipeline_core.cpp
    src/tide/coordinator/pipeline/metric_thread.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    src/tide/coordinator/window_controller.cpp
    src/tide/coordinator/window_action_history.cpp
    src/tide/coordinator/window_controller_runtime.cpp
    src/tide/coordinator/window_snapshot_transport.cpp
    src/tide/coordinator/window_native_control.cpp
    src/tide/coordinator/window_controller_tick.cpp
    src/tide/coordinator/window_controller_apply_proposal.cpp
    src/tide/coordinator/window_repair_planner.cpp
    ZMQ)

#=============================================================================
# Install
#=============================================================================

install(TARGETS
    tide-server
    tide-coordinator
    tide-build-pq
    RUNTIME DESTINATION bin
)
