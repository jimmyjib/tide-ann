cmake_minimum_required(VERSION 3.20)
project(tide_ann CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

enable_testing()

# ZeroMQ: libzmq and the cppzmq header zmq.hpp
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ QUIET libzmq)
endif()
if(NOT ZMQ_FOUND)
    find_library(ZMQ_LIBRARY NAMES zmq)
    if(NOT ZMQ_LIBRARY)
        message(FATAL_ERROR "libzmq not found (Debian/Ubuntu: libzmq3-dev, macOS: brew install zeromq), "
                            "or set ZMQ_LIBRARY")
    endif()
    set(ZMQ_LIBRARIES ${ZMQ_LIBRARY})
    set(ZMQ_INCLUDE_DIRS "")
    set(ZMQ_LIBRARY_DIRS "")
endif()
find_path(CPPZMQ_INCLUDE_DIR zmq.hpp HINTS ${ZMQ_INCLUDE_DIRS} ${ZMQ_INCLUDEDIR})
if(NOT CPPZMQ_INCLUDE_DIR)
    message(FATAL_ERROR "zmq.hpp not found (install cppzmq or copy zmq.hpp into an include directory), "
                        "or set CPPZMQ_INCLUDE_DIR")
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

# Common compile options
set(COMMON_FLAGS -Wall -Wextra ${SIMD_FLAGS})

# Common include directory
set(COMMON_INCLUDE src/common/include)

# Binaries

set(SERVER_SOURCES
    src/tide/server/graph_ops.cpp
    src/tide/server/handlers.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp
    src/tide/server/stats_window.cpp
)
set(COORDINATOR_SOURCES
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

# Index server: one partition of the graph, P2P state passing, inserts, migration.
add_executable(tide-server src/tide/tide_server.cpp ${SERVER_SOURCES})
target_include_directories(tide-server PRIVATE ${COMMON_INCLUDE} src/tide/include ${ZMQ_INCLUDE_DIRS})
target_link_directories(tide-server PRIVATE ${ZMQ_LIBRARY_DIRS})
target_link_libraries(tide-server PRIVATE pthread ${ZMQ_LIBRARIES})
target_compile_options(tide-server PRIVATE ${COMMON_FLAGS})

# Coordinator: query dispatch, insert pipeline, admin commands, controller.
add_executable(tide-coordinator src/tide/tide_coordinator.cpp ${COORDINATOR_SOURCES})
target_include_directories(tide-coordinator PRIVATE ${COMMON_INCLUDE} src/tide/include ${ZMQ_INCLUDE_DIRS})
target_link_directories(tide-coordinator PRIVATE ${ZMQ_LIBRARY_DIRS})
target_link_libraries(tide-coordinator PRIVATE ${ZMQ_LIBRARIES})
target_compile_options(tide-coordinator PRIVATE ${COMMON_FLAGS})

# PQ training and encoding (OpenMP when available)
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

# Tests

set(TIDE_TEST_DIR tests)

# tide_add_test(<name> <source>... [ZMQ] [ARGS <arg>...])
# -UNDEBUG keeps assert() active in Release builds.
function(tide_add_test name)
    cmake_parse_arguments(ARG "ZMQ" "" "ARGS" ${ARGN})
    set(target test_${name})
    add_executable(${target} ${ARG_UNPARSED_ARGUMENTS})
    target_include_directories(${target} PRIVATE ${COMMON_INCLUDE} src/tide/include)
    target_link_libraries(${target} PRIVATE pthread)
    target_compile_options(${target} PRIVATE ${COMMON_FLAGS} -UNDEBUG)
    if(ARG_ZMQ)
        target_include_directories(${target} PRIVATE ${ZMQ_INCLUDE_DIRS})
        target_link_directories(${target} PRIVATE ${ZMQ_LIBRARY_DIRS})
        target_link_libraries(${target} PRIVATE ${ZMQ_LIBRARIES})
    endif()
    add_test(NAME ${name} COMMAND ${target} ${ARG_ARGS})
    set_tests_properties(${name} PROPERTIES TIMEOUT 600)
endfunction()

tide_add_test(chunked_array ${TIDE_TEST_DIR}/test_chunked_array.cpp)
tide_add_test(statistics ${TIDE_TEST_DIR}/test_statistics.cpp)
tide_add_test(shutdown_protocol ${TIDE_TEST_DIR}/test_shutdown_protocol.cpp)
tide_add_test(search_state_wire ${TIDE_TEST_DIR}/test_search_state_wire.cpp)
tide_add_test(serving ${TIDE_TEST_DIR}/test_serving.cpp)
tide_add_test(workload_progress ${TIDE_TEST_DIR}/test_workload_progress.cpp)
tide_add_test(sha256 ${TIDE_TEST_DIR}/test_sha256.cpp)
tide_add_test(daemon_socket_ownership
    ${TIDE_TEST_DIR}/test_daemon_socket_ownership.cpp
    src/tide/coordinator/daemon_state.cpp
    ZMQ)

# tide-server argument checks
tide_add_test(server_args ${TIDE_TEST_DIR}/test_server_args.cpp ARGS $<TARGET_FILE:tide-server>)
add_dependencies(test_server_args tide-server)

# Product quantization (pq.h) and the PQ-mode storage helpers (server_types.h)
tide_add_test(pq ${TIDE_TEST_DIR}/test_pq.cpp)
tide_add_test(own_only_store
    ${TIDE_TEST_DIR}/test_own_only_store.cpp
    ${SERVER_SOURCES}
    ZMQ)

# Server search path (links the real handlers)
tide_add_test(search_source_accounting
    ${TIDE_TEST_DIR}/test_search_source_accounting.cpp
    ${SERVER_SOURCES}
    ZMQ)

# Concurrent searches, inserts and window controls (useful under -fsanitize=thread)
tide_add_test(server_concurrency
    ${TIDE_TEST_DIR}/test_server_concurrency.cpp
    ${SERVER_SOURCES}
    ZMQ)

# Membership migration and window statistics
tide_add_test(membership_migration
    ${TIDE_TEST_DIR}/test_membership_migration.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp)
tide_add_test(stats_window
    ${TIDE_TEST_DIR}/test_stats_window.cpp
    src/tide/server/stats_window.cpp)

# Controller
tide_add_test(window_controller
    ${TIDE_TEST_DIR}/test_window_controller.cpp
    ${TIDE_TEST_DIR}/test_window_action_history.cpp
    src/tide/coordinator/window_controller.cpp
    src/tide/coordinator/window_action_history.cpp
    src/tide/coordinator/window_controller_runtime.cpp
    src/tide/coordinator/window_snapshot_transport.cpp
    src/tide/coordinator/window_native_control.cpp
    src/tide/coordinator/window_repair_planner.cpp
    src/tide/server/membership_migration.cpp
    src/tide/server/membership_migration_storage.cpp)
# Controller tick with stubbed coordinator calls
tide_add_test(window_controller_lifecycle
    ${TIDE_TEST_DIR}/test_window_controller_lifecycle.cpp
    ${TIDE_TEST_DIR}/test_window_snapshot_transport.cpp
    ${TIDE_TEST_DIR}/test_window_native_control.cpp
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
tide_add_test(window_repair_planner
    ${TIDE_TEST_DIR}/test_window_repair_planner.cpp
    ${TIDE_TEST_DIR}/test_window_repair_reference_exact.cpp
    src/tide/coordinator/window_repair_planner.cpp)
# The reference planner is compiled without FP contraction (see test_window_repair_reference.h).
set_source_files_properties(${TIDE_TEST_DIR}/test_window_repair_reference_exact.cpp
    PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
# Regenerate with: test_window_repair_planner_golden --write <path>
tide_add_test(window_repair_planner_golden
    ${TIDE_TEST_DIR}/test_window_repair_planner_golden.cpp
    src/tide/coordinator/window_repair_planner.cpp
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/${TIDE_TEST_DIR}/testdata/repair_golden_k4.json)
# Coordinator against in-process fake servers
tide_add_test(coordinator_admin
    ${TIDE_TEST_DIR}/test_coordinator_admin.cpp
    ${COORDINATOR_SOURCES}
    ZMQ)

# Install

install(TARGETS
    tide-server
    tide-coordinator
    tide-build-pq
    RUNTIME DESTINATION bin
)
