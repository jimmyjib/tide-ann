// Shutdown quiescence and admin ACK handling.
#include "protocol/message_types.h"
#include "thread_pool.h"

#include <cstdlib>
#include <atomic>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ \
                  << ": " #expr "\n"; \
        return EXIT_FAILURE; \
    } \
} while (false)

int main() {
    using ann::AdminAckDisposition;
    using ann::MessageType;

    CHECK(ann::admin_ack_type(MessageType::RESET_REQUEST) ==
          MessageType::RESET_ACK);
    CHECK(ann::admin_ack_type(MessageType::DRAIN_REQUEST) ==
          MessageType::DRAIN_ACK);
    CHECK(ann::admin_ack_type(MessageType::SHUTDOWN_REQUEST) ==
          MessageType::SHUTDOWN_ACK);
    CHECK(ann::admin_ack_type(MessageType::MIGRATION_PREPARE_INLINE_REQUEST) ==
          MessageType::MIGRATION_PREPARE_ACK);
    CHECK(ann::admin_ack_type(MessageType::MIGRATION_ACCEPT_REQUEST) ==
          MessageType::MIGRATION_ACCEPT_ACK);

    // The insert maintenance replies, which every non-owner exchange puts back
    // (controller-stop's STOP exchange included).
    CHECK(ann::is_maintenance_reply(MessageType::INSERT_RESPONSE));
    CHECK(ann::is_maintenance_reply(MessageType::BATCH_BACKLINK_ACK));
    CHECK(ann::is_maintenance_reply(MessageType::VECTOR_BROADCAST_ACK));
    CHECK(!ann::is_maintenance_reply(MessageType::STATS_WINDOW_STOP_ACK));
    CHECK(!ann::is_maintenance_reply(MessageType::SEARCH_RESPONSE));
    CHECK(!ann::is_maintenance_reply(MessageType::GET_STATS_RESPONSE));
    // Admin exchanges put back foreign replies, except RESET, which drops them.
    CHECK(!ann::admin_command_requeues_foreign_replies(MessageType::RESET_REQUEST));
    CHECK(ann::admin_command_requeues_foreign_replies(MessageType::DRAIN_REQUEST));
    CHECK(ann::admin_command_requeues_foreign_replies(MessageType::SHUTDOWN_REQUEST));

    CHECK(ann::classify_admin_ack(
              MessageType::SHUTDOWN_ACK, MessageType::SHUTDOWN_ACK,
              2, 4, false) == AdminAckDisposition::ACCEPT);
    CHECK(ann::classify_admin_ack(
              MessageType::SHUTDOWN_ACK, MessageType::SHUTDOWN_ACK,
              2, 4, true) == AdminAckDisposition::DUPLICATE);
    CHECK(ann::classify_admin_ack(
              MessageType::SHUTDOWN_ACK, MessageType::DRAIN_ACK,
              2, 4, false) == AdminAckDisposition::UNMATCHED);
    CHECK(ann::classify_admin_ack(
              MessageType::SHUTDOWN_ACK, MessageType::SHUTDOWN_ACK,
              4, 4, false) == AdminAckDisposition::INVALID_SERVER);

    CHECK(ann::shutdown_state_is_quiescent(0, 0, 0, 0, 0, 0, 0));
    CHECK(!ann::shutdown_state_is_quiescent(1, 0, 0, 0, 0, 0, 0));
    CHECK(!ann::shutdown_state_is_quiescent(0, 1, 0, 0, 0, 0, 0));
    CHECK(!ann::shutdown_state_is_quiescent(0, 0, 1, 0, 0, 0, 0));
    CHECK(!ann::shutdown_state_is_quiescent(0, 0, 0, 1, 0, 0, 0));
    CHECK(!ann::shutdown_state_is_quiescent(0, 0, 0, 0, 1, 0, 0));
    CHECK(!ann::shutdown_state_is_quiescent(0, 0, 0, 0, 0, 1, 0));
    CHECK(!ann::shutdown_state_is_quiescent(0, 0, 0, 0, 0, 0, 1));

    // The dequeue-to-handler-lock window, deterministically:
    // ownership is published while holding the shared dequeue lock and stays
    // visible after that lock is released, so shutdown cannot report idle.
    std::shared_mutex admin_mutex;
    std::atomic<uint32_t> search_in_flight{0};
    std::atomic<bool> dequeued{false};
    std::atomic<bool> release_search{false};
    std::thread worker([&]() {
        ann::SearchInFlightOwnership ownership(search_in_flight);
        {
            std::shared_lock<std::shared_mutex> lock(admin_mutex);
            ownership.acquire_if_search(MessageType::SEARCH_REQUEST);
        }
        dequeued.store(true, std::memory_order_release);
        while (!release_search.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ownership.release();
    });
    while (!dequeued.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    {
        std::unique_lock<std::shared_mutex> lock(admin_mutex);
        CHECK(!ann::shutdown_state_is_quiescent(
            0, 0, 0, 0, 0, 0,
            search_in_flight.load(std::memory_order_acquire)));
    }
    release_search.store(true, std::memory_order_release);
    worker.join();
    CHECK(search_in_flight.load(std::memory_order_acquire) == 0);

    // Idle waiting is notification-driven but does not dequeue. Production
    // can therefore wait outside g_admin_mutex, then return to the existing
    // admin-locked try_pop/in-flight publication sequence.
    ann::ThreadSafeQueue<ann::WorkItem> idle_queue;
    std::atomic<bool> wait_started{false};
    std::atomic<bool> queue_became_ready{false};
    std::thread idle_waiter([&]() {
        wait_started.store(true, std::memory_order_release);
        queue_became_ready.store(
            idle_queue.wait_until_nonempty(500),
            std::memory_order_release);
    });
    while (!wait_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    ann::WorkItem queued_work;
    queued_work.msg_type = MessageType::SEARCH_REQUEST;
    idle_queue.push(std::move(queued_work));
    idle_waiter.join();
    CHECK(queue_became_ready.load(std::memory_order_acquire));
    CHECK(idle_queue.size() == 1);
    ann::WorkItem popped_work;
    CHECK(idle_queue.try_pop(popped_work));
    CHECK(idle_queue.size() == 0);

    // The same ownership rule covers all three maintenance queues. The
    // dequeue becomes invisible only while shared admin ownership and an
    // in-flight count are both published.
    for (MessageType maintenance_kind : {
             MessageType::INSERT_REQUEST,
             MessageType::BATCH_BACKLINK_REQUEST,
             MessageType::VECTOR_BROADCAST}) {
        (void)maintenance_kind;
        std::atomic<uint32_t> maintenance_in_flight{0};
        std::atomic<bool> dequeued_maintenance{false};
        std::atomic<bool> release_maintenance{false};
        std::thread maintenance_worker([&]() {
            std::shared_lock<std::shared_mutex> lock(admin_mutex);
            ann::InFlightOwnership ownership(maintenance_in_flight);
            ownership.acquire();
            dequeued_maintenance.store(true, std::memory_order_release);
            lock.unlock();
            while (!release_maintenance.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            ownership.release();
        });
        while (!dequeued_maintenance.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::shared_mutex> lock(admin_mutex);
            CHECK(!ann::shutdown_state_is_quiescent(
                0, 0, 0, 0, 0,
                maintenance_in_flight.load(std::memory_order_acquire), 0));
        }
        release_maintenance.store(true, std::memory_order_release);
        maintenance_worker.join();
        CHECK(maintenance_in_flight.load(std::memory_order_acquire) == 0);
    }

    CHECK(static_cast<uint8_t>(MessageType::DRAIN_REQUEST) == 18);
    CHECK(static_cast<uint8_t>(MessageType::DRAIN_ACK) == 19);
    CHECK(static_cast<uint8_t>(MessageType::SHUTDOWN_REQUEST) == 20);
    CHECK(static_cast<uint8_t>(MessageType::SHUTDOWN_ACK) == 21);

    std::cout << "shutdown protocol tests passed\n";
    return EXIT_SUCCESS;
}
