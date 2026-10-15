#pragma once
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace tide {
// One worker thread per controller. Only the owner submits and polls. Jobs own their inputs. The
// queue holds at most `capacity` jobs (window_control_queue_capacity).
class WindowControlWorker {
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::function<void()>> queue_;
    size_t capacity_ = 8;
    bool stopping_ = false;
    std::thread thread_;
public:
    WindowControlWorker() : thread_([this] {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) return;
                job = std::move(queue_.front()); queue_.pop_front();
            }
            job();
        }
    }) {}
    WindowControlWorker(const WindowControlWorker&) = delete;
    void set_capacity(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = capacity;
    }
    WindowControlWorker& operator=(const WindowControlWorker&) = delete;
    ~WindowControlWorker() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; queue_.clear(); }
        wake_.notify_one();
        if (thread_.joinable()) thread_.join();
    }
    template<class F> auto submit(F work) -> std::future<decltype(work())> {
        using Result = decltype(work());
        auto task = std::make_shared<std::packaged_task<Result()>>(std::move(work));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || queue_.size() >= capacity_) return {};
            queue_.emplace_back([task] { (*task)(); });
        }
        wake_.notify_one(); return future;
    }
};
template<class T> bool window_control_ready(std::future<T>& job) {
    return job.valid() && job.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}
} // namespace tide
