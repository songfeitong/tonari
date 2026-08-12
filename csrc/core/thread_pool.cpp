#include "thread_pool.h"

#include "errors.h"

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


namespace {

class ParallelJob {
public:
    ParallelJob(
        int64_t task_count,
        int64_t participant_count,
        const std::function<void(int64_t)>& function)
        : task_count_(task_count),
          remaining_participants_(participant_count),
          function_(function) {}

    void run() noexcept {
        while (!cancelled_.load(std::memory_order_relaxed)) {
            const int64_t task = next_task_.fetch_add(1, std::memory_order_relaxed);
            if (task >= task_count_) {
                break;
            }
            try {
                function_(task);
            } catch (...) {
                {
                    std::lock_guard lock(completion_mutex_);
                    if (exception_ == nullptr) {
                        exception_ = std::current_exception();
                    }
                }
                cancelled_.store(true, std::memory_order_relaxed);
            }
        }
        if (remaining_participants_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            {
                std::lock_guard lock(completion_mutex_);
                complete_ = true;
            }
            completion_condition_.notify_one();
        }
    }

    void wait() {
        std::unique_lock lock(completion_mutex_);
        completion_condition_.wait(lock, [this] { return complete_; });
        if (exception_ != nullptr) {
            std::rethrow_exception(exception_);
        }
    }

private:
    int64_t task_count_;
    std::atomic<int64_t> next_task_ = 0;
    std::atomic<int64_t> remaining_participants_;
    std::atomic<bool> cancelled_ = false;
    const std::function<void(int64_t)>& function_;
    std::mutex completion_mutex_;
    std::condition_variable completion_condition_;
    bool complete_ = false;
    std::exception_ptr exception_;
};


class ThreadPool {
public:
    void ensure_workers(int64_t worker_count) {
        std::lock_guard lock(worker_mutex_);
        while (static_cast<int64_t>(workers_.size()) < worker_count) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void submit(const std::shared_ptr<ParallelJob>& job, int64_t worker_count) {
        {
            std::lock_guard lock(queue_mutex_);
            for (int64_t worker = 0; worker < worker_count; ++worker) {
                queue_.push_back(job);
            }
        }
        queue_condition_.notify_all();
    }

private:
    void worker_loop() {
        while (true) {
            std::shared_ptr<ParallelJob> job;
            {
                std::unique_lock lock(queue_mutex_);
                queue_condition_.wait(lock, [this] { return !queue_.empty(); });
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            job->run();
        }
    }

    std::mutex worker_mutex_;
    std::vector<std::jthread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::deque<std::shared_ptr<ParallelJob>> queue_;
};


std::mutex global_pool_mutex;
ThreadPool* global_pool = nullptr;
bool atfork_registered = false;


void lock_pool_before_fork() {
    global_pool_mutex.lock();
}


void unlock_pool_after_fork() {
    global_pool_mutex.unlock();
}


void reset_pool_after_fork() {
    // Worker threads do not survive fork. The inherited object is deliberately
    // left untouched and unreachable because joining its stale thread handles
    // is not safe in the child process.
    global_pool = nullptr;
    global_pool_mutex.unlock();
}


ThreadPool& process_thread_pool() {
    std::lock_guard lock(global_pool_mutex);
    if (!atfork_registered) {
        neighbor_search::require_search(
            pthread_atfork(
                lock_pool_before_fork,
                unlock_pool_after_fork,
                reset_pool_after_fork) == 0,
            "failed to register CPU worker-pool fork handlers");
        atfork_registered = true;
    }
    if (global_pool == nullptr) {
        // This process-lifetime allocation also prevents extension shutdown
        // from racing workers that are waiting in native code.
        global_pool = new ThreadPool();
    }
    return *global_pool;
}

}  // namespace


void neighbor_search::parallel_for(
    int64_t task_count,
    int64_t num_threads,
    const std::function<void(int64_t)>& function) {
    require_input(num_threads > 0, "num_threads must be a positive integer");
    if (task_count <= 0) {
        return;
    }
    const int64_t participant_count = std::min(task_count, num_threads);
    if (participant_count == 1) {
        for (int64_t task = 0; task < task_count; ++task) {
            function(task);
        }
        return;
    }

    ThreadPool& pool = process_thread_pool();
    const int64_t worker_count = participant_count - 1;
    pool.ensure_workers(worker_count);
    auto job = std::make_shared<ParallelJob>(
        task_count, participant_count, function);
    pool.submit(job, worker_count);
    job->run();
    job->wait();
}
