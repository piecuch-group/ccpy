#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <limits>
#include <thread>
#include <utility>
#include <vector>
#include <charconv>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#define SERIAL_THRESHOLD 3

namespace ccpy {

namespace detail {

constexpr std::size_t max_threads = std::numeric_limits<std::size_t>::max();

constexpr bool is_space(const char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

/// @brief Parse a positive thread count from an environment variable.
/// @param value Environment variable value, or nullptr when it is unset.
/// @param allow_list Whether to allow commas. Only the first value is used if true.
/// @return The parsed limit, or max_threads when the value is unset or invalid.
inline std::size_t parse_thread_limit(const char* value, const bool allow_list = false) noexcept {
    if (value == nullptr)
        return max_threads;

    while (is_space(*value))
        ++value;

    const char* end = value;
    while (*end != '\0')
        ++end;

    std::size_t limit = 0;
    auto [ptr, ec] = std::from_chars(value, end, limit);
    if (ec != std::errc{}) // bail on any error
        return max_threads;

    while (is_space(*ptr))
        ++ptr;
    if (*ptr != '\0' && !(allow_list && *ptr == ','))
        return max_threads;
    if (limit == 0)
        return max_threads;
    return limit;
}

/// @brief Return the number of CPUs in the process affinity mask when available.
inline std::size_t get_affinity_thread_limit() noexcept {
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        std::size_t count = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &affinity))
                ++count;
        }
        if (count > 0)
            return count;
    }
#endif
    return max_threads;
}

/// @brief Join every joinable thread in a worker container when leaving scope.
/// Allows exceptions to propagate if a thread fails.
/// This guard must be declared immediately after the worker vector. If creating a later worker
/// throws, the guard joins the workers that were already created before their destructors run.
class ThreadJoiner {
  public:
    explicit ThreadJoiner(std::vector<std::thread>& threads) noexcept : threads_(threads) {}

    ThreadJoiner(const ThreadJoiner&) = delete;
    ThreadJoiner& operator=(const ThreadJoiner&) = delete;

    ~ThreadJoiner() {
        for (auto& thread : threads_) {
            if (thread.joinable())
                thread.join();
        }
    }

  private:
    std::vector<std::thread>& threads_;
};

} // namespace detail

/// @brief Get the number of threads to use for parallel execution
/// CCPY_NUM_THREADS_OVERRIDE is used if set/valid.
/// Otherwise, the smallest set value among
/// {   std::thread::hardware_concurrency() (logical CPUs, including SMT/hyperthreads),
///     # of CPUs in affinity mask,
///     OMP_NUM_THREADS,
///     OMP_THREAD_LIMIT,
///     SLURM_CPUS_PER_TASK,
/// } is used.
/// It always returns at least one thread.
/// @return The number of threads to use for parallel execution.
inline std::size_t get_num_threads() {
    auto num_threads = detail::parse_thread_limit(std::getenv("CCPY_NUM_THREADS_OVERRIDE"));

    if (num_threads != detail::max_threads) {
        return std::max<std::size_t>(1, num_threads);
    }

    num_threads = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    num_threads = std::min(num_threads, detail::get_affinity_thread_limit());
    num_threads =
        std::min(num_threads, detail::parse_thread_limit(std::getenv("OMP_NUM_THREADS"), true));
    num_threads =
        std::min(num_threads, detail::parse_thread_limit(std::getenv("OMP_THREAD_LIMIT")));
    num_threads =
        std::min(num_threads, detail::parse_thread_limit(std::getenv("SLURM_CPUS_PER_TASK")));
    return std::max<std::size_t>(1, num_threads);
}

template <typename F> void serial_for(const std::size_t begin, const std::size_t end, F&& func) {
    for (std::size_t i{begin}; i < end; ++i)
        func(i);
}

template <typename F>
void parallel_for_chunked_thread(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        func(begin, end);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    detail::ThreadJoiner thread_joiner(threads);
    const std::size_t block_size = (count + num_threads - 1) / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t block_begin = begin + t * block_size;
        std::size_t block_end = std::min(block_begin + block_size, end);
        if (block_begin < block_end) {
            threads.emplace_back(
                [block_begin, block_end, &func]() { func(block_begin, block_end); });
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename F>
void parallel_for_chunked_async(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        func(begin, end);
        return;
    }
    std::vector<std::future<void>> tasks;
    const std::size_t block_size = (count + num_threads - 1) / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t block_begin = begin + t * block_size;
        std::size_t block_end = std::min(block_begin + block_size, end);
        if (block_begin < block_end) {
            tasks.emplace_back(std::async(std::launch::async, [block_begin, block_end, &func]() {
                func(block_begin, block_end);
            }));
        }
    }

    for (auto& task : tasks) {
        task.get();
    }
}

template <typename F>
void parallel_for_interleaved_async(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();

    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }

    std::vector<std::future<void>> workers;
    workers.reserve(num_threads);
    for (std::size_t t{0}; t < num_threads; ++t) {
        workers.push_back(std::async(std::launch::async, [count, num_threads, t, begin, &func]() {
            for (std::size_t i{t}; i < count; i += num_threads)
                func(i + begin);
        }));
    }

    for (auto& w : workers)
        w.get();
}

template <typename F>
void parallel_for_interleaved_thread(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    detail::ThreadJoiner thread_joiner(workers);
    for (std::size_t t{0}; t < num_threads; ++t) {
        workers.emplace_back([count, num_threads, t, begin, &func]() {
            for (std::size_t i{t}; i < count; i += num_threads)
                func(i + begin);
        });
    }

    for (auto& w : workers)
        w.join();
}

template <typename F>
void parallel_for_dynamic_async(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || end - begin < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }
    std::atomic<std::size_t> next_index(begin);
    std::vector<std::future<void>> tasks;
    for (std::size_t t = 0; t < num_threads; ++t) {
        tasks.emplace_back(std::async(std::launch::async, [begin, end, &func, &next_index]() {
            while (true) {
                std::size_t index = next_index.fetch_add(1);
                if (index >= end)
                    break;
                func(index);
            }
        }));
    }
    for (auto& task : tasks) {
        task.get();
    }
}

template <typename F>
void parallel_for_dynamic_thread(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || end - begin < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }
    std::atomic<std::size_t> next_index(begin);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    detail::ThreadJoiner thread_joiner(threads);
    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([begin, end, &func, &next_index]() {
            while (true) {
                std::size_t index = next_index.fetch_add(1);
                if (index >= end)
                    break;
                func(index);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename F>
void parallel_for_chunked(const std::size_t begin, const std::size_t end, F&& func) {
    parallel_for_chunked_thread(begin, end, std::forward<F>(func));
}

template <typename F> void parallel_for_chunked(const std::size_t count, F&& func) {
    return parallel_for_chunked(0, count, std::forward<F>(func));
}

// if Apple, then use dispatch_apply for parallelism
// finally, fall back to std::thread
#if defined(__APPLE__)
template <typename F> void parallel_for(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;

    dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(std::size_t i) {
      func(begin + i);
    });
}

template <typename F> void parallel_for(const std::size_t count, F&& func) {
    return parallel_for(0, count, std::forward<F>(func));
}
#else
template <typename F> void parallel_for(const std::size_t begin, const std::size_t end, F&& func) {
    parallel_for_interleaved_thread(begin, end, std::forward<F>(func));
}

template <typename F> void parallel_for(const std::size_t count, F&& func) {
    return parallel_for(0, count, std::forward<F>(func));
}
#endif

} // namespace ccpy
