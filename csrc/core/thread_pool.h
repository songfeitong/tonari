#pragma once

#include <cstdint>
#include <functional>


namespace neighbor_search {

void parallel_for(
    int64_t task_count,
    int64_t num_threads,
    const std::function<void(int64_t)>& function);

}  // namespace neighbor_search
