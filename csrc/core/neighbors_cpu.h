#pragma once

#include "algorithm.h"
#include "pair_policy.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>


namespace neighbor_search {

struct PairBuffer {
    std::vector<int64_t> indices;
    std::vector<int32_t> shifts;
};


struct PairBuffers {
    std::vector<PairBuffer> chunks;
    size_t pair_count = 0;
};


void copy_pair_buffers(
    const PairBuffers& pairs,
    std::span<int64_t> indices,
    std::span<int32_t> shifts,
    int64_t num_threads);


template <typename scalar_t>
PairBuffers neighbor_list_cpu(
    std::span<const scalar_t> positions,
    std::span<const int64_t> batch_ptr,
    std::span<const scalar_t> cells,
    std::span<const uint8_t> pbc,
    double cutoff,
    PairMode mode,
    Algorithm algorithm,
    int64_t num_threads);


extern template PairBuffers neighbor_list_cpu<float>(
    std::span<const float>,
    std::span<const int64_t>,
    std::span<const float>,
    std::span<const uint8_t>,
    double,
    PairMode,
    Algorithm,
    int64_t);

extern template PairBuffers neighbor_list_cpu<double>(
    std::span<const double>,
    std::span<const int64_t>,
    std::span<const double>,
    std::span<const uint8_t>,
    double,
    PairMode,
    Algorithm,
    int64_t);

}  // namespace neighbor_search
