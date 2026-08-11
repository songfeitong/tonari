#pragma once

#include "algorithm.h"
#include "pair_policy.h"

#include <cstdint>
#include <span>
#include <vector>


namespace neighbor_search {

struct PairBuffers {
    std::vector<int64_t> indices;
    std::vector<int32_t> shifts;
};


template <typename scalar_t>
PairBuffers neighbor_list_cpu(
    std::span<const scalar_t> positions,
    std::span<const int64_t> batch_ptr,
    std::span<const scalar_t> cells,
    std::span<const uint8_t> pbc,
    double cutoff,
    PairMode mode,
    Algorithm algorithm);


extern template PairBuffers neighbor_list_cpu<float>(
    std::span<const float>,
    std::span<const int64_t>,
    std::span<const float>,
    std::span<const uint8_t>,
    double,
    PairMode,
    Algorithm);

extern template PairBuffers neighbor_list_cpu<double>(
    std::span<const double>,
    std::span<const int64_t>,
    std::span<const double>,
    std::span<const uint8_t>,
    double,
    PairMode,
    Algorithm);

}  // namespace neighbor_search
