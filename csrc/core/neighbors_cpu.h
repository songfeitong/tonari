#pragma once

#include "pair_policy.h"

#include <cstdint>
#include <span>
#include <vector>


namespace neighbor_search {

struct PairBuffers {
    std::vector<int64_t> sources;
    std::vector<int64_t> targets;
    std::vector<int32_t> shifts;
};


template <typename scalar_t>
PairBuffers find_neighbors_cpu(
    std::span<const scalar_t> positions,
    std::span<const int64_t> batch_ptr,
    std::span<const scalar_t> cells,
    std::span<const uint8_t> pbc,
    double cutoff,
    PairMode mode);


extern template PairBuffers find_neighbors_cpu<float>(
    std::span<const float>,
    std::span<const int64_t>,
    std::span<const float>,
    std::span<const uint8_t>,
    double,
    PairMode);

extern template PairBuffers find_neighbors_cpu<double>(
    std::span<const double>,
    std::span<const int64_t>,
    std::span<const double>,
    std::span<const uint8_t>,
    double,
    PairMode);

}  // namespace neighbor_search
