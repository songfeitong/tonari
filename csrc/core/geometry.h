#pragma once

#include <cstdint>
#include <span>
#include <vector>


namespace neighbor_search {

struct PeriodicMetadata {
    std::vector<double> duals;
    std::vector<int32_t> image_shifts;
    std::vector<int64_t> image_offsets;
};


PeriodicMetadata build_periodic_metadata(
    std::span<const double> cells,
    std::span<const uint8_t> pbc,
    std::span<const int64_t> atom_counts,
    double cutoff);

}  // namespace neighbor_search
