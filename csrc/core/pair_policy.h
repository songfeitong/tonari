#pragma once

#include <cstdint>


#if defined(__CUDACC__)
#define NEIGHBOR_SEARCH_HOST_DEVICE __host__ __device__
#define NEIGHBOR_SEARCH_FORCE_INLINE __forceinline__
#else
#define NEIGHBOR_SEARCH_HOST_DEVICE
#define NEIGHBOR_SEARCH_FORCE_INLINE inline
#endif


namespace neighbor_search {

enum class PairMode : uint8_t {
    Full = 0,
    FullWithSelf = 1,
    Half = 2,
    HalfWithSelf = 3,
};


inline PairMode pair_mode(bool half_list, bool include_self) {
    return static_cast<PairMode>(
        (half_list ? static_cast<uint8_t>(PairMode::Half) : 0) |
        (include_self ? static_cast<uint8_t>(PairMode::FullWithSelf) : 0));
}


template <PairMode Mode>
inline constexpr bool kHalfList =
    Mode == PairMode::Half || Mode == PairMode::HalfWithSelf;


template <PairMode Mode>
inline constexpr bool kIncludeSelf =
    Mode == PairMode::FullWithSelf || Mode == PairMode::HalfWithSelf;


NEIGHBOR_SEARCH_HOST_DEVICE NEIGHBOR_SEARCH_FORCE_INLINE bool is_zero_shift_self_pair(
    int64_t source,
    int64_t target,
    const int64_t (&cell_shift)[3]) {
    return source == target && cell_shift[0] == 0 && cell_shift[1] == 0 &&
        cell_shift[2] == 0;
}


NEIGHBOR_SEARCH_HOST_DEVICE NEIGHBOR_SEARCH_FORCE_INLINE bool is_canonical_half_pair(
    int64_t source,
    int64_t target,
    const int64_t (&cell_shift)[3]) {
    if (source != target) {
        return source < target;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (cell_shift[axis] != 0) {
            return cell_shift[axis] < 0;
        }
    }
    return true;
}


template <PairMode Mode>
NEIGHBOR_SEARCH_HOST_DEVICE NEIGHBOR_SEARCH_FORCE_INLINE bool keep_pair_identity(
    int64_t source,
    int64_t target,
    const int64_t (&cell_shift)[3]) {
    if (is_zero_shift_self_pair(source, target, cell_shift)) {
        return kIncludeSelf<Mode>;
    }
    if constexpr (kHalfList<Mode>) {
        return is_canonical_half_pair(source, target, cell_shift);
    }
    return true;
}

}  // namespace neighbor_search


#undef NEIGHBOR_SEARCH_FORCE_INLINE
#undef NEIGHBOR_SEARCH_HOST_DEVICE
