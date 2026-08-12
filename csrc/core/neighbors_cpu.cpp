#include "neighbors_cpu.h"
#include "errors.h"
#include "geometry.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>


namespace {

using neighbor_search::PairBuffer;
using neighbor_search::PairBuffers;

// The crossover was selected from a real-structure threshold sweep.
constexpr int64_t kBruteForceCandidateLimit = 16384;
// Avoid pathological dense allocation for sparse finite coordinates.
constexpr int64_t kMaximumDenseBins = int64_t{1} << 26;
constexpr int64_t kMaximumBinsPerImage = 64;
constexpr int kCellListRoundoffFactor = 64;
constexpr int64_t kBruteForceCandidatesPerTask = int64_t{1} << 18;
constexpr int64_t kCellListSourcesPerTask = 128;


struct CellNode {
    int32_t target;
    int32_t shift;
    int32_t next;
};


template <typename scalar_t>
struct BinLayout {
    scalar_t origins[3];
    scalar_t size;
    int64_t dimensions[3];
    int64_t count;
};


template <typename scalar_t>
struct CellListData {
    BinLayout<scalar_t> layout;
    std::vector<int32_t> bin_heads;
    std::vector<CellNode> nodes;
    scalar_t search_cutoff_squared;
    scalar_t inner_cutoff_squared;
};


template <typename scalar_t>
struct PreparedStructure {
    int64_t atom_offset = 0;
    int64_t n_atoms = 0;
    int64_t shift_offset = 0;
    int64_t n_shifts = 0;
    neighbor_search::Algorithm algorithm = neighbor_search::Algorithm::Auto;
    const scalar_t* positions = nullptr;
    const scalar_t* cell = nullptr;
    const scalar_t* dual = nullptr;
    const int32_t* image_shifts = nullptr;
    std::vector<scalar_t> wrapped_positions;
    std::vector<int32_t> atom_wraps;
    std::vector<scalar_t> translations;
    CellListData<scalar_t> cell_list;
};


struct QueryTask {
    int64_t structure;
    int64_t source_begin;
    int64_t source_end;
    bool prepare_structure;
};


bool brute_force_candidate_count_at_most(
    int64_t n_atoms, int64_t n_shifts, int64_t limit) {
    if (n_atoms == 0 || n_shifts == 0) {
        return true;
    }
    return n_atoms <= limit / n_atoms &&
        n_shifts <= limit / (n_atoms * n_atoms);
}


template <typename scalar_t>
void prepare_positions(
    const scalar_t* positions,
    const scalar_t* cell,
    const scalar_t* dual,
    int64_t n_atoms,
    std::vector<scalar_t>& wrapped_positions,
    std::vector<int32_t>& atom_wraps) {
    const scalar_t wrap_upper_bound = static_cast<scalar_t>(2147483648.0);
    wrapped_positions.resize(3 * n_atoms);
    atom_wraps.resize(3 * n_atoms);
    for (int64_t atom = 0; atom < n_atoms; ++atom) {
        const scalar_t* position = positions + 3 * atom;
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            neighbor_search::require_search(
                std::isfinite(position[cartesian]),
                "positions must contain only finite values");
        }
        for (int axis = 0; axis < 3; ++axis) {
            long double fractional = 0.0L;
            for (int cartesian = 0; cartesian < 3; ++cartesian) {
                fractional += static_cast<long double>(position[cartesian]) *
                    static_cast<long double>(dual[3 * cartesian + axis]);
            }
            neighbor_search::require_search(
                std::isfinite(fractional) &&
                    fractional >= -static_cast<long double>(wrap_upper_bound) &&
                    fractional < static_cast<long double>(wrap_upper_bound),
                "atom representatives require periodic wraps outside the int32 range");
            atom_wraps[3 * atom + axis] =
                static_cast<int32_t>(std::floor(fractional));
        }
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            scalar_t wrapped = position[cartesian];
            for (int axis = 0; axis < 3; ++axis) {
                wrapped -= static_cast<scalar_t>(atom_wraps[3 * atom + axis]) *
                    cell[3 * axis + cartesian];
            }
            wrapped_positions[3 * atom + cartesian] = wrapped;
        }
    }
}


template <typename scalar_t>
std::vector<scalar_t> image_translations(
    const int32_t* image_shifts,
    const scalar_t* cell,
    int64_t n_shifts) {
    std::vector<scalar_t> translations(3 * n_shifts);
    for (int64_t shift = 0; shift < n_shifts; ++shift) {
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            scalar_t translation = scalar_t(0);
            for (int axis = 0; axis < 3; ++axis) {
                translation += static_cast<scalar_t>(image_shifts[3 * shift + axis]) *
                    cell[3 * axis + cartesian];
            }
            translations[3 * shift + cartesian] = translation;
        }
    }
    return translations;
}


template <typename scalar_t>
bool conservative_cell_list_cutoff(
    const scalar_t* positions,
    const scalar_t* cell,
    int64_t n_atoms,
    const std::vector<scalar_t>& wrapped_positions,
    const std::vector<int32_t>& atom_wraps,
    const int32_t* image_shifts,
    int64_t n_shifts,
    scalar_t cutoff,
    scalar_t& search_cutoff) {
    long double maximum_position[3] = {};
    int64_t maximum_wrap[3] = {};
    int64_t maximum_image_shift[3] = {};
    for (int64_t atom = 0; atom < n_atoms; ++atom) {
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            if (!std::isfinite(wrapped_positions[3 * atom + cartesian])) {
                return false;
            }
            maximum_position[cartesian] = std::max(
                maximum_position[cartesian],
                std::abs(static_cast<long double>(
                    positions[3 * atom + cartesian])));
        }
        for (int axis = 0; axis < 3; ++axis) {
            maximum_wrap[axis] = std::max(
                maximum_wrap[axis],
                std::abs(static_cast<int64_t>(atom_wraps[3 * atom + axis])));
        }
    }
    for (int64_t shift = 0; shift < n_shifts; ++shift) {
        for (int axis = 0; axis < 3; ++axis) {
            maximum_image_shift[axis] = std::max(
                maximum_image_shift[axis],
                std::abs(static_cast<int64_t>(image_shifts[3 * shift + axis])));
        }
    }
    long double operation_scale = 0.0L;
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        long double component_scale = 2 * maximum_position[cartesian];
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t maximum_output_shift =
                maximum_image_shift[axis] + 2 * maximum_wrap[axis];
            component_scale += static_cast<long double>(maximum_output_shift) *
                std::abs(static_cast<long double>(cell[3 * axis + cartesian]));
        }
        operation_scale = std::max(operation_scale, component_scale);
    }
    // Wrapped and direct displacement formulas are algebraically identical,
    // but their floating-point cancellation differs for remote representatives.
    const long double padding = kCellListRoundoffFactor *
        std::numeric_limits<scalar_t>::epsilon() * operation_scale;
    const long double padded_cutoff =
        static_cast<long double>(cutoff) + padding;
    if (!std::isfinite(padded_cutoff) ||
        padded_cutoff > 2 * static_cast<long double>(cutoff) ||
        padded_cutoff > std::numeric_limits<scalar_t>::max()) {
        return false;
    }
    search_cutoff = std::nextafter(
        static_cast<scalar_t>(padded_cutoff),
        std::numeric_limits<scalar_t>::infinity());
    return std::isfinite(search_cutoff);
}


template <typename scalar_t, neighbor_search::PairMode Mode>
inline void append_candidate(
    int64_t source,
    int64_t target,
    int64_t atom_offset,
    int64_t shift,
    const scalar_t* positions,
    const scalar_t* cell,
    const std::vector<int32_t>& atom_wraps,
    const int32_t* image_shifts,
    bool distance_known_inside,
    scalar_t cutoff_squared,
    PairBuffer& pairs) {
    const int32_t* search_shift = image_shifts + 3 * shift;
    int64_t output_shift[3];
    for (int axis = 0; axis < 3; ++axis) {
        output_shift[axis] = static_cast<int64_t>(search_shift[axis]) -
            static_cast<int64_t>(atom_wraps[3 * target + axis]) +
            static_cast<int64_t>(atom_wraps[3 * source + axis]);
    }
    const bool zero_shift_self =
        neighbor_search::is_zero_shift_self_pair(source, target, output_shift);
    if (!neighbor_search::keep_pair_identity<Mode>(source, target, output_shift)) {
        return;
    }
    if (!zero_shift_self && !distance_known_inside) {
        scalar_t distance_squared = scalar_t(0);
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            scalar_t component = positions[3 * target + cartesian] -
                positions[3 * source + cartesian];
            for (int axis = 0; axis < 3; ++axis) {
                component += static_cast<scalar_t>(output_shift[axis]) *
                    cell[3 * axis + cartesian];
            }
            distance_squared += component * component;
        }
        if (distance_squared >= cutoff_squared) {
            return;
        }
    }
    for (const int64_t value : output_shift) {
        neighbor_search::require_search(
            value >= std::numeric_limits<int32_t>::min() &&
                value <= std::numeric_limits<int32_t>::max(),
            "a cell shift required by the cutoff pairs exceeds the int32 output range");
    }
    pairs.indices.push_back(atom_offset + source);
    pairs.indices.push_back(atom_offset + target);
    for (const int64_t value : output_shift) {
        pairs.shifts.push_back(static_cast<int32_t>(value));
    }
}


template <typename scalar_t, neighbor_search::PairMode Mode>
inline void append_cell_candidate(
    int64_t source,
    int64_t target,
    int64_t atom_offset,
    int64_t shift,
    const scalar_t* positions,
    const scalar_t* cell,
    const std::vector<scalar_t>& wrapped_positions,
    const std::vector<int32_t>& atom_wraps,
    const int32_t* image_shifts,
    const std::vector<scalar_t>& translations,
    scalar_t search_cutoff_squared,
    scalar_t inner_cutoff_squared,
    scalar_t cutoff_squared,
    PairBuffer& pairs) {
    if constexpr (neighbor_search::kHalfList<Mode>) {
        const int32_t* search_shift = image_shifts + 3 * shift;
        int64_t output_shift[3];
        for (int axis = 0; axis < 3; ++axis) {
            output_shift[axis] = static_cast<int64_t>(search_shift[axis]) -
                static_cast<int64_t>(atom_wraps[3 * target + axis]) +
                static_cast<int64_t>(atom_wraps[3 * source + axis]);
        }
        if (!neighbor_search::keep_pair_identity<Mode>(
                source, target, output_shift)) {
            return;
        }
    }
    scalar_t search_distance_squared = scalar_t(0);
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        const scalar_t component = wrapped_positions[3 * target + cartesian] -
            wrapped_positions[3 * source + cartesian] +
            translations[3 * shift + cartesian];
        search_distance_squared += component * component;
    }
    if (search_distance_squared >= search_cutoff_squared) {
        return;
    }
    append_candidate<scalar_t, Mode>(
        source,
        target,
        atom_offset,
        shift,
        positions,
        cell,
        atom_wraps,
        image_shifts,
        search_distance_squared < inner_cutoff_squared,
        cutoff_squared,
        pairs);
}


template <typename scalar_t, neighbor_search::PairMode Mode>
void search_brute_force(
    int64_t source_begin,
    int64_t source_end,
    int64_t atom_offset,
    int64_t n_atoms,
    int64_t n_shifts,
    const scalar_t* positions,
    const scalar_t* cell,
    const std::vector<int32_t>& atom_wraps,
    const int32_t* image_shifts,
    scalar_t cutoff_squared,
    PairBuffer& pairs) {
    for (int64_t source = source_begin; source < source_end; ++source) {
        for (int64_t target = 0; target < n_atoms; ++target) {
            for (int64_t shift = 0; shift < n_shifts; ++shift) {
                append_candidate<scalar_t, Mode>(
                    source,
                    target,
                    atom_offset,
                    shift,
                    positions,
                    cell,
                    atom_wraps,
                    image_shifts,
                    false,
                    cutoff_squared,
                    pairs);
            }
        }
    }
}


neighbor_search::Algorithm select_cpu_algorithm(
    neighbor_search::Algorithm requested,
    int64_t n_atoms,
    int64_t n_shifts) {
    if (requested != neighbor_search::Algorithm::Auto) {
        return requested;
    }
    if (brute_force_candidate_count_at_most(
            n_atoms, n_shifts, kBruteForceCandidateLimit)) {
        return neighbor_search::Algorithm::BruteForce;
    }
    return neighbor_search::Algorithm::CellList;
}


template <typename scalar_t>
bool build_bin_layout(
    const std::vector<scalar_t>& wrapped_positions,
    int64_t n_atoms,
    int64_t n_shifts,
    scalar_t cutoff,
    scalar_t (&bounds_minimum)[3],
    scalar_t (&bounds_maximum)[3],
    BinLayout<scalar_t>& layout) {
    const scalar_t bin_size = cutoff;
    if (bin_size == scalar_t(0)) {
        return false;
    }
    layout.size = bin_size;
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        bounds_minimum[cartesian] = std::numeric_limits<scalar_t>::infinity();
        bounds_maximum[cartesian] = -std::numeric_limits<scalar_t>::infinity();
    }
    for (int64_t atom = 0; atom < n_atoms; ++atom) {
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            bounds_minimum[cartesian] = std::min(
                bounds_minimum[cartesian],
                wrapped_positions[3 * atom + cartesian]);
            bounds_maximum[cartesian] = std::max(
                bounds_maximum[cartesian],
                wrapped_positions[3 * atom + cartesian]);
        }
    }
    layout.count = 1;
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        const scalar_t span =
            bounds_maximum[cartesian] - bounds_minimum[cartesian] + 2 * cutoff;
        const scalar_t dimension = span / bin_size;
        if (!std::isfinite(dimension) || dimension > kMaximumDenseBins) {
            return false;
        }
        layout.origins[cartesian] = bounds_minimum[cartesian] - cutoff;
        layout.dimensions[cartesian] = std::max(
            int64_t{1}, static_cast<int64_t>(std::ceil(dimension)));
        if (layout.count > kMaximumDenseBins / layout.dimensions[cartesian]) {
            return false;
        }
        layout.count *= layout.dimensions[cartesian];
    }
    if (n_atoms > std::numeric_limits<int64_t>::max() / n_shifts) {
        return false;
    }
    const int64_t possible_images = n_atoms * n_shifts;
    return layout.count <= kMaximumDenseBins &&
        (layout.count + kMaximumBinsPerImage - 1) / kMaximumBinsPerImage <=
        possible_images;
}


template <typename scalar_t>
int64_t bin_index(
    const scalar_t (&position)[3],
    const BinLayout<scalar_t>& layout) {
    int64_t coordinates[3];
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        coordinates[cartesian] = std::clamp(
            static_cast<int64_t>(std::floor(
                (position[cartesian] - layout.origins[cartesian]) / layout.size)),
            int64_t{0},
            layout.dimensions[cartesian] - 1);
    }
    return (coordinates[0] * layout.dimensions[1] + coordinates[1]) *
        layout.dimensions[2] + coordinates[2];
}


template <typename scalar_t>
bool prepare_cell_list(
    int64_t n_atoms,
    int64_t n_shifts,
    const std::vector<scalar_t>& wrapped_positions,
    const std::vector<scalar_t>& translations,
    scalar_t search_cutoff,
    scalar_t inner_cutoff_squared,
    CellListData<scalar_t>& data) {
    neighbor_search::require_search(
        n_atoms < std::numeric_limits<int32_t>::max() &&
            n_shifts < std::numeric_limits<int32_t>::max(),
        "cell-list atoms and image shifts must fit int32 indexing");
    scalar_t bounds_minimum[3];
    scalar_t bounds_maximum[3];
    if (!build_bin_layout(
            wrapped_positions,
            n_atoms,
            n_shifts,
            search_cutoff,
            bounds_minimum,
            bounds_maximum,
            data.layout)) {
        return false;
    }
    data.bin_heads.assign(data.layout.count, -1);
    const int64_t possible_images = n_atoms * n_shifts;
    data.nodes.reserve(
        static_cast<size_t>(std::min(possible_images, 4 * n_atoms)));
    for (int64_t target = 0; target < n_atoms; ++target) {
        for (int64_t shift = 0; shift < n_shifts; ++shift) {
            scalar_t image_position[3];
            bool inside_bounds = true;
            for (int cartesian = 0; cartesian < 3; ++cartesian) {
                image_position[cartesian] =
                    wrapped_positions[3 * target + cartesian] +
                    translations[3 * shift + cartesian];
                inside_bounds &= std::isfinite(image_position[cartesian]) &&
                    image_position[cartesian] >=
                        bounds_minimum[cartesian] - search_cutoff &&
                    image_position[cartesian] <=
                        bounds_maximum[cartesian] + search_cutoff;
            }
            if (!inside_bounds) {
                continue;
            }
            const int64_t bin = bin_index(image_position, data.layout);
            neighbor_search::require_search(
                data.nodes.size() <
                    static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "cell-list node count exceeds int32 indexing");
            const int32_t node = static_cast<int32_t>(data.nodes.size());
            data.nodes.push_back(
                {static_cast<int32_t>(target),
                 static_cast<int32_t>(shift),
                 data.bin_heads[bin]});
            data.bin_heads[bin] = node;
        }
    }
    data.search_cutoff_squared = search_cutoff * search_cutoff;
    data.inner_cutoff_squared = inner_cutoff_squared;
    return true;
}


template <typename scalar_t, neighbor_search::PairMode Mode>
void search_cell_list(
    int64_t source_begin,
    int64_t source_end,
    int64_t atom_offset,
    const scalar_t* positions,
    const scalar_t* cell,
    const std::vector<scalar_t>& wrapped_positions,
    const std::vector<int32_t>& atom_wraps,
    const int32_t* image_shifts,
    const std::vector<scalar_t>& translations,
    const CellListData<scalar_t>& data,
    scalar_t cutoff_squared,
    PairBuffer& pairs) {
    for (int64_t source = source_begin; source < source_end; ++source) {
        scalar_t source_position[3];
        int64_t source_coordinates[3];
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            source_position[cartesian] =
                wrapped_positions[3 * source + cartesian];
            source_coordinates[cartesian] = std::clamp(
                static_cast<int64_t>(std::floor(
                    (source_position[cartesian] -
                     data.layout.origins[cartesian]) /
                    data.layout.size)),
                int64_t{0},
                data.layout.dimensions[cartesian] - 1);
        }
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            const int64_t x = source_coordinates[0] + offset_x;
            if (x < 0 || x >= data.layout.dimensions[0]) {
                continue;
            }
            for (int offset_y = -1; offset_y <= 1; ++offset_y) {
                const int64_t y = source_coordinates[1] + offset_y;
                if (y < 0 || y >= data.layout.dimensions[1]) {
                    continue;
                }
                for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                    const int64_t z = source_coordinates[2] + offset_z;
                    if (z < 0 || z >= data.layout.dimensions[2]) {
                        continue;
                    }
                    const int64_t bin =
                        (x * data.layout.dimensions[1] + y) *
                            data.layout.dimensions[2] +
                        z;
                    for (int32_t node = data.bin_heads[bin]; node >= 0;
                         node = data.nodes[node].next) {
                        append_cell_candidate<scalar_t, Mode>(
                            source,
                            data.nodes[node].target,
                            atom_offset,
                            data.nodes[node].shift,
                            positions,
                            cell,
                            wrapped_positions,
                            atom_wraps,
                            image_shifts,
                            translations,
                            data.search_cutoff_squared,
                            data.inner_cutoff_squared,
                            cutoff_squared,
                            pairs);
                    }
                }
            }
        }
    }
}


template <typename scalar_t, neighbor_search::PairMode Mode>
void search_prepared_structure(
    const PreparedStructure<scalar_t>& structure,
    int64_t source_begin,
    int64_t source_end,
    scalar_t cutoff_squared,
    PairBuffer& pairs) {
    if (structure.algorithm == neighbor_search::Algorithm::BruteForce) {
        search_brute_force<scalar_t, Mode>(
            source_begin,
            source_end,
            structure.atom_offset,
            structure.n_atoms,
            structure.n_shifts,
            structure.positions,
            structure.cell,
            structure.atom_wraps,
            structure.image_shifts,
            cutoff_squared,
            pairs);
        return;
    }
    search_cell_list<scalar_t, Mode>(
        source_begin,
        source_end,
        structure.atom_offset,
        structure.positions,
        structure.cell,
        structure.wrapped_positions,
        structure.atom_wraps,
        structure.image_shifts,
        structure.translations,
        structure.cell_list,
        cutoff_squared,
        pairs);
}


template <typename scalar_t>
void prepare_structure(
    PreparedStructure<scalar_t>& structure,
    scalar_t cutoff,
    neighbor_search::Algorithm requested_algorithm) {
    if (structure.n_atoms == 0) {
        return;
    }
    prepare_positions(
        structure.positions,
        structure.cell,
        structure.dual,
        structure.n_atoms,
        structure.wrapped_positions,
        structure.atom_wraps);
    structure.algorithm = select_cpu_algorithm(
        requested_algorithm, structure.n_atoms, structure.n_shifts);
    if (structure.algorithm == neighbor_search::Algorithm::BruteForce) {
        return;
    }

    structure.translations = image_translations(
        structure.image_shifts, structure.cell, structure.n_shifts);
    scalar_t search_cutoff = scalar_t(0);
    const bool cell_list_cutoff_is_safe = conservative_cell_list_cutoff(
        structure.positions,
        structure.cell,
        structure.n_atoms,
        structure.wrapped_positions,
        structure.atom_wraps,
        structure.image_shifts,
        structure.n_shifts,
        cutoff,
        search_cutoff);
    // Candidates safely inside the error band keep the fast wrapped
    // distance; only the boundary shell needs the direct public formula.
    const scalar_t inner_cutoff = std::max(
        scalar_t(0), 2 * cutoff - search_cutoff);
    const bool cell_list_succeeded = cell_list_cutoff_is_safe &&
        prepare_cell_list(
            structure.n_atoms,
            structure.n_shifts,
            structure.wrapped_positions,
            structure.translations,
            search_cutoff,
            inner_cutoff * inner_cutoff,
            structure.cell_list);
    if (cell_list_succeeded) {
        return;
    }
    neighbor_search::require_search(
        requested_algorithm != neighbor_search::Algorithm::CellList,
        "cell_list cannot safely process this structure; use "
        "algorithm='auto' or 'brute_force'");
    structure.algorithm = neighbor_search::Algorithm::BruteForce;
}


int64_t query_chunk_size(
    neighbor_search::Algorithm algorithm,
    int64_t n_atoms,
    int64_t n_shifts) {
    if (algorithm == neighbor_search::Algorithm::CellList) {
        return kCellListSourcesPerTask;
    }
    if (n_atoms == 0 || n_shifts == 0 ||
        n_atoms > std::numeric_limits<int64_t>::max() / n_shifts) {
        return 1;
    }
    const int64_t candidates_per_source = n_atoms * n_shifts;
    return std::max(
        int64_t{1}, kBruteForceCandidatesPerTask / candidates_per_source);
}


template <typename scalar_t, neighbor_search::PairMode Mode>
PairBuffers build_neighbor_pairs(
    std::span<const scalar_t> positions,
    std::span<const int64_t> batch_ptr,
    std::span<const scalar_t> cells,
    std::span<const scalar_t> duals,
    std::span<const int32_t> image_shifts,
    std::span<const int64_t> image_offsets,
    double cutoff,
    neighbor_search::Algorithm requested_algorithm,
    int64_t num_threads) {
    const scalar_t* position_data = positions.data();
    const int64_t* batch_ptr_data = batch_ptr.data();
    const scalar_t* cell_data = cells.data();
    const scalar_t* dual_data = duals.data();
    const int32_t* image_shift_data = image_shifts.data();
    const int64_t* image_offsets_data = image_offsets.data();
    const int64_t n_atoms_total = static_cast<int64_t>(positions.size() / 3);
    const int64_t batch_size = static_cast<int64_t>(batch_ptr.size() - 1);
    const scalar_t scalar_cutoff = static_cast<scalar_t>(cutoff);
    const scalar_t cutoff_squared = static_cast<scalar_t>(cutoff * cutoff);

    if (num_threads == 1) {
        PairBuffer chunk;
        chunk.indices.reserve(static_cast<size_t>(n_atoms_total) * 64);
        chunk.shifts.reserve(static_cast<size_t>(n_atoms_total) * 96);
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            PreparedStructure<scalar_t> structure;
            structure.atom_offset = batch_ptr_data[batch];
            structure.n_atoms =
                batch_ptr_data[batch + 1] - structure.atom_offset;
            structure.shift_offset = image_offsets_data[batch];
            structure.n_shifts =
                image_offsets_data[batch + 1] - structure.shift_offset;
            structure.positions = position_data + 3 * structure.atom_offset;
            structure.cell = cell_data + 9 * batch;
            structure.dual = dual_data + 9 * batch;
            structure.image_shifts =
                image_shift_data + 3 * structure.shift_offset;
            prepare_structure(
                structure, scalar_cutoff, requested_algorithm);
            search_prepared_structure<scalar_t, Mode>(
                structure,
                0,
                structure.n_atoms,
                cutoff_squared,
                chunk);
        }
        PairBuffers pairs;
        pairs.pair_count = chunk.indices.size() / 2;
        pairs.storage = std::move(chunk);
        return pairs;
    }

    std::vector<PreparedStructure<scalar_t>> structures(
        static_cast<size_t>(batch_size));
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        PreparedStructure<scalar_t>& structure = structures[batch];
        structure.atom_offset = batch_ptr_data[batch];
        structure.n_atoms =
            batch_ptr_data[batch + 1] - structure.atom_offset;
        structure.shift_offset = image_offsets_data[batch];
        structure.n_shifts =
            image_offsets_data[batch + 1] - structure.shift_offset;
        structure.positions = position_data + 3 * structure.atom_offset;
        structure.cell = cell_data + 9 * batch;
        structure.dual = dual_data + 9 * batch;
        structure.image_shifts =
            image_shift_data + 3 * structure.shift_offset;
        structure.algorithm = select_cpu_algorithm(
            requested_algorithm, structure.n_atoms, structure.n_shifts);
    }

    std::vector<uint8_t> split_structure(static_cast<size_t>(batch_size), 0);
    std::vector<int64_t> split_indices;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const PreparedStructure<scalar_t>& structure = structures[batch];
        const int64_t chunk_size = query_chunk_size(
            structure.algorithm, structure.n_atoms, structure.n_shifts);
        if (structure.n_atoms > chunk_size) {
            split_structure[batch] = 1;
            split_indices.push_back(batch);
        }
    }
    neighbor_search::parallel_for(
        static_cast<int64_t>(split_indices.size()),
        num_threads,
        [&](int64_t index) {
            prepare_structure(
                structures[split_indices[index]],
                scalar_cutoff,
                requested_algorithm);
        });

    std::vector<QueryTask> tasks;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const PreparedStructure<scalar_t>& structure = structures[batch];
        if (structure.n_atoms == 0) {
            continue;
        }
        if (!split_structure[batch]) {
            tasks.push_back({batch, 0, structure.n_atoms, true});
            continue;
        }
        const int64_t chunk_size = query_chunk_size(
            structure.algorithm, structure.n_atoms, structure.n_shifts);
        for (int64_t source = 0; source < structure.n_atoms;
             source += chunk_size) {
            tasks.push_back(
                {batch,
                 source,
                 std::min(structure.n_atoms, source + chunk_size),
                 false});
        }
    }

    std::vector<PairBuffer> task_pairs(tasks.size());
    neighbor_search::parallel_for(
        static_cast<int64_t>(tasks.size()),
        num_threads,
        [&](int64_t task_index) {
            const QueryTask& task = tasks[task_index];
            PairBuffer& local_pairs = task_pairs[task_index];
            PreparedStructure<scalar_t>& structure =
                structures[task.structure];
            if (task.prepare_structure) {
                prepare_structure(
                    structure, scalar_cutoff, requested_algorithm);
            }
            const int64_t source_count = task.source_end - task.source_begin;
            local_pairs.indices.reserve(static_cast<size_t>(source_count) * 64);
            local_pairs.shifts.reserve(static_cast<size_t>(source_count) * 96);
            search_prepared_structure<scalar_t, Mode>(
                structure,
                task.source_begin,
                task.source_end,
                cutoff_squared,
                local_pairs);
            if (task.prepare_structure) {
                structure = PreparedStructure<scalar_t>();
            }
        });

    size_t index_count = 0;
    size_t shift_count = 0;
    for (const PairBuffer& local_pairs : task_pairs) {
        neighbor_search::require_search(
            local_pairs.indices.size() <=
                std::numeric_limits<size_t>::max() - index_count,
            "neighbor-list output size exceeds addressable memory");
        neighbor_search::require_search(
            local_pairs.shifts.size() <=
                std::numeric_limits<size_t>::max() - shift_count,
            "neighbor-list output size exceeds addressable memory");
        index_count += local_pairs.indices.size();
        shift_count += local_pairs.shifts.size();
    }
    neighbor_search::require_search(
        index_count % 2 == 0 && shift_count % 3 == 0 &&
            index_count / 2 == shift_count / 3,
        "neighbor-list pair buffers are inconsistent");
    PairBuffers pairs;
    pairs.storage = std::move(task_pairs);
    pairs.pair_count = index_count / 2;
    return pairs;
}


template <typename scalar_t>
PairBuffers dispatch_neighbor_pairs(
    std::span<const scalar_t> positions,
    std::span<const int64_t> batch_ptr,
    std::span<const scalar_t> cells,
    std::span<const scalar_t> duals,
    std::span<const int32_t> image_shifts,
    std::span<const int64_t> image_offsets,
    double cutoff,
    neighbor_search::PairMode mode,
    neighbor_search::Algorithm algorithm,
    int64_t num_threads) {
    auto build = [&]<neighbor_search::PairMode Mode>() {
        return build_neighbor_pairs<scalar_t, Mode>(
            positions,
            batch_ptr,
            cells,
            duals,
            image_shifts,
            image_offsets,
            cutoff,
            algorithm,
            num_threads);
    };
    switch (mode) {
        case neighbor_search::PairMode::Full:
            return build.template operator()<neighbor_search::PairMode::Full>();
        case neighbor_search::PairMode::FullWithSelf:
            return build.template operator()<neighbor_search::PairMode::FullWithSelf>();
        case neighbor_search::PairMode::Half:
            return build.template operator()<neighbor_search::PairMode::Half>();
        case neighbor_search::PairMode::HalfWithSelf:
            return build.template operator()<neighbor_search::PairMode::HalfWithSelf>();
    }
    throw neighbor_search::SearchError("invalid pair mode");
}

}  // namespace


void neighbor_search::copy_pair_buffers(
    const PairBuffers& pairs,
    std::span<int64_t> indices,
    std::span<int32_t> shifts,
    int64_t num_threads) {
    require_search(
        pairs.pair_count <= std::numeric_limits<size_t>::max() / 3,
        "neighbor-list output size exceeds addressable memory");
    require_search(
        indices.size() == 2 * pairs.pair_count &&
            shifts.size() == 3 * pairs.pair_count,
        "neighbor-list output arrays have inconsistent sizes");
    if (const PairBuffer* buffer = std::get_if<PairBuffer>(&pairs.storage)) {
        require_search(
            buffer->indices.size() == indices.size() &&
                buffer->shifts.size() == shifts.size(),
            "neighbor-list pair buffers do not match the reported size");
        if (!buffer->indices.empty()) {
            std::memcpy(
                indices.data(),
                buffer->indices.data(),
                buffer->indices.size() * sizeof(int64_t));
            std::memcpy(
                shifts.data(),
                buffer->shifts.data(),
                buffer->shifts.size() * sizeof(int32_t));
        }
        return;
    }
    const std::vector<PairBuffer>& chunks =
        std::get<std::vector<PairBuffer>>(pairs.storage);
    std::vector<size_t> offsets(chunks.size() + 1, 0);
    for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        const PairBuffer& buffer = chunks[chunk];
        require_search(
            buffer.indices.size() % 2 == 0 &&
                buffer.shifts.size() % 3 == 0 &&
                buffer.indices.size() / 2 == buffer.shifts.size() / 3,
            "neighbor-list pair buffers are inconsistent");
        const size_t chunk_pairs = buffer.indices.size() / 2;
        require_search(
            chunk_pairs <= pairs.pair_count - offsets[chunk],
            "neighbor-list pair buffers exceed the reported size");
        offsets[chunk + 1] = offsets[chunk] + chunk_pairs;
    }
    require_search(
        offsets.back() == pairs.pair_count,
        "neighbor-list pair buffers do not match the reported size");
    parallel_for(
        static_cast<int64_t>(chunks.size()),
        num_threads,
        [&](int64_t chunk_index) {
            const PairBuffer& buffer = chunks[chunk_index];
            const size_t pair_offset = offsets[chunk_index];
            if (!buffer.indices.empty()) {
                std::memcpy(
                    indices.data() + 2 * pair_offset,
                    buffer.indices.data(),
                    buffer.indices.size() * sizeof(int64_t));
                std::memcpy(
                    shifts.data() + 3 * pair_offset,
                    buffer.shifts.data(),
                    buffer.shifts.size() * sizeof(int32_t));
            }
        });
}


template <typename scalar_t>
neighbor_search::PairBuffers neighbor_search::neighbor_list_cpu(
    std::span<const scalar_t> positions,
    std::span<const int64_t> batch_ptr,
    std::span<const scalar_t> cells,
    std::span<const uint8_t> pbc,
    double cutoff,
    PairMode mode,
    Algorithm algorithm,
    int64_t num_threads) {
    require_input(positions.size() % 3 == 0, "positions must have shape (N_total, 3)");
    require_input(!batch_ptr.empty(), "batch_ptr must have shape (B + 1,)");
    const int64_t n_atoms = static_cast<int64_t>(positions.size() / 3);
    const int64_t batch_size = static_cast<int64_t>(batch_ptr.size() - 1);
    require_input(
        cells.size() == static_cast<size_t>(9 * batch_size),
        "cell must have shape (B, 3, 3)");
    require_input(
        pbc.size() == static_cast<size_t>(3 * batch_size),
        "pbc must have shape (B, 3)");
    require_input(
        std::isfinite(cutoff) && cutoff > 0,
        "cutoff must be finite and positive");
    require_input(num_threads > 0, "num_threads must be a positive integer");
    require_input(batch_ptr.front() == 0, "batch_ptr must start at zero");
    require_input(batch_ptr.back() == n_atoms, "batch_ptr must end at N_total");
    require_input(
        n_atoms < std::numeric_limits<int32_t>::max(),
        "the current implementation supports fewer than 2^31 atoms");

    std::vector<int64_t> atom_counts;
    atom_counts.reserve(static_cast<size_t>(batch_size));
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        require_input(
            batch_ptr[batch + 1] >= batch_ptr[batch],
            "batch_ptr must be nondecreasing");
        atom_counts.push_back(batch_ptr[batch + 1] - batch_ptr[batch]);
    }
    std::vector<double> cell_double(cells.begin(), cells.end());
    const PeriodicMetadata metadata = build_periodic_metadata(
        cell_double, pbc, atom_counts, cutoff);
    std::vector<scalar_t> duals(
        metadata.duals.begin(), metadata.duals.end());
    return dispatch_neighbor_pairs<scalar_t>(
        positions,
        batch_ptr,
        cells,
        duals,
        metadata.image_shifts,
        metadata.image_offsets,
        cutoff,
        mode,
        algorithm,
        num_threads);
}


template neighbor_search::PairBuffers neighbor_search::neighbor_list_cpu<float>(
    std::span<const float>,
    std::span<const int64_t>,
    std::span<const float>,
    std::span<const uint8_t>,
    double,
    PairMode,
    Algorithm,
    int64_t);


template neighbor_search::PairBuffers neighbor_search::neighbor_list_cpu<double>(
    std::span<const double>,
    std::span<const int64_t>,
    std::span<const double>,
    std::span<const uint8_t>,
    double,
    PairMode,
    Algorithm,
    int64_t);
