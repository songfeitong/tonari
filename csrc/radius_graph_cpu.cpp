#include "radius_graph_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>


namespace {

// The crossover was selected from a real-structure threshold sweep.
constexpr int64_t kExhaustiveCandidateLimit = 16384;
// Avoid pathological dense allocation for sparse finite coordinates.
constexpr int64_t kMaximumDenseBins = int64_t{1} << 26;
constexpr int64_t kMaximumBinsPerImage = 64;
constexpr int kCellListRoundoffFactor = 64;


struct GraphBuffers {
    std::vector<int64_t> sources;
    std::vector<int64_t> targets;
    std::vector<int32_t> shifts;
};


struct CellNode {
    int32_t source;
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


void validate_cpu_inputs(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr) {
    TORCH_CHECK(!positions.is_cuda(), "positions must be a CPU tensor");
    TORCH_CHECK(
        !ptr.is_cuda() && !cells.is_cuda() && !duals.is_cuda(),
        "all inputs must be CPU tensors");
    TORCH_CHECK(
        !image_shifts.is_cuda() && !image_ptr.is_cuda(),
        "all metadata must be CPU tensors");
    TORCH_CHECK(
        positions.is_contiguous() && ptr.is_contiguous() && cells.is_contiguous(),
        "inputs must be contiguous");
    TORCH_CHECK(
        duals.is_contiguous() && image_shifts.is_contiguous() &&
            image_ptr.is_contiguous(),
        "metadata must be contiguous");
    TORCH_CHECK(
        positions.scalar_type() == cells.scalar_type() &&
            positions.scalar_type() == duals.scalar_type(),
        "positions, cells, and duals must have the same dtype");
    TORCH_CHECK(ptr.scalar_type() == torch::kInt64, "ptr must have dtype int64");
    TORCH_CHECK(
        image_shifts.scalar_type() == torch::kInt32,
        "image_shifts must have dtype int32");
    TORCH_CHECK(
        image_ptr.scalar_type() == torch::kInt64,
        "image_ptr must have dtype int64");
}


bool candidate_count_at_most(
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
            TORCH_CHECK(
                std::isfinite(position[cartesian]),
                "positions must contain only finite values");
        }
        for (int axis = 0; axis < 3; ++axis) {
            long double fractional = 0.0L;
            for (int cartesian = 0; cartesian < 3; ++cartesian) {
                fractional += static_cast<long double>(position[cartesian]) *
                    static_cast<long double>(dual[3 * cartesian + axis]);
            }
            TORCH_CHECK(
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


template <typename scalar_t>
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
    GraphBuffers& graph) {
    const int32_t* search_shift = image_shifts + 3 * shift;
    if (source == target && search_shift[0] == 0 && search_shift[1] == 0 &&
        search_shift[2] == 0) {
        return;
    }
    int64_t output_shift[3];
    for (int axis = 0; axis < 3; ++axis) {
        output_shift[axis] = static_cast<int64_t>(search_shift[axis]) -
            static_cast<int64_t>(atom_wraps[3 * source + axis]) +
            static_cast<int64_t>(atom_wraps[3 * target + axis]);
    }
    if (!distance_known_inside) {
        scalar_t distance_squared = scalar_t(0);
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            scalar_t component = positions[3 * source + cartesian] -
                positions[3 * target + cartesian];
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
        TORCH_CHECK(
            value >= std::numeric_limits<int32_t>::min() &&
                value <= std::numeric_limits<int32_t>::max(),
            "a cell shift required by the cutoff graph exceeds the int32 output range");
    }
    graph.sources.push_back(atom_offset + source);
    graph.targets.push_back(atom_offset + target);
    for (const int64_t value : output_shift) {
        graph.shifts.push_back(static_cast<int32_t>(value));
    }
}


template <typename scalar_t>
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
    GraphBuffers& graph) {
    scalar_t search_distance_squared = scalar_t(0);
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        const scalar_t component = wrapped_positions[3 * source + cartesian] -
            wrapped_positions[3 * target + cartesian] +
            translations[3 * shift + cartesian];
        search_distance_squared += component * component;
    }
    if (search_distance_squared >= search_cutoff_squared) {
        return;
    }
    append_candidate(
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
        graph);
}


template <typename scalar_t>
void search_exhaustive(
    int64_t atom_offset,
    int64_t n_atoms,
    int64_t n_shifts,
    const scalar_t* positions,
    const scalar_t* cell,
    const std::vector<int32_t>& atom_wraps,
    const int32_t* image_shifts,
    scalar_t cutoff_squared,
    GraphBuffers& graph) {
    for (int64_t target = 0; target < n_atoms; ++target) {
        for (int64_t source = 0; source < n_atoms; ++source) {
            for (int64_t shift = 0; shift < n_shifts; ++shift) {
                append_candidate(
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
                    graph);
            }
        }
    }
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
bool search_cell_list(
    int64_t atom_offset,
    int64_t n_atoms,
    int64_t n_shifts,
    const scalar_t* positions,
    const scalar_t* cell,
    const std::vector<scalar_t>& wrapped_positions,
    const std::vector<int32_t>& atom_wraps,
    const int32_t* image_shifts,
    const std::vector<scalar_t>& translations,
    scalar_t search_cutoff,
    scalar_t inner_cutoff_squared,
    scalar_t cutoff_squared,
    GraphBuffers& graph) {
    TORCH_CHECK(
        n_atoms < std::numeric_limits<int32_t>::max() &&
            n_shifts < std::numeric_limits<int32_t>::max(),
        "cell-list atoms and image shifts must fit int32 indexing");
    scalar_t bounds_minimum[3];
    scalar_t bounds_maximum[3];
    BinLayout<scalar_t> layout;
    if (!build_bin_layout(
            wrapped_positions,
            n_atoms,
            n_shifts,
            search_cutoff,
            bounds_minimum,
            bounds_maximum,
            layout)) {
        return false;
    }
    std::vector<int32_t> bin_heads(layout.count, -1);
    std::vector<CellNode> nodes;
    const int64_t possible_images = n_atoms * n_shifts;
    nodes.reserve(static_cast<size_t>(std::min(possible_images, 4 * n_atoms)));
    for (int64_t source = 0; source < n_atoms; ++source) {
        for (int64_t shift = 0; shift < n_shifts; ++shift) {
            scalar_t image_position[3];
            bool inside_bounds = true;
            for (int cartesian = 0; cartesian < 3; ++cartesian) {
                image_position[cartesian] =
                    wrapped_positions[3 * source + cartesian] +
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
            const int64_t bin = bin_index(image_position, layout);
            TORCH_CHECK(
                nodes.size() < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "cell-list node count exceeds int32 indexing");
            const int32_t node = static_cast<int32_t>(nodes.size());
            nodes.push_back(
                {static_cast<int32_t>(source),
                 static_cast<int32_t>(shift),
                 bin_heads[bin]});
            bin_heads[bin] = node;
        }
    }
    const scalar_t search_cutoff_squared = search_cutoff * search_cutoff;
    for (int64_t target = 0; target < n_atoms; ++target) {
        scalar_t target_position[3];
        int64_t target_coordinates[3];
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            target_position[cartesian] =
                wrapped_positions[3 * target + cartesian];
            target_coordinates[cartesian] = std::clamp(
                static_cast<int64_t>(std::floor(
                    (target_position[cartesian] - layout.origins[cartesian]) /
                    layout.size)),
                int64_t{0},
                layout.dimensions[cartesian] - 1);
        }
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            const int64_t x = target_coordinates[0] + offset_x;
            if (x < 0 || x >= layout.dimensions[0]) {
                continue;
            }
            for (int offset_y = -1; offset_y <= 1; ++offset_y) {
                const int64_t y = target_coordinates[1] + offset_y;
                if (y < 0 || y >= layout.dimensions[1]) {
                    continue;
                }
                for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                    const int64_t z = target_coordinates[2] + offset_z;
                    if (z < 0 || z >= layout.dimensions[2]) {
                        continue;
                    }
                    const int64_t bin =
                        (x * layout.dimensions[1] + y) * layout.dimensions[2] + z;
                    for (int32_t node = bin_heads[bin]; node >= 0;
                         node = nodes[node].next) {
                        append_cell_candidate(
                            nodes[node].source,
                            target,
                            atom_offset,
                            nodes[node].shift,
                            positions,
                            cell,
                            wrapped_positions,
                            atom_wraps,
                            image_shifts,
                            translations,
                            search_cutoff_squared,
                            inner_cutoff_squared,
                            cutoff_squared,
                            graph);
                    }
                }
            }
        }
    }
    return true;
}


template <typename scalar_t>
GraphBuffers build_radius_graph(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    double cutoff) {
    const scalar_t* position_data = positions.data_ptr<scalar_t>();
    const int64_t* ptr_data = ptr.data_ptr<int64_t>();
    const scalar_t* cell_data = cells.data_ptr<scalar_t>();
    const scalar_t* dual_data = duals.data_ptr<scalar_t>();
    const int32_t* image_shift_data = image_shifts.data_ptr<int32_t>();
    const int64_t* image_ptr_data = image_ptr.data_ptr<int64_t>();
    const int64_t n_atoms_total = positions.size(0);
    const int64_t batch_size = ptr.numel() - 1;
    const scalar_t scalar_cutoff = static_cast<scalar_t>(cutoff);
    const scalar_t cutoff_squared = static_cast<scalar_t>(cutoff * cutoff);

    GraphBuffers graph;
    graph.sources.reserve(static_cast<size_t>(n_atoms_total) * 32);
    graph.targets.reserve(static_cast<size_t>(n_atoms_total) * 32);
    graph.shifts.reserve(static_cast<size_t>(n_atoms_total) * 96);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t atom_offset = ptr_data[batch];
        const int64_t n_atoms = ptr_data[batch + 1] - atom_offset;
        if (n_atoms == 0) {
            continue;
        }
        const int64_t shift_offset = image_ptr_data[batch];
        const int64_t n_shifts = image_ptr_data[batch + 1] - shift_offset;
        const scalar_t* cell = cell_data + 9 * batch;
        std::vector<scalar_t> wrapped_positions;
        std::vector<int32_t> atom_wraps;
        prepare_positions(
            position_data + 3 * atom_offset,
            cell,
            dual_data + 9 * batch,
            n_atoms,
            wrapped_positions,
            atom_wraps);
        const int32_t* structure_shifts =
            image_shift_data + 3 * shift_offset;
        if (candidate_count_at_most(
                n_atoms, n_shifts, kExhaustiveCandidateLimit)) {
            search_exhaustive(
                atom_offset,
                n_atoms,
                n_shifts,
                position_data + 3 * atom_offset,
                cell,
                atom_wraps,
                structure_shifts,
                cutoff_squared,
                graph);
            continue;
        }
        const auto translations =
            image_translations(structure_shifts, cell, n_shifts);
        scalar_t search_cutoff = scalar_t(0);
        const bool cell_list_cutoff_is_safe = conservative_cell_list_cutoff(
                position_data + 3 * atom_offset,
                cell,
                n_atoms,
                wrapped_positions,
                atom_wraps,
                structure_shifts,
                n_shifts,
                scalar_cutoff,
                search_cutoff);
        // Candidates safely inside the error band keep the fast wrapped
        // distance; only the boundary shell needs the direct public formula.
        const scalar_t inner_cutoff = std::max(
            scalar_t(0), 2 * scalar_cutoff - search_cutoff);
        if (!cell_list_cutoff_is_safe ||
            !search_cell_list(
                atom_offset,
                n_atoms,
                n_shifts,
                position_data + 3 * atom_offset,
                cell,
                wrapped_positions,
                atom_wraps,
                structure_shifts,
                translations,
                search_cutoff,
                inner_cutoff * inner_cutoff,
                cutoff_squared,
                graph)) {
            search_exhaustive(
                atom_offset,
                n_atoms,
                n_shifts,
                position_data + 3 * atom_offset,
                cell,
                atom_wraps,
                structure_shifts,
                cutoff_squared,
                graph);
        }
    }
    return graph;
}

}  // namespace


std::vector<torch::Tensor> radius_graph_pbc_cpu(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    double cutoff) {
    validate_cpu_inputs(
        positions, ptr, cells, duals, image_shifts, image_ptr);
    GraphBuffers graph;
    AT_DISPATCH_FLOATING_TYPES(
        positions.scalar_type(), "radius_graph_pbc_cpu", [&] {
            graph = build_radius_graph<scalar_t>(
                positions,
                ptr,
                cells,
                duals,
                image_shifts,
                image_ptr,
                cutoff);
        });

    const int64_t n_edges = static_cast<int64_t>(graph.sources.size());
    auto edge_index =
        torch::empty({2, n_edges}, positions.options().dtype(torch::kInt64));
    auto output_shifts =
        torch::empty({n_edges, 3}, positions.options().dtype(torch::kInt32));
    if (n_edges > 0) {
        int64_t* edge_data = edge_index.data_ptr<int64_t>();
        std::memcpy(
            edge_data,
            graph.sources.data(),
            static_cast<size_t>(n_edges) * sizeof(int64_t));
        std::memcpy(
            edge_data + n_edges,
            graph.targets.data(),
            static_cast<size_t>(n_edges) * sizeof(int64_t));
        std::memcpy(
            output_shifts.data_ptr<int32_t>(),
            graph.shifts.data(),
            static_cast<size_t>(3 * n_edges) * sizeof(int32_t));
    }
    return {edge_index, output_shifts};
}
