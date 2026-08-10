#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "neighbors_cuda.h"
#include "../core/pair_policy.h"

#include <cstdint>
#include <limits>


namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int64_t kMaximumDenseBins = int64_t{1} << 28;
constexpr int64_t kMaximumBinsPerNode = 64;


__device__ __forceinline__ int64_t segment_for_index(
    int64_t index,
    const int64_t* segment_offsets,
    int64_t n_segments) {
    int64_t lower = 0;
    int64_t upper = n_segments;
    while (lower < upper) {
        const int64_t middle = lower + (upper - lower) / 2;
        if (segment_offsets[middle + 1] <= index) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}


template <typename scalar_t>
__device__ __forceinline__ void atomic_minimum(scalar_t* address, scalar_t value);


template <>
__device__ __forceinline__ void atomic_minimum<float>(float* address, float value) {
    int* integer_address = reinterpret_cast<int*>(address);
    int old = *integer_address;
    while (value < __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(integer_address, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}


template <>
__device__ __forceinline__ void atomic_minimum<double>(double* address, double value) {
    auto* integer_address = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *integer_address;
    while (value < __longlong_as_double(old)) {
        const unsigned long long assumed = old;
        old = atomicCAS(integer_address, assumed, __double_as_longlong(value));
        if (old == assumed) {
            break;
        }
    }
}


template <typename scalar_t>
__device__ __forceinline__ void atomic_maximum(scalar_t* address, scalar_t value);


template <>
__device__ __forceinline__ void atomic_maximum<float>(float* address, float value) {
    int* integer_address = reinterpret_cast<int*>(address);
    int old = *integer_address;
    while (value > __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(integer_address, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}


template <>
__device__ __forceinline__ void atomic_maximum<double>(double* address, double value) {
    auto* integer_address = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *integer_address;
    while (value > __longlong_as_double(old)) {
        const unsigned long long assumed = old;
        old = atomicCAS(integer_address, assumed, __double_as_longlong(value));
        if (old == assumed) {
            break;
        }
    }
}


template <typename scalar_t>
__global__ void wrap_positions_kernel(
    const scalar_t* positions,
    const int64_t* offsets,
    const scalar_t* cells,
    const scalar_t* duals,
    int64_t n_atoms,
    int64_t batch_size,
    scalar_t* wrapped_positions,
    int32_t* atom_wraps,
    scalar_t* bounds_minimum,
    scalar_t* bounds_maximum,
    int64_t* input_status) {
    const int64_t atom = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) {
        return;
    }
    const int64_t batch = segment_for_index(atom, offsets, batch_size);
    const scalar_t* position = positions + 3 * atom;
    const scalar_t* cell = cells + 9 * batch;
    const scalar_t* dual = duals + 9 * batch;
    bool finite = true;
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        finite &= isfinite(position[cartesian]);
    }
    if (!finite) {
        atomicOr(
            reinterpret_cast<unsigned long long*>(input_status),
            static_cast<unsigned long long>(1));
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            atom_wraps[3 * atom + axis] = 0;
            wrapped_positions[3 * atom + axis] = scalar_t(0);
            atomic_minimum(bounds_minimum + 3 * batch + axis, scalar_t(0));
            atomic_maximum(bounds_maximum + 3 * batch + axis, scalar_t(0));
        }
        return;
    }
    int32_t wraps[3];
    const scalar_t upper_bound = static_cast<scalar_t>(2147483648.0);
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        scalar_t fractional = scalar_t(0);
#pragma unroll
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            fractional += position[cartesian] * dual[3 * cartesian + axis];
        }
        if (!isfinite(fractional) || fractional < -upper_bound ||
            fractional >= upper_bound) {
            atomicOr(
                reinterpret_cast<unsigned long long*>(input_status),
                static_cast<unsigned long long>(1));
            wraps[axis] = 0;
        } else {
            wraps[axis] = static_cast<int32_t>(floor(fractional));
        }
        atom_wraps[3 * atom + axis] = wraps[axis];
    }
    if (wraps[0] != 0 || wraps[1] != 0 || wraps[2] != 0) {
        // Wrapped arithmetic is only a search aid. For unwrapped
        // representatives, use the exhaustive public-vector predicate to
        // avoid a device-dependent cutoff decision from cancellation.
        atomicOr(
            reinterpret_cast<unsigned long long*>(input_status),
            static_cast<unsigned long long>(2));
    }
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        scalar_t wrapped = position[cartesian];
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            wrapped -= static_cast<scalar_t>(wraps[axis]) * cell[3 * axis + cartesian];
        }
        wrapped_positions[3 * atom + cartesian] = wrapped;
        atomic_minimum(bounds_minimum + 3 * batch + cartesian, wrapped);
        atomic_maximum(bounds_maximum + 3 * batch + cartesian, wrapped);
    }
}


template <typename scalar_t>
__global__ void define_bins_kernel(
    const int64_t* offsets,
    const scalar_t* bounds_minimum,
    const scalar_t* bounds_maximum,
    int64_t batch_size,
    scalar_t cutoff,
    scalar_t* bin_origins,
    int64_t* bin_dimensions,
    int64_t* bin_counts) {
    const int64_t batch = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch > batch_size) {
        return;
    }
    if (batch == batch_size) {
        // The final slot is a status word carried through the same cumsum as
        // the bin counts, so the existing host read returns both values.
        return;
    }
    int64_t count = 1;
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        int64_t dimension = 1;
        scalar_t origin = scalar_t(0);
        if (offsets[batch + 1] > offsets[batch]) {
            const scalar_t minimum = bounds_minimum[3 * batch + cartesian];
            const scalar_t maximum = bounds_maximum[3 * batch + cartesian];
            origin = minimum - cutoff;
            const scalar_t dimension_value =
                ceil((maximum - minimum + 2 * cutoff) / cutoff);
            dimension = !isfinite(dimension_value) ||
                    dimension_value > static_cast<scalar_t>(kMaximumDenseBins)
                ? kMaximumDenseBins + 1
                : max(int64_t{1}, static_cast<int64_t>(dimension_value));
        }
        bin_origins[3 * batch + cartesian] = origin;
        bin_dimensions[3 * batch + cartesian] = dimension;
        if (count <= kMaximumDenseBins / dimension) {
            count *= dimension;
        } else {
            // Exact counts above the allocation limit are irrelevant. The
            // saturated value keeps the batched cumsum representable and
            // deterministically selects the exhaustive fallback.
            count = kMaximumDenseBins + 1;
            break;
        }
    }
    bin_counts[batch] = count;
}


template <typename scalar_t>
__global__ void insert_images_kernel(
    const int64_t* offsets,
    const scalar_t* cells,
    const scalar_t* wrapped_positions,
    const int32_t* image_shifts,
    const int64_t* image_offsets,
    const int64_t* node_offsets,
    const scalar_t* bounds_minimum,
    const scalar_t* bounds_maximum,
    const scalar_t* bin_origins,
    const int64_t* bin_dimensions,
    const int64_t* bin_offsets,
    int64_t total_nodes,
    int64_t batch_size,
    scalar_t cutoff,
    int32_t* bin_heads,
    int32_t* node_next) {
    const int64_t node = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= total_nodes) {
        return;
    }
    const int64_t batch = segment_for_index(node, node_offsets, batch_size);
    const int64_t local_node = node - node_offsets[batch];
    const int64_t n_shifts = image_offsets[batch + 1] - image_offsets[batch];
    const int64_t local_target = local_node / n_shifts;
    const int64_t shift_index = local_node % n_shifts;
    const int64_t target = offsets[batch] + local_target;
    const scalar_t* cell = cells + 9 * batch;
    const int32_t* shift = image_shifts + 3 * (image_offsets[batch] + shift_index);
    int64_t coordinates[3];
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        scalar_t image_position = wrapped_positions[3 * target + cartesian];
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            image_position += static_cast<scalar_t>(shift[axis]) * cell[3 * axis + cartesian];
        }
        if (image_position < bounds_minimum[3 * batch + cartesian] - cutoff ||
            image_position > bounds_maximum[3 * batch + cartesian] + cutoff) {
            return;
        }
        const int64_t dimension = bin_dimensions[3 * batch + cartesian];
        coordinates[cartesian] = min(
            dimension - 1,
            max(
                int64_t{0},
                static_cast<int64_t>(floor(
                    (image_position - bin_origins[3 * batch + cartesian]) / cutoff))));
    }
    const int64_t local_bin =
        (coordinates[0] * bin_dimensions[3 * batch + 1] + coordinates[1]) *
            bin_dimensions[3 * batch + 2] +
        coordinates[2];
    const int64_t bin = bin_offsets[batch] + local_bin;
    node_next[node] = atomicExch(bin_heads + bin, static_cast<int32_t>(node));
}


template <typename scalar_t, bool write_pairs, neighbor_search::PairMode Mode>
__global__ void query_bins_kernel(
    const int64_t* offsets,
    const scalar_t* cells,
    const scalar_t* wrapped_positions,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_offsets,
    const int64_t* node_offsets,
    const scalar_t* bin_origins,
    const int64_t* bin_dimensions,
    const int64_t* bin_offsets,
    const int32_t* bin_heads,
    const int32_t* node_next,
    int64_t n_atoms,
    int64_t batch_size,
    scalar_t cutoff,
    scalar_t cutoff_squared,
    const int64_t* pair_offsets,
    int64_t* source_pair_counts,
    int64_t* shift_overflow,
    int64_t n_pairs,
    int64_t* pair_indices,
    int32_t* cell_shifts) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int64_t source =
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) + warp;
    if (source >= n_atoms) {
        return;
    }
    const int64_t batch = segment_for_index(source, offsets, batch_size);
    const scalar_t* cell = cells + 9 * batch;
    int64_t source_bin[3];
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        const int64_t dimension = bin_dimensions[3 * batch + cartesian];
        source_bin[cartesian] = min(
            dimension - 1,
            max(
                int64_t{0},
                static_cast<int64_t>(floor(
                    (wrapped_positions[3 * source + cartesian] -
                     bin_origins[3 * batch + cartesian]) /
                    cutoff))));
    }

    __shared__ int warp_output_cursors[kWarpsPerBlock];
    if constexpr (write_pairs) {
        if (lane == 0) {
            warp_output_cursors[warp] = 0;
        }
        __syncwarp();
    }

    int lane_count = 0;
    if (lane < 27) {
        const int offset_x = lane / 9 - 1;
        const int offset_y = (lane / 3) % 3 - 1;
        const int offset_z = lane % 3 - 1;
        const int64_t coordinate_x = source_bin[0] + offset_x;
        const int64_t coordinate_y = source_bin[1] + offset_y;
        const int64_t coordinate_z = source_bin[2] + offset_z;
        if (coordinate_x >= 0 && coordinate_x < bin_dimensions[3 * batch] &&
            coordinate_y >= 0 && coordinate_y < bin_dimensions[3 * batch + 1] &&
            coordinate_z >= 0 && coordinate_z < bin_dimensions[3 * batch + 2]) {
            const int64_t local_bin =
                (coordinate_x * bin_dimensions[3 * batch + 1] + coordinate_y) *
                    bin_dimensions[3 * batch + 2] +
                coordinate_z;
            int32_t node = bin_heads[bin_offsets[batch] + local_bin];
            while (node >= 0) {
                const int64_t local_node = node - node_offsets[batch];
                const int64_t n_shifts = image_offsets[batch + 1] - image_offsets[batch];
                const int64_t local_target = local_node / n_shifts;
                const int64_t shift_index = local_node % n_shifts;
                const int64_t target = offsets[batch] + local_target;
                const int32_t* wrapped_shift =
                    image_shifts + 3 * (image_offsets[batch] + shift_index);
                int64_t output_shift[3];
                scalar_t distance_squared = scalar_t(0);
#pragma unroll
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    scalar_t component =
                        wrapped_positions[3 * target + cartesian] -
                        wrapped_positions[3 * source + cartesian];
#pragma unroll
                    for (int axis = 0; axis < 3; ++axis) {
                        component += static_cast<scalar_t>(wrapped_shift[axis]) *
                            cell[3 * axis + cartesian];
                    }
                    distance_squared += component * component;
                }
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) {
                    output_shift[axis] = static_cast<int64_t>(wrapped_shift[axis]) -
                        static_cast<int64_t>(atom_wraps[3 * target + axis]) +
                        static_cast<int64_t>(atom_wraps[3 * source + axis]);
                }
                const bool zero_shift_self =
                    neighbor_search::is_zero_shift_self_pair(source, target, output_shift);
                if (neighbor_search::keep_pair_identity<Mode>(
                        source, target, output_shift) &&
                    (zero_shift_self || distance_squared < cutoff_squared)) {
                    bool shift_fits = true;
#pragma unroll
                    for (int axis = 0; axis < 3; ++axis) {
                        shift_fits &=
                            output_shift[axis] >= std::numeric_limits<int32_t>::min() &&
                            output_shift[axis] <= std::numeric_limits<int32_t>::max();
                    }
                    if (!shift_fits) {
                        if (shift_overflow != nullptr) {
                            atomicExch(
                                reinterpret_cast<unsigned long long*>(shift_overflow),
                                static_cast<unsigned long long>(
                                    std::numeric_limits<int64_t>::min()));
                        }
                        node = node_next[node];
                        continue;
                    }
                    if constexpr (write_pairs) {
                        const int local_output = atomicAdd(warp_output_cursors + warp, 1);
                        const int64_t output = pair_offsets[source] + local_output;
                        pair_indices[output] = source;
                        pair_indices[n_pairs + output] = target;
#pragma unroll
                        for (int axis = 0; axis < 3; ++axis) {
                            cell_shifts[3 * output + axis] =
                                static_cast<int32_t>(output_shift[axis]);
                        }
                    } else {
                        ++lane_count;
                    }
                }
                node = node_next[node];
            }
        }
    }

    if constexpr (!write_pairs) {
#pragma unroll
        for (int delta = kWarpSize / 2; delta > 0; delta /= 2) {
            lane_count += __shfl_down_sync(0xffffffff, lane_count, delta);
        }
        if (lane == 0) {
            source_pair_counts[source] = lane_count;
        }
    }
}


template <typename scalar_t>
struct QueryArguments {
    const int64_t* offsets;
    const scalar_t* cells;
    const scalar_t* wrapped_positions;
    const int32_t* atom_wraps;
    const int32_t* image_shifts;
    const int64_t* image_offsets;
    const int64_t* node_offsets;
    const scalar_t* bin_origins;
    const int64_t* bin_dimensions;
    const int64_t* bin_offsets;
    const int32_t* bin_heads;
    const int32_t* node_next;
    int64_t n_atoms;
    int64_t batch_size;
    scalar_t cutoff;
    scalar_t cutoff_squared;
    const int64_t* pair_offsets;
    int64_t* source_pair_counts;
    int64_t* shift_overflow;
    int64_t n_pairs;
    int64_t* pair_indices;
    int32_t* cell_shifts;
};


template <typename scalar_t, bool write_pairs, neighbor_search::PairMode Mode>
void launch_query_bins(
    const QueryArguments<scalar_t>& arguments,
    int64_t blocks,
    cudaStream_t stream) {
    query_bins_kernel<scalar_t, write_pairs, Mode>
        <<<blocks, kThreads, 0, stream>>>(
            arguments.offsets,
            arguments.cells,
            arguments.wrapped_positions,
            arguments.atom_wraps,
            arguments.image_shifts,
            arguments.image_offsets,
            arguments.node_offsets,
            arguments.bin_origins,
            arguments.bin_dimensions,
            arguments.bin_offsets,
            arguments.bin_heads,
            arguments.node_next,
            arguments.n_atoms,
            arguments.batch_size,
            arguments.cutoff,
            arguments.cutoff_squared,
            arguments.pair_offsets,
            arguments.source_pair_counts,
            arguments.shift_overflow,
            arguments.n_pairs,
            arguments.pair_indices,
            arguments.cell_shifts);
}


template <typename scalar_t, bool write_pairs>
void dispatch_query_bins(
    neighbor_search::PairMode mode,
    const QueryArguments<scalar_t>& arguments,
    int64_t blocks,
    cudaStream_t stream) {
    switch (mode) {
        case neighbor_search::PairMode::Full:
            launch_query_bins<scalar_t, write_pairs, neighbor_search::PairMode::Full>(
                arguments, blocks, stream);
            break;
        case neighbor_search::PairMode::FullWithSelf:
            launch_query_bins<
                scalar_t,
                write_pairs,
                neighbor_search::PairMode::FullWithSelf>(arguments, blocks, stream);
            break;
        case neighbor_search::PairMode::Half:
            launch_query_bins<scalar_t, write_pairs, neighbor_search::PairMode::Half>(
                arguments, blocks, stream);
            break;
        case neighbor_search::PairMode::HalfWithSelf:
            launch_query_bins<
                scalar_t,
                write_pairs,
                neighbor_search::PairMode::HalfWithSelf>(arguments, blocks, stream);
            break;
    }
}


void validate_cell_inputs(
    const torch::Tensor& positions,
    const torch::Tensor& offsets,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    const torch::Tensor& node_offsets,
    int64_t total_nodes) {
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(offsets.is_cuda() && cells.is_cuda() && duals.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(image_shifts.is_cuda() && image_offsets.is_cuda(), "all metadata must be CUDA tensors");
    TORCH_CHECK(block_offsets.is_cuda() && node_offsets.is_cuda(), "all metadata must be CUDA tensors");
    TORCH_CHECK(positions.is_contiguous() && offsets.is_contiguous() && cells.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(duals.is_contiguous() && image_shifts.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(image_offsets.is_contiguous() && block_offsets.is_contiguous() && node_offsets.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(positions.scalar_type() == cells.scalar_type(), "positions and cells must have the same dtype");
    TORCH_CHECK(positions.scalar_type() == duals.scalar_type(), "positions and duals must have the same dtype");
    TORCH_CHECK(total_nodes >= 0 && total_nodes < (int64_t{1} << 31), "cell-list node count exceeds int32 indexing");
}

}  // namespace


std::vector<torch::Tensor> find_neighbors_cuda_cell(
    const torch::Tensor& positions,
    const torch::Tensor& offsets,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    const torch::Tensor& node_offsets,
    int64_t total_blocks,
    int64_t total_nodes,
    double cutoff,
    bool half_list,
    bool include_self) {
    validate_cell_inputs(
        positions,
        offsets,
        cells,
        duals,
        image_shifts,
        image_offsets,
        block_offsets,
        node_offsets,
        total_nodes);
    const c10::cuda::CUDAGuard device_guard(positions.device());
    const auto stream = at::cuda::getCurrentCUDAStream(positions.get_device());
    const int64_t n_atoms = positions.size(0);
    const int64_t batch_size = offsets.numel() - 1;
    const auto mode = neighbor_search::pair_mode(half_list, include_self);
    auto pair_indices = torch::empty({2, 0}, positions.options().dtype(torch::kInt64));
    auto cell_shifts = torch::empty({0, 3}, positions.options().dtype(torch::kInt32));
    if (n_atoms == 0) {
        return {pair_indices, cell_shifts};
    }

    auto wrapped_positions = torch::empty_like(positions);
    auto atom_wraps = torch::empty({n_atoms, 3}, positions.options().dtype(torch::kInt32));
    auto bounds_minimum = torch::full(
        {batch_size, 3}, std::numeric_limits<double>::infinity(), positions.options());
    auto bounds_maximum = torch::full(
        {batch_size, 3}, -std::numeric_limits<double>::infinity(), positions.options());
    auto bin_counts = torch::empty(
        {batch_size + 1}, positions.options().dtype(torch::kInt64));
    C10_CUDA_CHECK(cudaMemsetAsync(
        bin_counts.data_ptr<int64_t>() + batch_size,
        0,
        sizeof(int64_t),
        stream));
    const int64_t atom_blocks = (n_atoms + kThreads - 1) / kThreads;
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "wrap_neighbor_search_positions", [&] {
        wrap_positions_kernel<scalar_t><<<atom_blocks, kThreads, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            offsets.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            duals.data_ptr<scalar_t>(),
            n_atoms,
            batch_size,
            wrapped_positions.data_ptr<scalar_t>(),
            atom_wraps.data_ptr<int32_t>(),
            bounds_minimum.data_ptr<scalar_t>(),
            bounds_maximum.data_ptr<scalar_t>(),
            bin_counts.data_ptr<int64_t>() + batch_size);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto bin_origins = torch::empty({batch_size, 3}, positions.options());
    auto bin_dimensions = torch::empty(
        {batch_size, 3}, positions.options().dtype(torch::kInt64));
    const int64_t batch_blocks = (batch_size + 1 + kThreads - 1) / kThreads;
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "define_neighbor_search_bins", [&] {
        define_bins_kernel<scalar_t><<<batch_blocks, kThreads, 0, stream>>>(
            offsets.data_ptr<int64_t>(),
            bounds_minimum.data_ptr<scalar_t>(),
            bounds_maximum.data_ptr<scalar_t>(),
            batch_size,
            static_cast<scalar_t>(cutoff),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_counts.data_ptr<int64_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto bin_offsets = torch::empty(
        {batch_size + 2}, positions.options().dtype(torch::kInt64));
    C10_CUDA_CHECK(cudaMemsetAsync(
        bin_offsets.data_ptr<int64_t>(), 0, sizeof(int64_t), stream));
    bin_offsets.slice(0, 1, batch_size + 2)
        .copy_(torch::cumsum(bin_counts, 0, torch::kInt64));
    const auto bin_result_cpu =
        bin_offsets.slice(0, batch_size, batch_size + 2).cpu();
    const int64_t* bin_result = bin_result_cpu.data_ptr<int64_t>();
    const int64_t total_bins = bin_result[0];
    TORCH_CHECK(total_bins >= 0, "cell-list bin count exceeds the int64 range");
    TORCH_CHECK(
        bin_result[1] >= total_bins,
        "cell-list bin count and status exceed the int64 range");
    const int64_t input_status = bin_result[1] - total_bins;
    TORCH_CHECK((input_status & ~int64_t{3}) == 0, "invalid cell-list input status");
    TORCH_CHECK(
        (input_status & 1) == 0,
        "positions must be finite and periodic representative wraps must fit int32");
    if ((input_status & 2) != 0) {
        TORCH_CHECK(
            total_blocks < (int64_t{1} << 31),
            "unwrapped representatives require the exhaustive CUDA path with "
            "fewer than 2^31 thread blocks");
        return find_neighbors_cuda_exhaustive(
            positions,
            offsets,
            cells,
            duals,
            image_shifts,
            image_offsets,
            block_offsets,
            total_blocks,
            cutoff,
            half_list,
            include_self);
    }
    if (total_bins > kMaximumDenseBins ||
        (total_nodes > 0 && total_bins > kMaximumBinsPerNode * total_nodes)) {
        TORCH_CHECK(
            total_blocks < (int64_t{1} << 31),
            "the cell-list bin layout exceeds its safety limits and the exhaustive "
            "fallback requires fewer than 2^31 thread blocks");
        return find_neighbors_cuda_exhaustive(
            positions,
            offsets,
            cells,
            duals,
            image_shifts,
            image_offsets,
            block_offsets,
            total_blocks,
            cutoff,
            half_list,
            include_self);
    }

    auto bin_heads = torch::full(
        {total_bins}, -1, positions.options().dtype(torch::kInt32));
    auto node_next = torch::full(
        {total_nodes}, -1, positions.options().dtype(torch::kInt32));
    const int64_t node_blocks = (total_nodes + kThreads - 1) / kThreads;
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "insert_neighbor_search_images", [&] {
        insert_images_kernel<scalar_t><<<node_blocks, kThreads, 0, stream>>>(
            offsets.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            wrapped_positions.data_ptr<scalar_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_offsets.data_ptr<int64_t>(),
            node_offsets.data_ptr<int64_t>(),
            bounds_minimum.data_ptr<scalar_t>(),
            bounds_maximum.data_ptr<scalar_t>(),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_offsets.data_ptr<int64_t>(),
            total_nodes,
            batch_size,
            static_cast<scalar_t>(cutoff),
            bin_heads.data_ptr<int32_t>(),
            node_next.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto source_pair_counts = torch::empty(
        {n_atoms + 1}, positions.options().dtype(torch::kInt64));
    // The final slot is also a status word. The query kernel writes INT64_MIN
    // on shift overflow, and the required pair-count cumsum returns it to the host.
    C10_CUDA_CHECK(cudaMemsetAsync(
        source_pair_counts.data_ptr<int64_t>() + n_atoms,
        0,
        sizeof(int64_t),
        stream));
    const int64_t query_blocks = (n_atoms + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto pair_offsets = torch::empty(
        {n_atoms + 2}, positions.options().dtype(torch::kInt64));
    C10_CUDA_CHECK(cudaMemsetAsync(
        pair_offsets.data_ptr<int64_t>(), 0, sizeof(int64_t), stream));
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "count_neighbor_search_cell_pairs", [&] {
        const QueryArguments<scalar_t> arguments{
            offsets.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            wrapped_positions.data_ptr<scalar_t>(),
            atom_wraps.data_ptr<int32_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_offsets.data_ptr<int64_t>(),
            node_offsets.data_ptr<int64_t>(),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_offsets.data_ptr<int64_t>(),
            bin_heads.data_ptr<int32_t>(),
            node_next.data_ptr<int32_t>(),
            n_atoms,
            batch_size,
            static_cast<scalar_t>(cutoff),
            static_cast<scalar_t>(cutoff * cutoff),
            nullptr,
            source_pair_counts.data_ptr<int64_t>(),
            source_pair_counts.data_ptr<int64_t>() + n_atoms,
            0,
            nullptr,
            nullptr};
        dispatch_query_bins<scalar_t, false>(
            mode, arguments, query_blocks, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pair_offsets.slice(0, 1, n_atoms + 2)
        .copy_(torch::cumsum(source_pair_counts, 0, torch::kInt64));
    const int64_t n_pairs = pair_offsets[n_atoms + 1].item<int64_t>();
    TORCH_CHECK(
        n_pairs >= 0,
        "a cell shift required by the neighbor list exceeds the int32 output range");
    pair_indices = torch::empty({2, n_pairs}, positions.options().dtype(torch::kInt64));
    cell_shifts = torch::empty({n_pairs, 3}, positions.options().dtype(torch::kInt32));
    if (n_pairs == 0) {
        return {pair_indices, cell_shifts};
    }
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "write_neighbor_search_cell_pairs", [&] {
        const QueryArguments<scalar_t> arguments{
            offsets.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            wrapped_positions.data_ptr<scalar_t>(),
            atom_wraps.data_ptr<int32_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_offsets.data_ptr<int64_t>(),
            node_offsets.data_ptr<int64_t>(),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_offsets.data_ptr<int64_t>(),
            bin_heads.data_ptr<int32_t>(),
            node_next.data_ptr<int32_t>(),
            n_atoms,
            batch_size,
            static_cast<scalar_t>(cutoff),
            static_cast<scalar_t>(cutoff * cutoff),
            pair_offsets.data_ptr<int64_t>(),
            nullptr,
            nullptr,
            n_pairs,
            pair_indices.data_ptr<int64_t>(),
            cell_shifts.data_ptr<int32_t>()};
        dispatch_query_bins<scalar_t, true>(
            mode, arguments, query_blocks, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pair_indices, cell_shifts};
}
