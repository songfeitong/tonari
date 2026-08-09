#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>


std::vector<torch::Tensor> radius_graph_pbc_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    const torch::Tensor& block_ptr,
    int64_t total_blocks,
    double cutoff);


namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int64_t kMaximumDenseBins = int64_t{1} << 28;
constexpr int64_t kMaximumBinsPerNode = 64;


__device__ __forceinline__ int64_t segment_for_index(
    int64_t index,
    const int64_t* segment_ptr,
    int64_t n_segments) {
    int64_t lower = 0;
    int64_t upper = n_segments;
    while (lower < upper) {
        const int64_t middle = lower + (upper - lower) / 2;
        if (segment_ptr[middle + 1] <= index) {
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
    const int64_t* ptr,
    const scalar_t* cells,
    const scalar_t* duals,
    int64_t n_atoms,
    int64_t batch_size,
    scalar_t* wrapped_positions,
    int32_t* atom_wraps,
    scalar_t* bounds_minimum,
    scalar_t* bounds_maximum,
    int64_t* invalid_input) {
    const int64_t atom = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) {
        return;
    }
    const int64_t batch = segment_for_index(atom, ptr, batch_size);
    const scalar_t* position = positions + 3 * atom;
    const scalar_t* cell = cells + 9 * batch;
    const scalar_t* dual = duals + 9 * batch;
    bool finite = true;
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        finite &= isfinite(position[cartesian]);
    }
    if (!finite) {
        atomicExch(
            reinterpret_cast<unsigned long long*>(invalid_input),
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
            atomicExch(
                reinterpret_cast<unsigned long long*>(invalid_input),
                static_cast<unsigned long long>(1));
            wraps[axis] = 0;
        } else {
            wraps[axis] = static_cast<int32_t>(floor(fractional));
        }
        atom_wraps[3 * atom + axis] = wraps[axis];
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
    const int64_t* ptr,
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
        bin_counts[batch] = bin_counts[batch] == 0
            ? 0
            : std::numeric_limits<int64_t>::min();
        return;
    }
    int64_t count = 1;
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        int64_t dimension = 1;
        scalar_t origin = scalar_t(0);
        if (ptr[batch + 1] > ptr[batch]) {
            const scalar_t minimum = bounds_minimum[3 * batch + cartesian];
            const scalar_t maximum = bounds_maximum[3 * batch + cartesian];
            origin = minimum - cutoff;
            dimension = max(
                int64_t{1},
                static_cast<int64_t>(ceil((maximum - minimum + 2 * cutoff) / cutoff)));
        }
        bin_origins[3 * batch + cartesian] = origin;
        bin_dimensions[3 * batch + cartesian] = dimension;
        if (count <= std::numeric_limits<int64_t>::max() / dimension) {
            count *= dimension;
        } else {
            count = std::numeric_limits<int64_t>::max();
        }
    }
    bin_counts[batch] = count;
}


template <typename scalar_t>
__global__ void insert_images_kernel(
    const int64_t* ptr,
    const scalar_t* cells,
    const scalar_t* wrapped_positions,
    const int32_t* image_shifts,
    const int64_t* image_ptr,
    const int64_t* node_ptr,
    const scalar_t* bounds_minimum,
    const scalar_t* bounds_maximum,
    const scalar_t* bin_origins,
    const int64_t* bin_dimensions,
    const int64_t* bin_ptr,
    int64_t total_nodes,
    int64_t batch_size,
    scalar_t cutoff,
    int32_t* bin_heads,
    int32_t* node_next) {
    const int64_t node = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= total_nodes) {
        return;
    }
    const int64_t batch = segment_for_index(node, node_ptr, batch_size);
    const int64_t local_node = node - node_ptr[batch];
    const int64_t n_shifts = image_ptr[batch + 1] - image_ptr[batch];
    const int64_t local_source = local_node / n_shifts;
    const int64_t shift_index = local_node % n_shifts;
    const int64_t source = ptr[batch] + local_source;
    const scalar_t* cell = cells + 9 * batch;
    const int32_t* shift = image_shifts + 3 * (image_ptr[batch] + shift_index);
    int64_t coordinates[3];
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        scalar_t image_position = wrapped_positions[3 * source + cartesian];
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
    const int64_t bin = bin_ptr[batch] + local_bin;
    node_next[node] = atomicExch(bin_heads + bin, static_cast<int32_t>(node));
}


template <typename scalar_t, bool write_edges>
__global__ void query_bins_kernel(
    const int64_t* ptr,
    const scalar_t* cells,
    const scalar_t* wrapped_positions,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_ptr,
    const int64_t* node_ptr,
    const scalar_t* bin_origins,
    const int64_t* bin_dimensions,
    const int64_t* bin_ptr,
    const int32_t* bin_heads,
    const int32_t* node_next,
    int64_t n_atoms,
    int64_t batch_size,
    scalar_t cutoff,
    const int64_t* edge_ptr,
    int64_t* target_counts,
    int64_t* shift_overflow,
    int64_t n_edges,
    int64_t* edge_index,
    int32_t* output_shifts) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int64_t target =
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) + warp;
    if (target >= n_atoms) {
        return;
    }
    const int64_t batch = segment_for_index(target, ptr, batch_size);
    const scalar_t* cell = cells + 9 * batch;
    int64_t target_bin[3];
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        const int64_t dimension = bin_dimensions[3 * batch + cartesian];
        target_bin[cartesian] = min(
            dimension - 1,
            max(
                int64_t{0},
                static_cast<int64_t>(floor(
                    (wrapped_positions[3 * target + cartesian] -
                     bin_origins[3 * batch + cartesian]) /
                    cutoff))));
    }

    __shared__ int warp_output_cursors[kWarpsPerBlock];
    if constexpr (write_edges) {
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
        const int64_t coordinate_x = target_bin[0] + offset_x;
        const int64_t coordinate_y = target_bin[1] + offset_y;
        const int64_t coordinate_z = target_bin[2] + offset_z;
        if (coordinate_x >= 0 && coordinate_x < bin_dimensions[3 * batch] &&
            coordinate_y >= 0 && coordinate_y < bin_dimensions[3 * batch + 1] &&
            coordinate_z >= 0 && coordinate_z < bin_dimensions[3 * batch + 2]) {
            const int64_t local_bin =
                (coordinate_x * bin_dimensions[3 * batch + 1] + coordinate_y) *
                    bin_dimensions[3 * batch + 2] +
                coordinate_z;
            int32_t node = bin_heads[bin_ptr[batch] + local_bin];
            while (node >= 0) {
                const int64_t local_node = node - node_ptr[batch];
                const int64_t n_shifts = image_ptr[batch + 1] - image_ptr[batch];
                const int64_t local_source = local_node / n_shifts;
                const int64_t shift_index = local_node % n_shifts;
                const int64_t source = ptr[batch] + local_source;
                const int32_t* wrapped_shift =
                    image_shifts + 3 * (image_ptr[batch] + shift_index);
                int64_t output_shift[3];
                scalar_t distance_squared = scalar_t(0);
#pragma unroll
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    scalar_t component =
                        wrapped_positions[3 * source + cartesian] -
                        wrapped_positions[3 * target + cartesian];
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
                        static_cast<int64_t>(atom_wraps[3 * source + axis]) +
                        static_cast<int64_t>(atom_wraps[3 * target + axis]);
                }
                const bool onsite = source == target && output_shift[0] == 0 &&
                    output_shift[1] == 0 && output_shift[2] == 0;
                if (!onsite && distance_squared < cutoff * cutoff) {
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
                    if constexpr (write_edges) {
                        const int local_output = atomicAdd(warp_output_cursors + warp, 1);
                        const int64_t output = edge_ptr[target] + local_output;
                        edge_index[output] = source;
                        edge_index[n_edges + output] = target;
#pragma unroll
                        for (int axis = 0; axis < 3; ++axis) {
                            output_shifts[3 * output + axis] =
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

    if constexpr (!write_edges) {
#pragma unroll
        for (int delta = kWarpSize / 2; delta > 0; delta /= 2) {
            lane_count += __shfl_down_sync(0xffffffff, lane_count, delta);
        }
        if (lane == 0) {
            target_counts[target] = lane_count;
        }
    }
}


void validate_cell_inputs(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    const torch::Tensor& block_ptr,
    const torch::Tensor& node_ptr,
    int64_t total_nodes) {
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(ptr.is_cuda() && cells.is_cuda() && duals.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(image_shifts.is_cuda() && image_ptr.is_cuda(), "all metadata must be CUDA tensors");
    TORCH_CHECK(block_ptr.is_cuda() && node_ptr.is_cuda(), "all metadata must be CUDA tensors");
    TORCH_CHECK(positions.is_contiguous() && ptr.is_contiguous() && cells.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(duals.is_contiguous() && image_shifts.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(image_ptr.is_contiguous() && block_ptr.is_contiguous() && node_ptr.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(positions.scalar_type() == cells.scalar_type(), "positions and cells must have the same dtype");
    TORCH_CHECK(positions.scalar_type() == duals.scalar_type(), "positions and duals must have the same dtype");
    TORCH_CHECK(total_nodes >= 0 && total_nodes < (int64_t{1} << 31), "cell-list node count exceeds int32 indexing");
}

}  // namespace


std::vector<torch::Tensor> radius_graph_pbc_cell_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    const torch::Tensor& block_ptr,
    const torch::Tensor& node_ptr,
    int64_t total_blocks,
    int64_t total_nodes,
    double cutoff) {
    validate_cell_inputs(
        positions,
        ptr,
        cells,
        duals,
        image_shifts,
        image_ptr,
        block_ptr,
        node_ptr,
        total_nodes);
    const c10::cuda::CUDAGuard device_guard(positions.device());
    const auto stream = at::cuda::getCurrentCUDAStream(positions.get_device());
    const int64_t n_atoms = positions.size(0);
    const int64_t batch_size = ptr.numel() - 1;
    auto edge_index = torch::empty({2, 0}, positions.options().dtype(torch::kInt64));
    auto output_shifts = torch::empty({0, 3}, positions.options().dtype(torch::kInt32));
    if (n_atoms == 0) {
        return {edge_index, output_shifts};
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
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "wrap_radius_graph_positions", [&] {
        wrap_positions_kernel<scalar_t><<<atom_blocks, kThreads, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            ptr.data_ptr<int64_t>(),
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
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "define_radius_graph_bins", [&] {
        define_bins_kernel<scalar_t><<<batch_blocks, kThreads, 0, stream>>>(
            ptr.data_ptr<int64_t>(),
            bounds_minimum.data_ptr<scalar_t>(),
            bounds_maximum.data_ptr<scalar_t>(),
            batch_size,
            static_cast<scalar_t>(cutoff),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_counts.data_ptr<int64_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto bin_ptr = torch::empty(
        {batch_size + 2}, positions.options().dtype(torch::kInt64));
    C10_CUDA_CHECK(cudaMemsetAsync(
        bin_ptr.data_ptr<int64_t>(), 0, sizeof(int64_t), stream));
    bin_ptr.slice(0, 1, batch_size + 2)
        .copy_(torch::cumsum(bin_counts, 0, torch::kInt64));
    const int64_t total_bins = bin_ptr[batch_size + 1].item<int64_t>();
    TORCH_CHECK(
        total_bins >= 0,
        "positions must be finite and periodic representative wraps must fit int32");
    if (total_bins > kMaximumDenseBins ||
        (total_nodes > 0 && total_bins > kMaximumBinsPerNode * total_nodes)) {
        TORCH_CHECK(
            total_blocks < (int64_t{1} << 31),
            "the cell-list bin layout exceeds its safety limits and the exhaustive "
            "fallback requires fewer than 2^31 thread blocks");
        return radius_graph_pbc_cuda(
            positions,
            ptr,
            cells,
            duals,
            image_shifts,
            image_ptr,
            block_ptr,
            total_blocks,
            cutoff);
    }

    auto bin_heads = torch::full(
        {total_bins}, -1, positions.options().dtype(torch::kInt32));
    auto node_next = torch::full(
        {total_nodes}, -1, positions.options().dtype(torch::kInt32));
    const int64_t node_blocks = (total_nodes + kThreads - 1) / kThreads;
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "insert_radius_graph_images", [&] {
        insert_images_kernel<scalar_t><<<node_blocks, kThreads, 0, stream>>>(
            ptr.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            wrapped_positions.data_ptr<scalar_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_ptr.data_ptr<int64_t>(),
            node_ptr.data_ptr<int64_t>(),
            bounds_minimum.data_ptr<scalar_t>(),
            bounds_maximum.data_ptr<scalar_t>(),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_ptr.data_ptr<int64_t>(),
            total_nodes,
            batch_size,
            static_cast<scalar_t>(cutoff),
            bin_heads.data_ptr<int32_t>(),
            node_next.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto target_counts = torch::empty(
        {n_atoms + 1}, positions.options().dtype(torch::kInt64));
    C10_CUDA_CHECK(cudaMemsetAsync(
        target_counts.data_ptr<int64_t>() + n_atoms,
        0,
        sizeof(int64_t),
        stream));
    const int64_t query_blocks = (n_atoms + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto edge_ptr = torch::empty(
        {n_atoms + 2}, positions.options().dtype(torch::kInt64));
    C10_CUDA_CHECK(cudaMemsetAsync(
        edge_ptr.data_ptr<int64_t>(), 0, sizeof(int64_t), stream));
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "count_radius_graph_cell_edges", [&] {
        query_bins_kernel<scalar_t, false><<<query_blocks, kThreads, 0, stream>>>(
            ptr.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            wrapped_positions.data_ptr<scalar_t>(),
            atom_wraps.data_ptr<int32_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_ptr.data_ptr<int64_t>(),
            node_ptr.data_ptr<int64_t>(),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_ptr.data_ptr<int64_t>(),
            bin_heads.data_ptr<int32_t>(),
            node_next.data_ptr<int32_t>(),
            n_atoms,
            batch_size,
            static_cast<scalar_t>(cutoff),
            nullptr,
            target_counts.data_ptr<int64_t>(),
            target_counts.data_ptr<int64_t>() + n_atoms,
            0,
            nullptr,
            nullptr);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    edge_ptr.slice(0, 1, n_atoms + 2)
        .copy_(torch::cumsum(target_counts, 0, torch::kInt64));
    const int64_t n_edges = edge_ptr[n_atoms + 1].item<int64_t>();
    TORCH_CHECK(
        n_edges >= 0,
        "a cell shift required by the cutoff graph exceeds the int32 output range");
    edge_index = torch::empty({2, n_edges}, positions.options().dtype(torch::kInt64));
    output_shifts = torch::empty({n_edges, 3}, positions.options().dtype(torch::kInt32));
    if (n_edges == 0) {
        return {edge_index, output_shifts};
    }
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "write_radius_graph_cell_edges", [&] {
        query_bins_kernel<scalar_t, true><<<query_blocks, kThreads, 0, stream>>>(
            ptr.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            wrapped_positions.data_ptr<scalar_t>(),
            atom_wraps.data_ptr<int32_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_ptr.data_ptr<int64_t>(),
            node_ptr.data_ptr<int64_t>(),
            bin_origins.data_ptr<scalar_t>(),
            bin_dimensions.data_ptr<int64_t>(),
            bin_ptr.data_ptr<int64_t>(),
            bin_heads.data_ptr<int32_t>(),
            node_next.data_ptr<int32_t>(),
            n_atoms,
            batch_size,
            static_cast<scalar_t>(cutoff),
            edge_ptr.data_ptr<int64_t>(),
            nullptr,
            nullptr,
            n_edges,
            edge_index.data_ptr<int64_t>(),
            output_shifts.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {edge_index, output_shifts};
}
