#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "neighbors_cuda.h"
#include "../core/pair_policy.h"

#include <cstdint>
#include <limits>


namespace {

constexpr int kBlockSize = 256;


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
__global__ void prepare_atom_wraps_kernel(
    const scalar_t* positions,
    const int64_t* batch_ptr,
    const scalar_t* duals,
    int64_t n_atoms,
    int64_t batch_size,
    int32_t* atom_wraps,
    int64_t* nonfinite_input,
    int64_t* wrap_overflow) {
    const int64_t atom = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) {
        return;
    }
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        if (!isfinite(positions[3 * atom + cartesian])) {
            atomicExch(
                reinterpret_cast<unsigned long long*>(nonfinite_input),
                static_cast<unsigned long long>(1));
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                atom_wraps[3 * atom + axis] = 0;
            }
            return;
        }
    }
    const int64_t batch = segment_for_index(atom, batch_ptr, batch_size);
    const scalar_t* dual = duals + 9 * batch;
    const scalar_t upper_bound = static_cast<scalar_t>(2147483648.0);
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        scalar_t fractional = scalar_t(0);
#pragma unroll
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            fractional +=
                positions[3 * atom + cartesian] * dual[3 * cartesian + axis];
        }
        if (!isfinite(fractional) || fractional < -upper_bound ||
            fractional >= upper_bound) {
            atomicExch(
                reinterpret_cast<unsigned long long*>(wrap_overflow),
                static_cast<unsigned long long>(1));
            atom_wraps[3 * atom + axis] = 0;
        } else {
            atom_wraps[3 * atom + axis] =
                static_cast<int32_t>(floor(fractional));
        }
    }
}


template <typename scalar_t, neighbor_search::PairMode Mode>
__device__ __forceinline__ bool evaluate_candidate(
    int64_t task_index,
    int64_t batch_index,
    const scalar_t* positions,
    const int64_t* batch_ptr,
    const scalar_t* cells,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_offsets,
    scalar_t cutoff_squared,
    int64_t& source,
    int64_t& target,
    int64_t (&cell_shift)[3],
    int64_t* shift_overflow) {
    const int64_t atom_start = batch_ptr[batch_index];
    const int64_t n_atoms = batch_ptr[batch_index + 1] - atom_start;
    const int64_t shift_start = image_offsets[batch_index];
    const int64_t n_shifts = image_offsets[batch_index + 1] - shift_start;
    const int64_t n_tasks = n_atoms * n_atoms * n_shifts;
    if (task_index >= n_tasks) {
        return false;
    }

    const int64_t shift_index = task_index % n_shifts;
    const int64_t pair_index = task_index / n_shifts;
    const int64_t local_source = pair_index % n_atoms;
    const int64_t local_target = pair_index / n_atoms;
    source = atom_start + local_source;
    target = atom_start + local_target;

    const scalar_t* source_position = positions + 3 * source;
    const scalar_t* target_position = positions + 3 * target;
    const scalar_t* structure_cell = cells + 9 * batch_index;
    const int32_t* wrapped_shift = image_shifts + 3 * (shift_start + shift_index);

#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        cell_shift[axis] = static_cast<int64_t>(wrapped_shift[axis]) -
            static_cast<int64_t>(atom_wraps[3 * target + axis]) +
            static_cast<int64_t>(atom_wraps[3 * source + axis]);
    }

    const bool zero_shift_self =
        neighbor_search::is_zero_shift_self_pair(source, target, cell_shift);
    if (!neighbor_search::keep_pair_identity<Mode>(source, target, cell_shift)) {
        return false;
    }

    scalar_t distance_squared = scalar_t(0);
    if (!zero_shift_self) {
#pragma unroll
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            scalar_t component =
                target_position[cartesian] - source_position[cartesian];
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                component += static_cast<scalar_t>(cell_shift[axis]) *
                    structure_cell[3 * axis + cartesian];
            }
            distance_squared += component * component;
        }
        if (distance_squared >= cutoff_squared) {
            return false;
        }
    }
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        if (cell_shift[axis] < std::numeric_limits<int32_t>::min() ||
            cell_shift[axis] > std::numeric_limits<int32_t>::max()) {
            if (shift_overflow != nullptr) {
                atomicExch(
                    reinterpret_cast<unsigned long long*>(shift_overflow),
                    static_cast<unsigned long long>(1));
            }
            return false;
        }
    }
    return true;
}


template <typename scalar_t, neighbor_search::PairMode Mode>
__global__ void count_pairs_kernel(
    const scalar_t* positions,
    const int64_t* batch_ptr,
    const scalar_t* cells,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_offsets,
    const int64_t* block_offsets,
    int64_t batch_size,
    scalar_t cutoff_squared,
    int64_t* pair_count,
    int64_t* shift_overflow) {
    const int64_t batch_index =
        segment_for_index(blockIdx.x, block_offsets, batch_size);
    const int64_t local_block = blockIdx.x - block_offsets[batch_index];
    const int64_t task_index = local_block * blockDim.x + threadIdx.x;
    int64_t source = 0;
    int64_t target = 0;
    int64_t cell_shift[3] = {0, 0, 0};
    const int hit = evaluate_candidate<scalar_t, Mode>(
        task_index,
        batch_index,
        positions,
        batch_ptr,
        cells,
        atom_wraps,
        image_shifts,
        image_offsets,
        cutoff_squared,
        source,
        target,
        cell_shift,
        shift_overflow)
        ? 1
        : 0;

    using BlockReduce = cub::BlockReduce<int, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;
    const int block_count = BlockReduce(reduction_storage).Sum(hit);
    if (threadIdx.x == 0 && block_count > 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(pair_count),
            static_cast<unsigned long long>(block_count));
    }
}


template <typename scalar_t, neighbor_search::PairMode Mode>
__global__ void write_pairs_kernel(
    const scalar_t* positions,
    const int64_t* batch_ptr,
    const scalar_t* cells,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_offsets,
    const int64_t* block_offsets,
    int64_t batch_size,
    scalar_t cutoff_squared,
    int64_t n_pairs,
    int64_t* pair_cursor,
    int64_t* pair_indices,
    int32_t* cell_shifts) {
    const int64_t batch_index =
        segment_for_index(blockIdx.x, block_offsets, batch_size);
    const int64_t local_block = blockIdx.x - block_offsets[batch_index];
    const int64_t task_index = local_block * blockDim.x + threadIdx.x;
    int64_t source = 0;
    int64_t target = 0;
    int64_t cell_shift[3] = {0, 0, 0};
    const int hit = evaluate_candidate<scalar_t, Mode>(
        task_index,
        batch_index,
        positions,
        batch_ptr,
        cells,
        atom_wraps,
        image_shifts,
        image_offsets,
        cutoff_squared,
        source,
        target,
        cell_shift,
        nullptr)
        ? 1
        : 0;

    using BlockScan = cub::BlockScan<int, kBlockSize>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int64_t block_output_start;
    int local_output_index = 0;
    int block_count = 0;
    BlockScan(scan_storage).ExclusiveSum(hit, local_output_index, block_count);
    if (threadIdx.x == 0 && block_count > 0) {
        block_output_start = static_cast<int64_t>(atomicAdd(
            reinterpret_cast<unsigned long long*>(pair_cursor),
            static_cast<unsigned long long>(block_count)));
    }
    __syncthreads();
    if (!hit) {
        return;
    }
    const int64_t output_index = block_output_start + local_output_index;
    pair_indices[output_index] = source;
    pair_indices[n_pairs + output_index] = target;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        cell_shifts[3 * output_index + axis] = static_cast<int32_t>(cell_shift[axis]);
    }
}


template <typename scalar_t, neighbor_search::PairMode Mode>
void launch_count_pairs(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cells,
    const torch::Tensor& atom_wraps,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    int64_t batch_size,
    scalar_t cutoff_squared,
    torch::Tensor& count_result,
    int64_t total_blocks,
    cudaStream_t stream) {
    count_pairs_kernel<scalar_t, Mode><<<total_blocks, kBlockSize, 0, stream>>>(
        positions.data_ptr<scalar_t>(),
        batch_ptr.data_ptr<int64_t>(),
        cells.data_ptr<scalar_t>(),
        atom_wraps.data_ptr<int32_t>(),
        image_shifts.data_ptr<int32_t>(),
        image_offsets.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        batch_size,
        cutoff_squared,
        count_result.data_ptr<int64_t>(),
        count_result.data_ptr<int64_t>() + 1);
}


template <typename scalar_t>
void dispatch_count_pairs(
    neighbor_search::PairMode mode,
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cells,
    const torch::Tensor& atom_wraps,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    int64_t batch_size,
    scalar_t cutoff_squared,
    torch::Tensor& count_result,
    int64_t total_blocks,
    cudaStream_t stream) {
    auto launch = [&]<neighbor_search::PairMode Mode>() {
        launch_count_pairs<scalar_t, Mode>(
            positions,
            batch_ptr,
            cells,
            atom_wraps,
            image_shifts,
            image_offsets,
            block_offsets,
            batch_size,
            cutoff_squared,
            count_result,
            total_blocks,
            stream);
    };
    switch (mode) {
        case neighbor_search::PairMode::Full:
            launch.template operator()<neighbor_search::PairMode::Full>();
            break;
        case neighbor_search::PairMode::FullWithSelf:
            launch.template operator()<neighbor_search::PairMode::FullWithSelf>();
            break;
        case neighbor_search::PairMode::Half:
            launch.template operator()<neighbor_search::PairMode::Half>();
            break;
        case neighbor_search::PairMode::HalfWithSelf:
            launch.template operator()<neighbor_search::PairMode::HalfWithSelf>();
            break;
    }
}


template <typename scalar_t, neighbor_search::PairMode Mode>
void launch_write_pairs(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cells,
    const torch::Tensor& atom_wraps,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    int64_t batch_size,
    scalar_t cutoff_squared,
    int64_t n_pairs,
    torch::Tensor& pair_cursor,
    torch::Tensor& pair_indices,
    torch::Tensor& cell_shifts,
    int64_t total_blocks,
    cudaStream_t stream) {
    write_pairs_kernel<scalar_t, Mode><<<total_blocks, kBlockSize, 0, stream>>>(
        positions.data_ptr<scalar_t>(),
        batch_ptr.data_ptr<int64_t>(),
        cells.data_ptr<scalar_t>(),
        atom_wraps.data_ptr<int32_t>(),
        image_shifts.data_ptr<int32_t>(),
        image_offsets.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        batch_size,
        cutoff_squared,
        n_pairs,
        pair_cursor.data_ptr<int64_t>(),
        pair_indices.data_ptr<int64_t>(),
        cell_shifts.data_ptr<int32_t>());
}


template <typename scalar_t>
void dispatch_write_pairs(
    neighbor_search::PairMode mode,
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cells,
    const torch::Tensor& atom_wraps,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    int64_t batch_size,
    scalar_t cutoff_squared,
    int64_t n_pairs,
    torch::Tensor& pair_cursor,
    torch::Tensor& pair_indices,
    torch::Tensor& cell_shifts,
    int64_t total_blocks,
    cudaStream_t stream) {
    auto launch = [&]<neighbor_search::PairMode Mode>() {
        launch_write_pairs<scalar_t, Mode>(
            positions,
            batch_ptr,
            cells,
            atom_wraps,
            image_shifts,
            image_offsets,
            block_offsets,
            batch_size,
            cutoff_squared,
            n_pairs,
            pair_cursor,
            pair_indices,
            cell_shifts,
            total_blocks,
            stream);
    };
    switch (mode) {
        case neighbor_search::PairMode::Full:
            launch.template operator()<neighbor_search::PairMode::Full>();
            break;
        case neighbor_search::PairMode::FullWithSelf:
            launch.template operator()<neighbor_search::PairMode::FullWithSelf>();
            break;
        case neighbor_search::PairMode::Half:
            launch.template operator()<neighbor_search::PairMode::Half>();
            break;
        case neighbor_search::PairMode::HalfWithSelf:
            launch.template operator()<neighbor_search::PairMode::HalfWithSelf>();
            break;
    }
}

}  // namespace


std::vector<torch::Tensor> find_neighbors_cuda_exhaustive(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    int64_t total_blocks,
    double cutoff,
    bool half_list,
    bool include_self) {
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(batch_ptr.is_cuda() && cells.is_cuda() && duals.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(image_shifts.is_cuda() && image_offsets.is_cuda() && block_offsets.is_cuda(), "all metadata must be CUDA tensors");
    TORCH_CHECK(positions.is_contiguous() && batch_ptr.is_contiguous() && cells.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(duals.is_contiguous() && image_shifts.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(image_offsets.is_contiguous() && block_offsets.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(positions.scalar_type() == cells.scalar_type(), "positions and cells must have the same dtype");
    TORCH_CHECK(positions.scalar_type() == duals.scalar_type(), "positions and duals must have the same dtype");
    TORCH_CHECK(batch_ptr.scalar_type() == torch::kInt64, "batch_ptr must have dtype int64");
    TORCH_CHECK(image_shifts.scalar_type() == torch::kInt32, "image_shifts must have dtype int32");
    TORCH_CHECK(image_offsets.scalar_type() == torch::kInt64 && block_offsets.scalar_type() == torch::kInt64, "metadata pointers must have dtype int64");
    TORCH_CHECK(total_blocks >= 0 && total_blocks < (int64_t{1} << 31), "invalid number of CUDA blocks");

    const c10::cuda::CUDAGuard device_guard(positions.device());
    const auto stream = at::cuda::getCurrentCUDAStream(positions.get_device());
    const int64_t batch_size = batch_ptr.numel() - 1;
    const auto mode = neighbor_search::pair_mode(half_list, include_self);
    auto pair_indices = torch::empty({2, 0}, positions.options().dtype(torch::kInt64));
    auto cell_shifts = torch::empty({0, 3}, positions.options().dtype(torch::kInt32));
    if (total_blocks == 0) {
        return {pair_indices, cell_shifts};
    }

    auto atom_wraps = torch::empty(
        {positions.size(0), 3}, positions.options().dtype(torch::kInt32));
    auto count_result = torch::zeros({4}, positions.options().dtype(torch::kInt64));
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "count_neighbor_search_pairs", [&] {
        const scalar_t cutoff_squared = static_cast<scalar_t>(cutoff * cutoff);
        const int64_t position_blocks =
            (positions.size(0) + kBlockSize - 1) / kBlockSize;
        prepare_atom_wraps_kernel<scalar_t><<<position_blocks, kBlockSize, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            batch_ptr.data_ptr<int64_t>(),
            duals.data_ptr<scalar_t>(),
            positions.size(0),
            batch_size,
            atom_wraps.data_ptr<int32_t>(),
            count_result.data_ptr<int64_t>() + 2,
            count_result.data_ptr<int64_t>() + 3);
        dispatch_count_pairs<scalar_t>(
            mode,
            positions,
            batch_ptr,
            cells,
            atom_wraps,
            image_shifts,
            image_offsets,
            block_offsets,
            batch_size,
            cutoff_squared,
            count_result,
            total_blocks,
            stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const auto count_result_cpu = count_result.cpu();
    const int64_t* count_result_data = count_result_cpu.data_ptr<int64_t>();
    TORCH_CHECK(
        count_result_data[2] == 0,
        "positions must contain only finite values");
    TORCH_CHECK(
        count_result_data[3] == 0,
        "atom representatives require periodic wraps outside the int32 range");
    TORCH_CHECK(
        count_result_data[1] == 0,
        "a cell shift required by the neighbor list exceeds the int32 output range");
    const int64_t n_pairs = count_result_data[0];
    pair_indices = torch::empty({2, n_pairs}, positions.options().dtype(torch::kInt64));
    cell_shifts = torch::empty({n_pairs, 3}, positions.options().dtype(torch::kInt32));
    if (n_pairs == 0) {
        return {pair_indices, cell_shifts};
    }

    auto pair_cursor = torch::zeros({1}, positions.options().dtype(torch::kInt64));
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "write_neighbor_search_pairs", [&] {
        const scalar_t cutoff_squared = static_cast<scalar_t>(cutoff * cutoff);
        dispatch_write_pairs<scalar_t>(
            mode,
            positions,
            batch_ptr,
            cells,
            atom_wraps,
            image_shifts,
            image_offsets,
            block_offsets,
            batch_size,
            cutoff_squared,
            n_pairs,
            pair_cursor,
            pair_indices,
            cell_shifts,
            total_blocks,
            stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pair_indices, cell_shifts};
}
