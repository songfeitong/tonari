#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>


namespace {

constexpr int kBlockSize = 256;


__device__ __forceinline__ int64_t structure_for_atom(
    int64_t atom,
    const int64_t* ptr,
    int64_t batch_size) {
    int64_t lower = 0;
    int64_t upper = batch_size;
    while (lower < upper) {
        const int64_t middle = lower + (upper - lower) / 2;
        if (ptr[middle + 1] <= atom) {
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
    const int64_t* ptr,
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
    const int64_t batch = structure_for_atom(atom, ptr, batch_size);
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


__device__ __forceinline__ int64_t structure_for_block(
    int64_t block_index,
    const int64_t* block_ptr,
    int64_t batch_size) {
    int64_t lower = 0;
    int64_t upper = batch_size;
    while (lower < upper) {
        const int64_t middle = lower + (upper - lower) / 2;
        if (block_ptr[middle + 1] <= block_index) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}


template <typename scalar_t>
__device__ __forceinline__ bool evaluate_candidate(
    int64_t task_index,
    int64_t batch_index,
    const scalar_t* positions,
    const int64_t* ptr,
    const scalar_t* cells,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_ptr,
    scalar_t cutoff_squared,
    int64_t& source,
    int64_t& target,
    int64_t (&cell_shift)[3],
    int64_t* shift_overflow) {
    const int64_t atom_start = ptr[batch_index];
    const int64_t n_atoms = ptr[batch_index + 1] - atom_start;
    const int64_t shift_start = image_ptr[batch_index];
    const int64_t n_shifts = image_ptr[batch_index + 1] - shift_start;
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
            static_cast<int64_t>(atom_wraps[3 * source + axis]) +
            static_cast<int64_t>(atom_wraps[3 * target + axis]);
    }

    if (source == target && cell_shift[0] == 0 && cell_shift[1] == 0 &&
        cell_shift[2] == 0) {
        return false;
    }

    scalar_t distance_squared = scalar_t(0);
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        scalar_t component = source_position[cartesian] - target_position[cartesian];
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


template <typename scalar_t>
__global__ void count_edges_kernel(
    const scalar_t* positions,
    const int64_t* ptr,
    const scalar_t* cells,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_ptr,
    const int64_t* block_ptr,
    int64_t batch_size,
    scalar_t cutoff_squared,
    int64_t* edge_count,
    int64_t* shift_overflow) {
    const int64_t batch_index = structure_for_block(blockIdx.x, block_ptr, batch_size);
    const int64_t local_block = blockIdx.x - block_ptr[batch_index];
    const int64_t task_index = local_block * blockDim.x + threadIdx.x;
    int64_t source = 0;
    int64_t target = 0;
    int64_t cell_shift[3] = {0, 0, 0};
    const int hit = evaluate_candidate(
        task_index,
        batch_index,
        positions,
        ptr,
        cells,
        atom_wraps,
        image_shifts,
        image_ptr,
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
            reinterpret_cast<unsigned long long*>(edge_count),
            static_cast<unsigned long long>(block_count));
    }
}


template <typename scalar_t>
__global__ void write_edges_kernel(
    const scalar_t* positions,
    const int64_t* ptr,
    const scalar_t* cells,
    const int32_t* atom_wraps,
    const int32_t* image_shifts,
    const int64_t* image_ptr,
    const int64_t* block_ptr,
    int64_t batch_size,
    scalar_t cutoff_squared,
    int64_t n_edges,
    int64_t* edge_cursor,
    int64_t* edge_index,
    int32_t* output_shifts) {
    const int64_t batch_index = structure_for_block(blockIdx.x, block_ptr, batch_size);
    const int64_t local_block = blockIdx.x - block_ptr[batch_index];
    const int64_t task_index = local_block * blockDim.x + threadIdx.x;
    int64_t source = 0;
    int64_t target = 0;
    int64_t cell_shift[3] = {0, 0, 0};
    const int hit = evaluate_candidate(
        task_index,
        batch_index,
        positions,
        ptr,
        cells,
        atom_wraps,
        image_shifts,
        image_ptr,
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
            reinterpret_cast<unsigned long long*>(edge_cursor),
            static_cast<unsigned long long>(block_count)));
    }
    __syncthreads();
    if (!hit) {
        return;
    }
    const int64_t output_index = block_output_start + local_output_index;
    edge_index[output_index] = source;
    edge_index[n_edges + output_index] = target;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        output_shifts[3 * output_index + axis] = static_cast<int32_t>(cell_shift[axis]);
    }
}

}  // namespace


std::vector<torch::Tensor> radius_graph_pbc_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    const torch::Tensor& block_ptr,
    int64_t total_blocks,
    double cutoff) {
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(ptr.is_cuda() && cells.is_cuda() && duals.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(image_shifts.is_cuda() && image_ptr.is_cuda() && block_ptr.is_cuda(), "all metadata must be CUDA tensors");
    TORCH_CHECK(positions.is_contiguous() && ptr.is_contiguous() && cells.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(duals.is_contiguous() && image_shifts.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(image_ptr.is_contiguous() && block_ptr.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(positions.scalar_type() == cells.scalar_type(), "positions and cells must have the same dtype");
    TORCH_CHECK(positions.scalar_type() == duals.scalar_type(), "positions and duals must have the same dtype");
    TORCH_CHECK(ptr.scalar_type() == torch::kInt64, "ptr must have dtype int64");
    TORCH_CHECK(image_shifts.scalar_type() == torch::kInt32, "image_shifts must have dtype int32");
    TORCH_CHECK(image_ptr.scalar_type() == torch::kInt64 && block_ptr.scalar_type() == torch::kInt64, "metadata pointers must have dtype int64");
    TORCH_CHECK(total_blocks >= 0 && total_blocks < (int64_t{1} << 31), "invalid number of CUDA blocks");

    const c10::cuda::CUDAGuard device_guard(positions.device());
    const auto stream = at::cuda::getCurrentCUDAStream(positions.get_device());
    const int64_t batch_size = ptr.numel() - 1;
    auto edge_index = torch::empty({2, 0}, positions.options().dtype(torch::kInt64));
    auto output_shifts = torch::empty({0, 3}, positions.options().dtype(torch::kInt32));
    if (total_blocks == 0) {
        return {edge_index, output_shifts};
    }

    auto atom_wraps = torch::empty(
        {positions.size(0), 3}, positions.options().dtype(torch::kInt32));
    auto count_result = torch::zeros({4}, positions.options().dtype(torch::kInt64));
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "count_radius_graph_edges", [&] {
        const scalar_t cutoff_squared = static_cast<scalar_t>(cutoff * cutoff);
        const int64_t position_blocks =
            (positions.size(0) + kBlockSize - 1) / kBlockSize;
        prepare_atom_wraps_kernel<scalar_t><<<position_blocks, kBlockSize, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            ptr.data_ptr<int64_t>(),
            duals.data_ptr<scalar_t>(),
            positions.size(0),
            batch_size,
            atom_wraps.data_ptr<int32_t>(),
            count_result.data_ptr<int64_t>() + 2,
            count_result.data_ptr<int64_t>() + 3);
        count_edges_kernel<scalar_t><<<total_blocks, kBlockSize, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            ptr.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            atom_wraps.data_ptr<int32_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_ptr.data_ptr<int64_t>(),
            block_ptr.data_ptr<int64_t>(),
            batch_size,
            cutoff_squared,
            count_result.data_ptr<int64_t>(),
            count_result.data_ptr<int64_t>() + 1);
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
        "a cell shift required by the cutoff graph exceeds the int32 output range");
    const int64_t n_edges = count_result_data[0];
    edge_index = torch::empty({2, n_edges}, positions.options().dtype(torch::kInt64));
    output_shifts = torch::empty({n_edges, 3}, positions.options().dtype(torch::kInt32));
    if (n_edges == 0) {
        return {edge_index, output_shifts};
    }

    auto edge_cursor = torch::zeros({1}, positions.options().dtype(torch::kInt64));
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "write_radius_graph_edges", [&] {
        const scalar_t cutoff_squared = static_cast<scalar_t>(cutoff * cutoff);
        write_edges_kernel<scalar_t><<<total_blocks, kBlockSize, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            ptr.data_ptr<int64_t>(),
            cells.data_ptr<scalar_t>(),
            atom_wraps.data_ptr<int32_t>(),
            image_shifts.data_ptr<int32_t>(),
            image_ptr.data_ptr<int64_t>(),
            block_ptr.data_ptr<int64_t>(),
            batch_size,
            cutoff_squared,
            n_edges,
            edge_cursor.data_ptr<int64_t>(),
            edge_index.data_ptr<int64_t>(),
            output_shifts.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {edge_index, output_shifts};
}
