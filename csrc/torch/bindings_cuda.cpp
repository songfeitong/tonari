#include "../core/errors.h"
#include "../core/geometry.h"
#include "neighbors_cuda.h"

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>


namespace {

constexpr int64_t kBlockSize = 256;
constexpr int64_t kCellListMinimumAtoms = 256;
constexpr int64_t kInt32IndexLimit = int64_t{1} << 31;


template <typename scalar_t>
torch::Tensor tensor_from_vector(
    const std::vector<scalar_t>& values,
    torch::IntArrayRef sizes,
    torch::ScalarType dtype,
    const torch::Device& device) {
    auto host = torch::empty(
        sizes,
        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    if (!values.empty()) {
        std::memcpy(
            host.data_ptr<scalar_t>(),
            values.data(),
            values.size() * sizeof(scalar_t));
    }
    return host.to(device, dtype, false, true);
}


std::vector<torch::Tensor> neighbor_list(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cell,
    const torch::Tensor& pbc,
    double cutoff,
    bool half_list,
    bool include_self) {
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(
        batch_ptr.is_cuda() && cell.is_cuda() && pbc.is_cuda(),
        "all inputs must be CUDA tensors");
    TORCH_CHECK(
        positions.is_contiguous() && batch_ptr.is_contiguous() &&
            cell.is_contiguous() && pbc.is_contiguous(),
        "all inputs must be contiguous");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3);
    TORCH_CHECK(batch_ptr.dim() == 1 && batch_ptr.numel() > 0);
    TORCH_CHECK(cell.sizes() == torch::IntArrayRef({batch_ptr.numel() - 1, 3, 3}));
    TORCH_CHECK(pbc.sizes() == torch::IntArrayRef({batch_ptr.numel() - 1, 3}));
    TORCH_CHECK(positions.scalar_type() == cell.scalar_type());
    TORCH_CHECK(
        positions.scalar_type() == torch::kFloat32 ||
            positions.scalar_type() == torch::kFloat64);
    TORCH_CHECK(batch_ptr.scalar_type() == torch::kInt64);
    TORCH_CHECK(pbc.scalar_type() == torch::kBool);
    neighbor_search::require_input(
        std::isfinite(cutoff) && cutoff > 0,
        "cutoff must be finite and positive");
    neighbor_search::require_input(
        positions.size(0) < kInt32IndexLimit,
        "the current implementation supports fewer than 2^31 atoms");

    const auto batch_ptr_cpu = batch_ptr.to(torch::kCPU).contiguous();
    const auto cell_cpu = cell.to(torch::kCPU, torch::kFloat64).contiguous();
    const auto pbc_cpu = pbc.to(torch::kCPU).contiguous();
    const int64_t* batch_ptr_data = batch_ptr_cpu.data_ptr<int64_t>();
    const int64_t batch_size = batch_ptr.numel() - 1;
    neighbor_search::require_input(
        batch_ptr_data[0] == 0,
        "batch_ptr must start at zero");
    neighbor_search::require_input(
        batch_ptr_data[batch_size] == positions.size(0),
        "batch_ptr must end at N_total");

    std::vector<int64_t> atom_counts;
    atom_counts.reserve(static_cast<size_t>(batch_size));
    int64_t maximum_atoms = 0;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        neighbor_search::require_input(
            batch_ptr_data[batch + 1] >= batch_ptr_data[batch],
            "batch_ptr must be nondecreasing");
        const int64_t count = batch_ptr_data[batch + 1] - batch_ptr_data[batch];
        atom_counts.push_back(count);
        maximum_atoms = std::max(maximum_atoms, count);
    }
    const bool* pbc_data = pbc_cpu.data_ptr<bool>();
    std::vector<uint8_t> pbc_values(static_cast<size_t>(pbc_cpu.numel()));
    for (int64_t index = 0; index < pbc_cpu.numel(); ++index) {
        pbc_values[index] = pbc_data[index] ? 1 : 0;
    }
    const neighbor_search::PeriodicMetadata metadata =
        neighbor_search::build_periodic_metadata(
            std::span(
                cell_cpu.data_ptr<double>(),
                static_cast<size_t>(cell_cpu.numel())),
            pbc_values,
            atom_counts,
            cutoff);

    std::vector<int64_t> block_offsets = {0};
    std::vector<int64_t> node_offsets = {0};
    block_offsets.reserve(static_cast<size_t>(batch_size + 1));
    node_offsets.reserve(static_cast<size_t>(batch_size + 1));
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t image_count =
            metadata.image_offsets[batch + 1] - metadata.image_offsets[batch];
        const __int128 task_count = static_cast<__int128>(atom_counts[batch]) *
            atom_counts[batch] * image_count;
        const __int128 block_count =
            (task_count + kBlockSize - 1) / kBlockSize;
        const __int128 node_count =
            static_cast<__int128>(atom_counts[batch]) * image_count;
        neighbor_search::require_input(
            block_count <= std::numeric_limits<int64_t>::max() -
                    block_offsets.back(),
            "CUDA search schedule exceeds the int64 range");
        neighbor_search::require_input(
            node_count <= std::numeric_limits<int64_t>::max() -
                    node_offsets.back(),
            "CUDA search schedule exceeds the int64 range");
        block_offsets.push_back(
            block_offsets.back() + static_cast<int64_t>(block_count));
        node_offsets.push_back(
            node_offsets.back() + static_cast<int64_t>(node_count));
    }

    const torch::Device device = positions.device();
    auto duals = tensor_from_vector(
        metadata.duals,
        {batch_size, 3, 3},
        torch::kFloat64,
        device).to(positions.scalar_type());
    auto image_shifts = tensor_from_vector(
        metadata.image_shifts,
        {static_cast<int64_t>(metadata.image_shifts.size() / 3), 3},
        torch::kInt32,
        device);
    auto image_offsets = tensor_from_vector(
        metadata.image_offsets,
        {batch_size + 1},
        torch::kInt64,
        device);
    auto block_offset_tensor = tensor_from_vector(
        block_offsets,
        {batch_size + 1},
        torch::kInt64,
        device);
    auto node_offset_tensor = tensor_from_vector(
        node_offsets,
        {batch_size + 1},
        torch::kInt64,
        device);
    const int64_t total_blocks = block_offsets.back();
    const int64_t total_nodes = node_offsets.back();

    if (maximum_atoms >= kCellListMinimumAtoms &&
        total_nodes < kInt32IndexLimit) {
        return neighbor_list_cuda_cell(
            positions,
            batch_ptr,
            cell,
            duals,
            image_shifts,
            image_offsets,
            block_offset_tensor,
            node_offset_tensor,
            total_blocks,
            total_nodes,
            cutoff,
            half_list,
            include_self);
    }
    neighbor_search::require_input(
        total_blocks < kInt32IndexLimit,
        "the exhaustive CUDA path requires fewer than 2^31 thread blocks");
    return neighbor_list_cuda_exhaustive(
        positions,
        batch_ptr,
        cell,
        duals,
        image_shifts,
        image_offsets,
        block_offset_tensor,
        total_blocks,
        cutoff,
        half_list,
        include_self);
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "neighbor_list",
        &neighbor_list,
        "Find batched neighbor pairs on CUDA");
}
