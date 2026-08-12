#include "../core/neighbors_cpu.h"

#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>


namespace py = pybind11;


namespace {

template <typename scalar_t>
std::vector<torch::Tensor> neighbor_list_typed(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cell,
    const torch::Tensor& pbc,
    double cutoff,
    bool half_list,
    bool include_self,
    neighbor_search::Algorithm algorithm,
    int64_t num_threads) {
    const bool* pbc_data = pbc.data_ptr<bool>();
    std::vector<uint8_t> pbc_values(static_cast<size_t>(pbc.numel()));
    for (int64_t index = 0; index < pbc.numel(); ++index) {
        pbc_values[index] = pbc_data[index] ? 1 : 0;
    }
    neighbor_search::PairBuffers pairs = neighbor_search::neighbor_list_cpu<scalar_t>(
        std::span(
            positions.data_ptr<scalar_t>(),
            static_cast<size_t>(positions.numel())),
        std::span(
            batch_ptr.data_ptr<int64_t>(),
            static_cast<size_t>(batch_ptr.numel())),
        std::span(
            cell.data_ptr<scalar_t>(),
            static_cast<size_t>(cell.numel())),
        pbc_values,
        cutoff,
        neighbor_search::pair_mode(half_list, include_self),
        algorithm,
        num_threads);

    neighbor_search::require_search(
        pairs.pair_count <=
            static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "neighbor-list output exceeds the int64 range");
    const int64_t n_pairs = static_cast<int64_t>(pairs.pair_count);
    auto pair_indices =
        torch::empty({n_pairs, 2}, positions.options().dtype(torch::kInt64));
    auto cell_shifts =
        torch::empty({n_pairs, 3}, positions.options().dtype(torch::kInt32));
    if (n_pairs > 0) {
        neighbor_search::copy_pair_buffers(
            pairs,
            std::span(
                pair_indices.data_ptr<int64_t>(),
                2 * static_cast<size_t>(n_pairs)),
            std::span(
                cell_shifts.data_ptr<int32_t>(),
                3 * static_cast<size_t>(n_pairs)),
            num_threads);
    }
    return {pair_indices, cell_shifts};
}


std::vector<torch::Tensor> neighbor_list(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cell,
    const torch::Tensor& pbc,
    double cutoff,
    bool half_list,
    bool include_self,
    const std::string& algorithm,
    int64_t num_threads) {
    TORCH_CHECK(!positions.is_cuda(), "positions must be a CPU tensor");
    TORCH_CHECK(
        !batch_ptr.is_cuda() && !cell.is_cuda() && !pbc.is_cuda(),
        "all inputs must be CPU tensors");
    TORCH_CHECK(
        positions.is_contiguous() && batch_ptr.is_contiguous() &&
            cell.is_contiguous() && pbc.is_contiguous(),
        "all inputs must be contiguous");
    TORCH_CHECK(batch_ptr.scalar_type() == torch::kInt64);
    TORCH_CHECK(pbc.scalar_type() == torch::kBool);
    TORCH_CHECK(positions.scalar_type() == cell.scalar_type());
    const neighbor_search::Algorithm parsed_algorithm =
        neighbor_search::parse_algorithm(algorithm);
    std::vector<torch::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(
        positions.scalar_type(), "neighbor_search_cpu", [&] {
            result = neighbor_list_typed<scalar_t>(
                positions,
                batch_ptr,
                cell,
                pbc,
                cutoff,
                half_list,
                include_self,
                parsed_algorithm,
                num_threads);
        });
    return result;
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "neighbor_list",
        &neighbor_list,
        py::call_guard<py::gil_scoped_release>(),
        "Find batched neighbor pairs on CPU");
}
