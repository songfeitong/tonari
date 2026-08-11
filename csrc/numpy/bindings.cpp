#include "../core/neighbors_cpu.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>


namespace py = pybind11;


namespace {

template <typename scalar_t>
std::pair<py::array, py::array> neighbor_list_typed(
    const py::array& positions,
    const py::array& batch_ptr,
    const py::array& cell,
    const py::array& pbc,
    double cutoff,
    bool half_list,
    bool include_self) {
    const auto position_info = positions.request();
    const auto batch_ptr_info = batch_ptr.request();
    const auto cell_info = cell.request();
    const auto pbc_info = pbc.request();
    neighbor_search::PairBuffers pairs;
    {
        py::gil_scoped_release release;
        pairs = neighbor_search::neighbor_list_cpu<scalar_t>(
            std::span(
                static_cast<const scalar_t*>(position_info.ptr),
                static_cast<size_t>(position_info.size)),
            std::span(
                static_cast<const int64_t*>(batch_ptr_info.ptr),
                static_cast<size_t>(batch_ptr_info.size)),
            std::span(
                static_cast<const scalar_t*>(cell_info.ptr),
                static_cast<size_t>(cell_info.size)),
            std::span(
                static_cast<const uint8_t*>(pbc_info.ptr),
                static_cast<size_t>(pbc_info.size)),
            cutoff,
            neighbor_search::pair_mode(half_list, include_self));
    }

    const auto n_pairs = static_cast<py::ssize_t>(pairs.indices.size() / 2);
    py::array_t<int64_t> pair_indices(
        std::vector<py::ssize_t>{n_pairs, 2});
    py::array_t<int32_t> cell_shifts(
        std::vector<py::ssize_t>{n_pairs, 3});
    if (n_pairs > 0) {
        auto* pair_data = pair_indices.mutable_data();
        std::memcpy(
            pair_data,
            pairs.indices.data(),
            static_cast<size_t>(2 * n_pairs) * sizeof(int64_t));
        std::memcpy(
            cell_shifts.mutable_data(),
            pairs.shifts.data(),
            static_cast<size_t>(3 * n_pairs) * sizeof(int32_t));
    }
    return {std::move(pair_indices), std::move(cell_shifts)};
}


std::pair<py::array, py::array> neighbor_list(
    const py::array& positions,
    const py::array& batch_ptr,
    const py::array& cell,
    const py::array& pbc,
    double cutoff,
    bool half_list,
    bool include_self) {
    const auto require_contiguous = [](const py::array& array, const char* name) {
        if ((array.flags() & py::array::c_style) == 0) {
            throw py::value_error(std::string(name) + " must be C-contiguous");
        }
    };
    require_contiguous(positions, "positions");
    require_contiguous(batch_ptr, "batch_ptr");
    require_contiguous(cell, "cell");
    require_contiguous(pbc, "pbc");
    if (!batch_ptr.dtype().is(py::dtype::of<int64_t>())) {
        throw py::value_error("batch_ptr must have dtype int64");
    }
    if (!pbc.dtype().is(py::dtype::of<bool>())) {
        throw py::value_error("pbc must have dtype bool");
    }
    if (!cell.dtype().is(positions.dtype())) {
        throw py::value_error("cell and positions must have the same dtype");
    }
    if (positions.dtype().is(py::dtype::of<float>())) {
        return neighbor_list_typed<float>(
            positions, batch_ptr, cell, pbc, cutoff, half_list, include_self);
    }
    if (positions.dtype().is(py::dtype::of<double>())) {
        return neighbor_list_typed<double>(
            positions, batch_ptr, cell, pbc, cutoff, half_list, include_self);
    }
    throw py::value_error("positions must have dtype float32 or float64");
}

}  // namespace


PYBIND11_MODULE(_numpy_cpu, module) {
    module.def(
        "neighbor_list",
        &neighbor_list,
        "Find batched neighbor pairs from NumPy arrays");
}
