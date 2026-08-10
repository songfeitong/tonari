#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>


std::vector<torch::Tensor> find_neighbors_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& offsets,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    int64_t total_blocks,
    double cutoff,
    bool half_list,
    bool include_self);

std::vector<torch::Tensor> find_neighbors_cell_cuda(
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
    bool include_self);
