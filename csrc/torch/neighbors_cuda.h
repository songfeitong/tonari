#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>


std::vector<torch::Tensor> neighbor_list_cuda_brute_force(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cell,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    int64_t total_blocks,
    double cutoff,
    bool half_list,
    bool include_self);

std::vector<torch::Tensor> neighbor_list_cuda_cell(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cell,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    const torch::Tensor& block_offsets,
    const torch::Tensor& node_offsets,
    int64_t total_blocks,
    int64_t total_nodes,
    double cutoff,
    bool half_list,
    bool include_self,
    bool fallback_to_brute_force);
