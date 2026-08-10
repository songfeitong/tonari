#pragma once

#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> find_neighbors_cpu(
    const torch::Tensor& positions,
    const torch::Tensor& offsets,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_offsets,
    double cutoff,
    bool half_list,
    bool include_self);
