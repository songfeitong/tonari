#pragma once

#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> radius_graph_pbc_cpu(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    double cutoff);
