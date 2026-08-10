#pragma once

#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> build_periodic_metadata_cpu(
    const torch::Tensor& cells,
    const torch::Tensor& pbc,
    const torch::Tensor& atom_counts,
    double cutoff);
