#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> radius_graph_pbc_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& ptr,
    const torch::Tensor& cells,
    const torch::Tensor& duals,
    const torch::Tensor& image_shifts,
    const torch::Tensor& image_ptr,
    const torch::Tensor& block_ptr,
    int64_t total_blocks,
    double cutoff);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "radius_graph_pbc_cuda",
        &radius_graph_pbc_cuda,
        "Batched periodic radius graph (CUDA)");
}

