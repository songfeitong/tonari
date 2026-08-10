#include "neighbors_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "find_neighbors_cuda",
        &find_neighbors_cuda,
        "Find batched neighbor pairs exhaustively on CUDA");
    module.def(
        "find_neighbors_cell_cuda",
        &find_neighbors_cell_cuda,
        "Find batched neighbor pairs with Cartesian cell lists on CUDA");
}
