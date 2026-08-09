#include "radius_graph_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "radius_graph_pbc_cuda",
        &radius_graph_pbc_cuda,
        "Batched periodic radius graph (CUDA)");
    module.def(
        "radius_graph_pbc_cell_cuda",
        &radius_graph_pbc_cell_cuda,
        "Batched periodic radius graph with Cartesian cell lists (CUDA)");
}
