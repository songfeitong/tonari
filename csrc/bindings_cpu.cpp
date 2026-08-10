#include "neighbors_cpu.h"
#include "periodic_geometry.h"

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "find_neighbors_cpu",
        &find_neighbors_cpu,
        py::call_guard<py::gil_scoped_release>(),
        "Find batched neighbor pairs on CPU");
    module.def(
        "build_periodic_metadata_cpu",
        &build_periodic_metadata_cpu,
        py::call_guard<py::gil_scoped_release>(),
        "Build periodic neighbor-search metadata on CPU");
}
