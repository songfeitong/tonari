#include "radius_graph_cpu.h"
#include "periodic_geometry.h"

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "radius_graph_pbc_cpu",
        &radius_graph_pbc_cpu,
        py::call_guard<py::gil_scoped_release>(),
        "Batched periodic radius graph (CPU)");
    module.def(
        "build_periodic_metadata_cpu",
        &build_periodic_metadata_cpu,
        py::call_guard<py::gil_scoped_release>(),
        "Periodic radius-graph search metadata (CPU)");
}
