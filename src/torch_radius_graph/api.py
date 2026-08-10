from __future__ import annotations

from torch import Tensor

from ._geometry import build_cuda_schedule, build_search_metadata, validate_inputs

_CELL_LIST_MINIMUM_ATOMS = 256
_INT32_INDEX_LIMIT = 2**31


def radius_graph_pbc(
    positions: Tensor,
    ptr: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Construct the complete directed periodic cutoff graph for a tensor batch.

    ``positions`` and ``cells`` may require gradients, but connectivity is
    discrete and the returned tensors are integers. Recompute edge vectors from
    the original floating tensors to differentiate continuous geometry while
    holding connectivity fixed.
    """

    cutoff = float(cutoff)
    validate_inputs(positions, ptr, cells, pbc, cutoff)
    metadata = build_search_metadata(
        ptr, cells, pbc, cutoff, n_atoms_total=len(positions)
    )
    arguments = (
        positions.detach().contiguous(),
        ptr.contiguous(),
        cells.detach().contiguous(),
        metadata.duals.contiguous(),
        metadata.image_shifts,
        metadata.image_ptr,
    )
    if not positions.is_cuda:
        try:
            from . import _C_cpu
        except ImportError as error:
            raise RuntimeError(
                "the torch_radius_graph CPU extension is not built; install the project"
            ) from error
        return _C_cpu.radius_graph_pbc_cpu(*arguments, cutoff)

    try:
        from . import _C_cuda
    except ImportError as error:
        raise RuntimeError(
            "the torch_radius_graph CUDA extension is not built; install with a CUDA toolkit"
        ) from error
    schedule = build_cuda_schedule(metadata)
    cuda_arguments = (*arguments, schedule.block_ptr)
    if (
        metadata.maximum_atoms >= _CELL_LIST_MINIMUM_ATOMS
        and schedule.total_nodes < _INT32_INDEX_LIMIT
    ):
        return _C_cuda.radius_graph_pbc_cell_cuda(
            *cuda_arguments,
            schedule.node_ptr,
            schedule.total_blocks,
            schedule.total_nodes,
            cutoff,
        )
    if schedule.total_blocks >= _INT32_INDEX_LIMIT:
        raise ValueError(
            "the exhaustive CUDA path requires fewer than 2^31 thread blocks"
        )
    return _C_cuda.radius_graph_pbc_cuda(
        *cuda_arguments,
        schedule.total_blocks,
        cutoff,
    )
