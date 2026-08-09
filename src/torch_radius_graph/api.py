from __future__ import annotations

from torch import Tensor

from ._geometry import build_search_metadata, validate_inputs

_CELL_LIST_MINIMUM_ATOMS = 256
_INT32_INDEX_LIMIT = 2**31


def radius_graph_pbc(
    positions: Tensor,
    ptr: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Construct the complete directed periodic cutoff graph for a CUDA batch.

    ``positions`` and ``cells`` may require gradients, but connectivity is
    discrete and the returned tensors are integers. Recompute edge vectors from
    the original floating tensors to differentiate continuous geometry while
    holding connectivity fixed.
    """

    cutoff = float(cutoff)
    validate_inputs(positions, ptr, cells, pbc, cutoff, require_cuda=True)
    metadata = build_search_metadata(
        ptr, cells, pbc, cutoff, n_atoms_total=len(positions)
    )
    try:
        from . import _C
    except ImportError as error:
        raise RuntimeError(
            "the torch_radius_graph CUDA extension is not built; install the project with uv sync"
        ) from error
    arguments = (
        positions.detach().contiguous(),
        ptr.contiguous(),
        cells.detach().contiguous(),
        metadata.duals.contiguous(),
        metadata.image_shifts,
        metadata.image_ptr,
        metadata.block_ptr,
    )
    if (
        metadata.maximum_atoms >= _CELL_LIST_MINIMUM_ATOMS
        and metadata.total_nodes < _INT32_INDEX_LIMIT
    ):
        return _C.radius_graph_pbc_cell_cuda(
            *arguments,
            metadata.node_ptr,
            metadata.total_blocks,
            metadata.total_nodes,
            cutoff,
        )
    if metadata.total_blocks >= _INT32_INDEX_LIMIT:
        raise ValueError(
            "the exhaustive CUDA path requires fewer than 2^31 thread blocks"
        )
    return _C.radius_graph_pbc_cuda(
        *arguments,
        metadata.total_blocks,
        cutoff,
    )
