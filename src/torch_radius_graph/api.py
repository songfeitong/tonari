from __future__ import annotations

from torch import Tensor

from ._geometry import build_search_metadata, validate_inputs


def radius_graph_pbc(
    positions: Tensor,
    ptr: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Construct the complete directed periodic cutoff graph for a CUDA batch.

    ``positions`` and ``cells`` may require gradients, but connectivity is discrete and the returned tensors are integers. Recompute edge vectors from the original floating tensors to differentiate continuous geometry while holding connectivity fixed.
    """

    cutoff = float(cutoff)
    validate_inputs(positions, ptr, cells, pbc, cutoff, require_cuda=True)
    if ptr.detach().cpu()[-1].item() != len(positions):
        raise ValueError("ptr must end at n_atoms_total")
    metadata = build_search_metadata(ptr, cells, pbc, cutoff)
    try:
        from . import _C
    except ImportError as error:
        raise RuntimeError(
            "the torch_radius_graph CUDA extension is not built; install the project with uv sync"
        ) from error
    return _C.radius_graph_pbc_cuda(
        positions.detach().contiguous(),
        ptr.contiguous(),
        cells.detach().contiguous(),
        metadata.duals.contiguous(),
        metadata.image_shifts,
        metadata.image_ptr,
        metadata.block_ptr,
        metadata.total_blocks,
        cutoff,
    )

