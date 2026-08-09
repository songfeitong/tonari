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
    """为一个 CUDA batch 构造完整的有向 periodic cutoff graph。

    ``positions`` 和 ``cells`` 可以带梯度，但 connectivity 是离散的，返回 tensors 为整数。请用原始浮点 tensors 重算 edge vectors，以便在 connectivity 固定时对连续几何求导。
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
