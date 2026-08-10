from __future__ import annotations

from typing import overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ._search import build_cuda_schedule, build_search_metadata, validate_torch_inputs

_CELL_LIST_MINIMUM_ATOMS = 256
_INT32_INDEX_LIMIT = 2**31


@overload
def find_neighbors(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor | None = None,
) -> tuple[Tensor, Tensor]: ...


@overload
def find_neighbors(
    positions: NDArray[np.float32] | NDArray[np.float64],
    cells: NDArray[np.float32] | NDArray[np.float64],
    pbc: NDArray[np.bool_],
    cutoff: float,
    offsets: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int32]]: ...


def find_neighbors(
    positions: Tensor | np.ndarray,
    cells: Tensor | np.ndarray,
    pbc: Tensor | np.ndarray,
    cutoff: float,
    offsets: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor] | tuple[np.ndarray, np.ndarray]:
    """Find all directed atom-image pairs within a strict distance cutoff.

    Args:
        positions: Atomic Cartesian positions. For a single structure, use a
            ``(N, 3)`` PyTorch tensor or NumPy array. For a batch, concatenate
            all positions into ``(N_total, 3)``. The dtype must be ``float32``
            or ``float64``. Torch inputs may be on CPU or CUDA; NumPy inputs use
            the CPU backend. All values must be finite.
        cells: Cartesian cell vectors stored as rows. Use shape ``(3, 3)`` when
            ``offsets`` is ``None`` and ``(B, 3, 3)`` for a batch. Its floating
            dtype and, for Torch, device must match ``positions``. All values
            must be finite. For every nonempty structure, the rows enabled by
            ``pbc`` must be linearly independent; inactive rows and the full
            cell may be rank deficient.
        pbc: Periodic boundary flags for the three cell rows. Use shape ``(3,)``
            for one structure and ``(B, 3)`` for a batch. The dtype must be
            ``bool`` and the array ecosystem/device must match ``positions``.
        cutoff: Strict, finite, positive distance cutoff. ``positions``,
            ``cells``, and ``cutoff`` must use the same length unit.
        offsets: Optional ``int64`` structure boundaries in the concatenated
            ``positions``, with shape ``(B + 1,)``. It must start at zero, be
            nondecreasing, and end at ``N_total``. ``None`` denotes one
            structure and is equivalent to ``[0, N]``. Its array ecosystem
            and, for Torch, device must match ``positions``.

    Returns:
        A tuple ``(pair_indices, cell_shifts)`` in the same array ecosystem as
        the inputs and, for Torch, on the same device. ``pair_indices`` has
        dtype ``int64`` and shape ``(2, num_pairs)``;
        ``source, target = pair_indices``. ``cell_shifts`` has dtype ``int32``
        and shape ``(num_pairs, 3)``. Each shift translates the source image.
        For one structure, pair ``k`` has Cartesian displacement
        ``positions[source[k]] - positions[target[k]] + cell_shifts[k] @
        cells``. For a batch, first locate structure ``b`` from ``offsets`` and
        use the same formula with ``cells[b]``. Both outputs are dimensionless.

    Raises:
        TypeError: If array arguments mix PyTorch and NumPy, or use an
            unsupported container type.
        ValueError: If frontend shapes, dtypes, devices, offsets, cutoff,
            periodic cells, or host-validated index/resource bounds are invalid.
        RuntimeError: If the required native CPU or CUDA extension is missing;
            if native search discovers nonfinite positions or a representative
            wrap/output shift outside its integer range; or if backend execution
            otherwise fails.

    Note:
        The result contains every directed atom-image pair whose squared
        distance is strictly less than ``cutoff**2``. The zero-shift onsite pair
        ``(i, i, [0, 0, 0])`` is excluded, while periodic self-images and
        multiple images of the same atom pair are retained. Pairs never cross
        structures, and shifts along inactive ``pbc`` axes are zero. Output
        order is unspecified.

        Pair discovery is discrete and is not differentiable. Torch
        ``positions`` and ``cells`` may require gradients; recompute
        displacements from the returned integer arrays and the original
        floating tensors to differentiate continuous geometry while holding
        the neighbor identity fixed.

    Example:
        >>> import torch
        >>> from tonari import find_neighbors
        >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        >>> cells = torch.eye(3) * 4.0
        >>> pbc = torch.tensor([False, False, False])
        >>> pair_indices, cell_shifts = find_neighbors(
        ...     positions, cells, pbc, cutoff=1.0
        ... )
        >>> source, target = pair_indices
        >>> displacements = (
        ...     positions[source]
        ...     - positions[target]
        ...     + cell_shifts.to(positions.dtype) @ cells
        ... )
        >>> distances = torch.linalg.vector_norm(displacements, dim=1)
        >>> pair_indices.shape, distances.tolist()
        (torch.Size([2, 2]), [0.800000011920929, 0.800000011920929])
    """

    if isinstance(positions, Tensor):
        _require_matching_ecosystem(Tensor, cells=cells, pbc=pbc, offsets=offsets)
        return _find_neighbors_torch(positions, cells, pbc, cutoff, offsets)
    if isinstance(positions, np.ndarray):
        _require_matching_ecosystem(np.ndarray, cells=cells, pbc=pbc, offsets=offsets)
        return _find_neighbors_numpy(positions, cells, pbc, cutoff, offsets)
    raise TypeError("positions must be a PyTorch tensor or NumPy array")


def _require_matching_ecosystem(
    expected_type: type[Tensor | np.ndarray],
    **arrays: Tensor | np.ndarray | None,
) -> None:
    for name, array in arrays.items():
        if array is not None and not isinstance(array, expected_type):
            ecosystem = "PyTorch tensors" if expected_type is Tensor else "NumPy arrays"
            raise TypeError(
                f"positions, cells, pbc, and offsets must all be {ecosystem}; "
                f"{name} has type {type(array).__name__}"
            )


def _normalize_torch_inputs(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    offsets: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N_total, 3)")
    if offsets is None:
        if cells.ndim != 2 or cells.shape != (3, 3):
            raise ValueError("single-structure cells must have shape (3, 3)")
        if pbc.ndim != 1 or pbc.shape != (3,):
            raise ValueError("single-structure pbc must have shape (3,)")
        offsets = torch.tensor(
            [0, len(positions)], dtype=torch.int64, device=positions.device
        )
        cells = cells.unsqueeze(0)
        pbc = pbc.unsqueeze(0)
    else:
        if cells.ndim != 3:
            raise ValueError("batched cells must have shape (B, 3, 3)")
        if pbc.ndim != 2:
            raise ValueError("batched pbc must have shape (B, 3)")
    return positions, cells, pbc, offsets


def _find_neighbors_torch(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor | None,
) -> tuple[Tensor, Tensor]:
    positions, cells, pbc, offsets = _normalize_torch_inputs(
        positions, cells, pbc, offsets
    )
    cutoff = float(cutoff)
    validate_torch_inputs(positions, cells, pbc, cutoff, offsets)
    metadata = build_search_metadata(
        offsets, cells, pbc, cutoff, n_atoms_total=len(positions)
    )
    arguments = (
        positions.detach().contiguous(),
        offsets.contiguous(),
        cells.detach().contiguous(),
        metadata.duals.contiguous(),
        metadata.image_shifts,
        metadata.image_offsets,
    )
    if not positions.is_cuda:
        try:
            from . import _C_cpu
        except ImportError as error:
            raise RuntimeError(
                "the tonari CPU extension is not built; install the project"
            ) from error
        return _C_cpu.find_neighbors_cpu(*arguments, cutoff)

    try:
        from . import _C_cuda
    except ImportError as error:
        raise RuntimeError(
            "the tonari CUDA extension is not built; install with a CUDA toolkit"
        ) from error
    schedule = build_cuda_schedule(metadata)
    cuda_arguments = (*arguments, schedule.block_offsets)
    if (
        metadata.maximum_atoms >= _CELL_LIST_MINIMUM_ATOMS
        and schedule.total_nodes < _INT32_INDEX_LIMIT
    ):
        return _C_cuda.find_neighbors_cell_cuda(
            *cuda_arguments,
            schedule.node_offsets,
            schedule.total_blocks,
            schedule.total_nodes,
            cutoff,
        )
    if schedule.total_blocks >= _INT32_INDEX_LIMIT:
        raise ValueError(
            "the exhaustive CUDA path requires fewer than 2^31 thread blocks"
        )
    return _C_cuda.find_neighbors_cuda(
        *cuda_arguments,
        schedule.total_blocks,
        cutoff,
    )


def _numpy_to_torch(array: np.ndarray) -> Tensor:
    if (
        not array.flags.writeable
        or not array.flags.aligned
        or any(stride < 0 for stride in array.strides)
    ):
        array = np.array(array, copy=True, order="C")
    return torch.from_numpy(array)


def _find_neighbors_numpy(
    positions: np.ndarray,
    cells: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
    offsets: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    torch_offsets = None if offsets is None else _numpy_to_torch(offsets)
    pair_indices, cell_shifts = _find_neighbors_torch(
        _numpy_to_torch(positions),
        _numpy_to_torch(cells),
        _numpy_to_torch(pbc),
        cutoff,
        torch_offsets,
    )
    return pair_indices.numpy(), cell_shifts.numpy()
