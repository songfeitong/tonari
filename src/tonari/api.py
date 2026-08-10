from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor


@overload
def find_neighbors(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor | None = None,
    *,
    half_list: bool = False,
    include_self: bool = False,
) -> tuple[Tensor, Tensor]: ...


@overload
def find_neighbors(
    positions: NDArray[np.float32] | NDArray[np.float64],
    cells: NDArray[np.float32] | NDArray[np.float64],
    pbc: NDArray[np.bool_],
    cutoff: float,
    offsets: NDArray[np.int64] | None = None,
    *,
    half_list: bool = False,
    include_self: bool = False,
) -> tuple[NDArray[np.int64], NDArray[np.int32]]: ...


def find_neighbors(
    positions: Tensor | np.ndarray,
    cells: Tensor | np.ndarray,
    pbc: Tensor | np.ndarray,
    cutoff: float,
    offsets: Tensor | np.ndarray | None = None,
    *,
    half_list: bool = False,
    include_self: bool = False,
) -> tuple[Tensor, Tensor] | tuple[np.ndarray, np.ndarray]:
    """Find atom-image pairs within a strict distance cutoff.

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
        half_list: If ``False`` (default), return the full directed list. If
            ``True``, retain one canonical representative from each pair and
            reverse-pair equivalence class. The canonical representative is
            the lexicographically smaller of ``(source, target, Sx, Sy, Sz)``
            and ``(target, source, -Sx, -Sy, -Sz)``.
        include_self: Whether to include exactly one zero-shift self pair
            ``(i, i, [0, 0, 0])`` for every atom. The default is ``False``.

    Returns:
        A tuple ``(pair_indices, cell_shifts)`` in the same array ecosystem as
        the inputs and, for Torch, on the same device. ``pair_indices`` has
        dtype ``int64`` and shape ``(2, num_pairs)``;
        ``source, target = pair_indices``. ``cell_shifts`` has dtype ``int32``
        and shape ``(num_pairs, 3)``. Each shift translates the target image.
        For one structure, pair ``k`` has Cartesian displacement
        ``positions[target[k]] - positions[source[k]] + cell_shifts[k] @
        cells``. For a batch, first locate structure ``b`` from ``offsets`` and
        use the same formula with ``cells[b]``. Both outputs are dimensionless.

    Raises:
        TypeError: If array arguments mix PyTorch and NumPy, use an unsupported
            container type, or either pair option is not a Python ``bool``.
        ValueError: If frontend shapes, dtypes, devices, offsets, cutoff,
            periodic cells, or host-validated index/resource bounds are invalid.
        RuntimeError: If the required native CPU or CUDA extension is missing;
            if native search discovers nonfinite positions or a representative
            wrap/output shift outside its integer range; or if backend execution
            otherwise fails.

    Note:
        The result contains atom-image pairs whose squared distance is strictly
        less than ``cutoff**2``. ``half_list=False`` returns both directions:
        pair ``(source, target, S)`` has reverse pair
        ``(target, source, -S)``. ``half_list=True`` returns only the canonical
        direction defined above. A zero-shift self pair
        ``(i, i, [0, 0, 0])`` is controlled only by ``include_self`` and has
        zero displacement and distance. Periodic self-images ``(i, i, S)``
        with ``S != 0`` remain ordinary cutoff pairs; a half list keeps one of
        ``S`` and ``-S``. Multiple periodic images are retained rather than
        reduced to a minimum image. Pairs never cross structures, shifts along
        inactive ``pbc`` axes are zero, and output order is unspecified.

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
        ...     positions[target]
        ...     - positions[source]
        ...     + cell_shifts.to(positions.dtype) @ cells
        ... )
        >>> distances = torch.linalg.vector_norm(displacements, dim=1)
        >>> pair_indices.shape, distances.tolist()
        (torch.Size([2, 2]), [0.800000011920929, 0.800000011920929])
    """

    if not isinstance(half_list, bool) or not isinstance(include_self, bool):
        raise TypeError("half_list and include_self must be bool")
    if isinstance(positions, np.ndarray):
        from ._numpy_frontend import find_neighbors_numpy

        return find_neighbors_numpy(
            positions,
            cells,
            pbc,
            cutoff,
            offsets,
            half_list=half_list,
            include_self=include_self,
        )
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(positions, torch.Tensor):
        from ._torch_frontend import find_neighbors_torch

        return find_neighbors_torch(
            positions,
            cells,
            pbc,
            cutoff,
            offsets,
            half_list=half_list,
            include_self=include_self,
        )
    raise TypeError("positions must be a PyTorch tensor or NumPy array")
