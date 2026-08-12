from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor


_VALID_QUANTITIES = frozenset("ijPSdD")
_Algorithm = Literal["auto", "brute_force", "cell_list"]
_VALID_ALGORITHMS = frozenset(("auto", "brute_force", "cell_list"))


@overload
def neighbor_list(
    quantities: str,
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    batch_ptr: Tensor | None = None,
    *,
    algorithm: _Algorithm = "auto",
    num_threads: int = 1,
    sorted: bool = False,
    half_list: bool = False,
    include_self: bool = False,
) -> tuple[Tensor, ...]: ...


@overload
def neighbor_list(
    quantities: str,
    positions: NDArray[np.float32] | NDArray[np.float64],
    cell: NDArray[np.float32] | NDArray[np.float64],
    pbc: NDArray[np.bool_],
    cutoff: float,
    batch_ptr: NDArray[np.int64] | None = None,
    *,
    algorithm: _Algorithm = "auto",
    num_threads: int = 1,
    sorted: bool = False,
    half_list: bool = False,
    include_self: bool = False,
) -> tuple[np.ndarray, ...]: ...


def neighbor_list(
    quantities: str,
    positions: Tensor | np.ndarray,
    cell: Tensor | np.ndarray,
    pbc: Tensor | np.ndarray,
    cutoff: float,
    batch_ptr: Tensor | np.ndarray | None = None,
    *,
    algorithm: _Algorithm = "auto",
    num_threads: int = 1,
    sorted: bool = False,
    half_list: bool = False,
    include_self: bool = False,
) -> tuple[Tensor, ...] | tuple[np.ndarray, ...]:
    """Build an atomistic neighbor list within a strict distance cutoff.

    Args:
        quantities: String selecting the returned quantities and their order.
            The supported characters are ``"i"`` for source indices, ``"j"``
            for target indices, ``"P"`` for paired indices, ``"S"`` for
            integer cell shifts, ``"d"`` for distances, and ``"D"`` for
            displacement vectors. Characters may be repeated. An empty string
            returns an empty tuple.
        positions: Atomic Cartesian positions. For one structure, use an
            ``(N, 3)`` PyTorch tensor or NumPy array. For a batch, concatenate
            all positions into ``(N_total, 3)``. The dtype must be ``float32``
            or ``float64``. Torch inputs may be on CPU or CUDA; NumPy inputs use
            the CPU backend. All values must be finite.
        cell: Cartesian cell vectors stored as rows. Use shape ``(3, 3)`` when
            ``batch_ptr`` is ``None`` and ``(B, 3, 3)`` for a batch. Its
            floating dtype and, for Torch, device must match ``positions``. All
            values must be finite. For every nonempty structure, the rows
            enabled by ``pbc`` must be linearly independent; inactive rows and
            the full cell may be rank deficient.
        pbc: Periodic boundary flags for the three cell rows. Use shape ``(3,)``
            for one structure and ``(B, 3)`` for a batch. The dtype must be
            ``bool`` and the array ecosystem/device must match ``positions``.
        cutoff: Strict, finite, positive distance cutoff. ``positions``,
            ``cell``, and ``cutoff`` must use the same length unit.
        batch_ptr: Optional ``int64`` structure boundaries in the concatenated
            ``positions``, with shape ``(B + 1,)``. It must start at zero, be
            nondecreasing, and end at ``N_total``. ``None`` denotes one
            structure and is equivalent to ``[0, N]``. Its array ecosystem
            and, for Torch, device must match ``positions``.
        algorithm: Search method. ``"auto"`` (default) selects a backend-
            appropriate method. ``"brute_force"`` exhaustively checks every
            relevant atom pair. ``"cell_list"`` partitions space so that atoms
            only inspect nearby regions.
        num_threads: Number of CPU threads used by this call, including the
            calling thread. It must be a positive integer and defaults to ``1``
            to avoid implicit oversubscription. CPU workers are reused across
            calls. CUDA calls only accept the default value because CUDA
            execution does not use the CPU search pool.
        sorted: If ``True``, sort pairs by source index. The order of target
            indices and cell shifts within each source is unspecified. The
            default is ``False``.
        half_list: If ``False`` (default), return the full directed list. If
            ``True``, retain the lexicographically smaller of
            ``(source, target, Sx, Sy, Sz)`` and
            ``(target, source, -Sx, -Sy, -Sz)``.
        include_self: Whether to include exactly one zero-shift self pair
            ``(i, i, [0, 0, 0])`` for every atom. The default is ``False``.

    Returns:
        A tuple containing one array for each character in ``quantities``, in
        the same order. All arrays use the same ecosystem as the inputs and,
        for Torch, the same device. If ``E`` pairs are found:

        - ``i`` and ``j`` have dtype ``int64`` and shape ``(E,)``.
        - ``P`` has dtype ``int64`` and shape ``(E, 2)``; its columns are
          ``source`` and ``target``.
        - ``S`` has dtype ``int32`` and shape ``(E, 3)`` and translates the
          target image.
        - ``d`` has the input floating dtype and shape ``(E,)``.
        - ``D`` has the input floating dtype and shape ``(E, 3)``.

        For pair ``k`` in structure ``b``, ``D[k]`` is
        ``positions[target[k]] - positions[source[k]] + S[k] @ cell[b]``.
        For a single structure, use ``cell`` directly.

    Raises:
        TypeError: If ``quantities`` or ``algorithm`` is not a string;
            ``num_threads`` is not a Python ``int``; array arguments mix
            PyTorch and NumPy or use an unsupported container type; or a
            boolean option is not a Python ``bool``.
        ValueError: If ``quantities`` contains an unsupported character;
            ``algorithm`` is unsupported; or frontend shapes, dtypes, devices,
            ``batch_ptr``, ``cutoff``, ``num_threads``, periodic cells, or
            host-validated index/resource bounds are invalid.
        RuntimeError: If the required native CPU or CUDA extension is missing;
            if native search discovers nonfinite positions or a representative
            wrap/output shift outside its integer range; if an explicitly
            requested cell list cannot safely process the input; or if backend
            execution otherwise fails.

    Note:
        The result contains atom-image pairs whose squared distance is strictly
        less than ``cutoff**2``. ``half_list=False`` returns both directions:
        pair ``(source, target, S)`` has reverse pair
        ``(target, source, -S)``. A zero-shift self pair is controlled only by
        ``include_self``. Periodic self-images remain ordinary cutoff pairs,
        and multiple periodic images are retained. Pairs never cross structures,
        shifts along inactive ``pbc`` axes are zero. Output order is
        unspecified unless ``sorted=True``.

        Neighbor identity is discrete and is not differentiable. For Torch
        inputs, returned distances and displacement vectors are computed from
        the original floating tensors and remain differentiable while the
        neighbor identity is fixed.

    Example:
        >>> import torch
        >>> from tonari import neighbor_list
        >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        >>> cell = torch.eye(3) * 4.0
        >>> pbc = torch.tensor([False, False, False])
        >>> pairs, shifts, distances = neighbor_list(
        ...     "PSd", positions, cell, pbc, cutoff=1.0
        ... )
        >>> pairs.shape, shifts.shape, distances.tolist()
        (torch.Size([2, 2]), torch.Size([2, 3]), [0.800000011920929, 0.800000011920929])
    """

    if not isinstance(quantities, str):
        raise TypeError("quantities must be a string")
    invalid_quantities = set(quantities) - _VALID_QUANTITIES
    if invalid_quantities:
        invalid = "".join(builtins.sorted(invalid_quantities))
        raise ValueError(f"unsupported quantities: {invalid!r}")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError("algorithm must be 'auto', 'brute_force', or 'cell_list'")
    if isinstance(num_threads, bool) or not isinstance(num_threads, int):
        raise TypeError("num_threads must be an integer")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if num_threads > (1 << 63) - 1:
        raise ValueError("num_threads is too large")
    if not all(
        isinstance(option, bool) for option in (sorted, half_list, include_self)
    ):
        raise TypeError("sorted, half_list, and include_self must be bool")
    if isinstance(positions, np.ndarray):
        from ._numpy_frontend import neighbor_list_numpy

        return neighbor_list_numpy(
            quantities,
            positions,
            cell,
            pbc,
            cutoff,
            batch_ptr,
            algorithm=algorithm,
            num_threads=num_threads,
            sorted=sorted,
            half_list=half_list,
            include_self=include_self,
        )
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(positions, torch.Tensor):
        from ._torch_frontend import neighbor_list_torch

        return neighbor_list_torch(
            quantities,
            positions,
            cell,
            pbc,
            cutoff,
            batch_ptr,
            algorithm=algorithm,
            num_threads=num_threads,
            sorted=sorted,
            half_list=half_list,
            include_self=include_self,
        )
    raise TypeError("positions must be a PyTorch tensor or NumPy array")
