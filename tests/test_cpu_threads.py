from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from tests.reference import neighbor_list_reference
from tonari import neighbor_list


def pair_keys(
    pair_indices: np.ndarray | torch.Tensor,
    cell_shifts: np.ndarray | torch.Tensor,
) -> set[tuple[int, ...]]:
    if isinstance(pair_indices, np.ndarray):
        rows = np.concatenate(
            (pair_indices, cell_shifts.astype(np.int64)), axis=1
        ).tolist()
    else:
        rows = (
            torch.cat((pair_indices, cell_shifts.to(torch.int64)), dim=1).cpu().tolist()
        )
    return {tuple(row) for row in rows}


def heterogeneous_batch() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20_260_812)
    counts = (0, 12, 144, 0, 288)
    cells = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[7.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.4, 0.1, 8.0]],
            [[1e-12, 0.0, 0.0], [0.0, 1e-12, 0.0], [0.0, 0.0, 1e-12]],
            [[9.0, 0.3, 0.2], [0.1, 8.5, 0.4], [0.2, 0.5, 9.5]],
        ],
        dtype=torch.float64,
    )
    pbc = torch.tensor(
        [
            [False, False, False],
            [False, False, False],
            [True, False, True],
            [True, True, True],
            [True, True, True],
        ]
    )
    finite = 4 * torch.rand((counts[1], 3), generator=generator, dtype=torch.float64)
    partial = (
        torch.rand((counts[2], 3), generator=generator, dtype=torch.float64) @ cells[2]
    )
    partial += torch.tensor([2.0, 0.0, -1.0], dtype=torch.float64) @ cells[2]
    periodic = (
        torch.rand((counts[4], 3), generator=generator, dtype=torch.float64) @ cells[4]
    )
    positions = torch.cat((finite, partial, periodic))
    batch_ptr = torch.tensor((0, 0, 12, 156, 156, 444), dtype=torch.int64)
    return positions, cells, pbc, batch_ptr


@pytest.fixture(scope="module")
def heterogeneous_reference() -> tuple[torch.Tensor, torch.Tensor]:
    positions, cell, pbc, batch_ptr = heterogeneous_batch()
    return neighbor_list_reference("PS", positions, cell, pbc, 1.0, batch_ptr)


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize("algorithm", ["auto", "brute_force", "cell_list"])
@pytest.mark.parametrize("num_threads", [1, 2, 4])
def test_cpu_thread_counts_match_exact_reference(
    ecosystem: str,
    algorithm: str,
    num_threads: int,
    heterogeneous_reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    positions, cell, pbc, batch_ptr = heterogeneous_batch()
    arguments: tuple[object, ...]
    if ecosystem == "numpy":
        arguments = (
            positions.numpy(),
            cell.numpy(),
            pbc.numpy(),
            batch_ptr.numpy(),
        )
    else:
        arguments = (positions, cell, pbc, batch_ptr)
    actual = neighbor_list(
        "PS",
        arguments[0],
        arguments[1],
        arguments[2],
        1.0,
        arguments[3],
        algorithm=algorithm,
        num_threads=num_threads,
        sorted=True,
    )
    assert pair_keys(*actual) == pair_keys(*heterogeneous_reference)
    source = actual[0][:, 0]
    assert (
        bool(np.all(source[1:] >= source[:-1]))
        if ecosystem == "numpy"
        else bool(torch.all(source[1:] >= source[:-1]))
    )


@pytest.mark.parametrize("half_list", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_threaded_pair_modes_and_quantities_stay_aligned(
    half_list: bool, include_self: bool
) -> None:
    positions, cell, pbc, batch_ptr = heterogeneous_batch()
    expected = neighbor_list_reference(
        "PS",
        positions,
        cell,
        pbc,
        1.0,
        batch_ptr,
        half_list=half_list,
        include_self=include_self,
    )
    pairs, shifts, distances, displacements = neighbor_list(
        "PSdD",
        positions,
        cell,
        pbc,
        1.0,
        batch_ptr,
        num_threads=4,
        sorted=True,
        half_list=half_list,
        include_self=include_self,
    )
    assert pair_keys(pairs, shifts) == pair_keys(*expected)
    pair_batch = torch.bucketize(pairs[:, 0].contiguous(), batch_ptr[1:], right=True)
    expected_displacements = (
        positions[pairs[:, 1]]
        - positions[pairs[:, 0]]
        + torch.einsum("ei,eij->ej", shifts.to(positions.dtype), cell[pair_batch])
    )
    assert torch.equal(displacements, expected_displacements)
    assert torch.equal(distances, torch.linalg.vector_norm(displacements, dim=1))
    assert torch.all(pairs[1:, 0] >= pairs[:-1, 0])


def test_large_single_structure_matches_across_thread_counts() -> None:
    generator = torch.Generator().manual_seed(1948)
    cell = torch.tensor(
        [[24.0, 0.4, 0.2], [0.3, 23.0, 0.5], [0.2, 0.4, 25.0]],
        dtype=torch.float64,
    )
    positions = torch.rand((2048, 3), generator=generator, dtype=torch.float64) @ cell
    pbc = torch.ones(3, dtype=torch.bool)
    expected = neighbor_list(
        "PS", positions, cell, pbc, 1.2, algorithm="cell_list", num_threads=1
    )
    for num_threads in (2, 4, 8):
        actual = neighbor_list(
            "PS",
            positions,
            cell,
            pbc,
            1.2,
            algorithm="cell_list",
            num_threads=num_threads,
        )
        assert pair_keys(*actual) == pair_keys(*expected)


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
def test_concurrent_python_calls_share_workers_safely(ecosystem: str) -> None:
    positions, cell, pbc, batch_ptr = heterogeneous_batch()
    if ecosystem == "numpy":
        arguments = (
            positions.numpy(),
            cell.numpy(),
            pbc.numpy(),
            batch_ptr.numpy(),
        )
    else:
        arguments = (positions, cell, pbc, batch_ptr)

    def run() -> set[tuple[int, ...]]:
        return pair_keys(
            *neighbor_list(
                "PS",
                arguments[0],
                arguments[1],
                arguments[2],
                1.0,
                arguments[3],
                num_threads=4,
            )
        )

    expected = run()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: run(), range(8)))
    assert all(result == expected for result in results)


def test_numpy_and_torch_cpu_calls_can_overlap() -> None:
    positions, cell, pbc, batch_ptr = heterogeneous_batch()
    torch_arguments = (positions, cell, pbc, batch_ptr)
    numpy_arguments = tuple(tensor.numpy() for tensor in torch_arguments)
    expected = pair_keys(
        *neighbor_list("PS", positions, cell, pbc, 1.0, batch_ptr, num_threads=4)
    )

    def run(ecosystem: str) -> set[tuple[int, ...]]:
        arguments = numpy_arguments if ecosystem == "numpy" else torch_arguments
        return pair_keys(
            *neighbor_list(
                "PS",
                arguments[0],
                arguments[1],
                arguments[2],
                1.0,
                arguments[3],
                num_threads=4,
            )
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, ("numpy", "torch") * 4))
    assert all(result == expected for result in results)


def test_parallel_exception_propagates_and_pool_remains_usable() -> None:
    positions, cell, pbc, batch_ptr = heterogeneous_batch()
    invalid = positions.clone()
    invalid[20, 0] = torch.nan
    with pytest.raises(RuntimeError, match="positions must contain only finite values"):
        neighbor_list("PS", invalid, cell, pbc, 1.0, batch_ptr, num_threads=4)
    actual = neighbor_list("PS", positions, cell, pbc, 1.0, batch_ptr, num_threads=4)
    assert len(actual[0]) > 0


def _forked_numpy_search(queue: multiprocessing.Queue) -> None:
    positions = np.arange(384, dtype=np.float64).reshape(128, 3) * 0.01
    pairs, _ = neighbor_list(
        "PS",
        positions,
        np.zeros((3, 3), dtype=np.float64),
        np.zeros(3, dtype=np.bool_),
        0.1,
        num_threads=2,
    )
    queue.put(len(pairs))


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable"
)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_worker_pool_reinitializes_after_fork() -> None:
    positions = np.arange(384, dtype=np.float64).reshape(128, 3) * 0.01
    neighbor_list(
        "PS",
        positions,
        np.zeros((3, 3), dtype=np.float64),
        np.zeros(3, dtype=np.bool_),
        0.1,
        num_threads=4,
    )
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_forked_numpy_search, args=(queue,))
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("forked child deadlocked in the inherited CPU worker pool")
    assert process.exitcode == 0
    assert queue.get(timeout=1) > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_rejects_cpu_thread_override() -> None:
    with pytest.raises(ValueError, match="only applies to CPU"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), device="cuda"),
            torch.zeros((3, 3), device="cuda"),
            torch.zeros(3, dtype=torch.bool, device="cuda"),
            1.0,
            num_threads=2,
        )
