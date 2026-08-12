from __future__ import annotations

import pytest
import torch

from tests.assertions import assert_sorted_by_source, pair_keys
from tests.reference import neighbor_list_reference
from tonari import neighbor_list


def periodic_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1945)
    counts = (32, 288)
    cell = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 6.0]],
            [[8.0, 0.0, 0.0], [0.0, 7.5, 0.0], [0.0, 0.0, 9.0]],
        ],
        dtype=torch.float64,
    )
    positions = torch.cat(
        [
            (
                0.1
                + 0.8 * torch.rand((count, 3), generator=generator, dtype=torch.float64)
            )
            @ cell[index]
            for index, count in enumerate(counts)
        ]
    )
    pbc = torch.tensor([[True, False, False], [True, False, False]])
    batch_ptr = torch.tensor([0, counts[0], sum(counts)])
    return positions, cell, pbc, batch_ptr


@pytest.mark.parametrize("algorithm", ["auto", "brute_force", "cell_list"])
@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
def test_cpu_algorithms_match_reference(algorithm: str, ecosystem: str) -> None:
    positions, cell, pbc, batch_ptr = periodic_batch()
    expected = neighbor_list_reference(
        "PS",
        positions,
        cell,
        pbc,
        1.2,
        batch_ptr,
        half_list=True,
        include_self=True,
    )
    if ecosystem == "numpy":
        actual = neighbor_list(
            "PS",
            positions.numpy(),
            cell.numpy(),
            pbc.numpy(),
            1.2,
            batch_ptr.numpy(),
            algorithm=algorithm,
            sorted=True,
            half_list=True,
            include_self=True,
        )
    else:
        actual = neighbor_list(
            "PS",
            positions,
            cell,
            pbc,
            1.2,
            batch_ptr,
            algorithm=algorithm,
            sorted=True,
            half_list=True,
            include_self=True,
        )
    assert pair_keys(*actual) == pair_keys(*expected)
    assert_sorted_by_source(actual[0])


@pytest.mark.parametrize("cpu_threads", [1, 4])
def test_forced_cpu_cell_list_reports_an_unsupported_layout(
    cpu_threads: int,
) -> None:
    positions = torch.zeros((256, 3))
    positions[:, 0] = torch.arange(256) * 10000.0
    with pytest.raises(RuntimeError, match="cell_list cannot safely"):
        neighbor_list(
            "PS",
            positions,
            torch.zeros((3, 3)),
            torch.zeros(3, dtype=torch.bool),
            1.0,
            algorithm="cell_list",
            cpu_threads=cpu_threads,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("algorithm", ["auto", "brute_force", "cell_list"])
def test_cuda_algorithms_match_reference(algorithm: str) -> None:
    positions, cell, pbc, batch_ptr = periodic_batch()
    expected = neighbor_list_reference(
        "PS",
        positions,
        cell,
        pbc,
        1.2,
        batch_ptr,
        half_list=True,
        include_self=True,
    )
    actual = neighbor_list(
        "PS",
        positions.cuda(),
        cell.cuda(),
        pbc.cuda(),
        1.2,
        batch_ptr.cuda(),
        algorithm=algorithm,
        sorted=True,
        half_list=True,
        include_self=True,
    )
    assert pair_keys(*actual) == pair_keys(*expected)
    assert_sorted_by_source(actual[0])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_forced_cuda_cell_list_reports_an_unsupported_layout() -> None:
    positions = torch.zeros((256, 3), device="cuda")
    positions[:, 0] = torch.arange(256, device="cuda") * 10000.0
    with pytest.raises(RuntimeError, match="cell_list cannot safely"):
        neighbor_list(
            "PS",
            positions,
            torch.zeros((3, 3), device="cuda"),
            torch.zeros(3, dtype=torch.bool, device="cuda"),
            1.0,
            algorithm="cell_list",
        )
