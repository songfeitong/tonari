from __future__ import annotations

import torch

from benchmarks.baselines import dense_candidate_count, torch_dense_batch


def test_dense_baseline_supports_finite_batches() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch_ptr = torch.tensor([0, 2, 3])
    cell = torch.zeros((2, 3, 3), dtype=torch.float64)
    pbc = torch.zeros((2, 3), dtype=torch.bool)
    pair_indices, cell_shifts = torch_dense_batch(positions, cell, pbc, 1.0, batch_ptr)
    assert dense_candidate_count(batch_ptr, cell, pbc, 1.0) == 5
    assert set(map(tuple, pair_indices.tolist())) == {(0, 1), (1, 0)}
    assert torch.count_nonzero(cell_shifts) == 0
