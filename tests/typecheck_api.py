from __future__ import annotations

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from tonari import neighbor_list

numpy_positions: NDArray[np.float64] = np.zeros((2, 3), dtype=np.float64)
numpy_cell: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
numpy_pbc: NDArray[np.bool_] = np.zeros(3, dtype=np.bool_)
assert_type(
    neighbor_list("PS", numpy_positions, numpy_cell, numpy_pbc, 1.0),
    tuple[np.ndarray, ...],
)

torch_positions = torch.zeros((2, 3))
torch_cell = torch.zeros((3, 3))
torch_pbc = torch.zeros(3, dtype=torch.bool)
assert_type(
    neighbor_list("PS", torch_positions, torch_cell, torch_pbc, 1.0),
    tuple[torch.Tensor, ...],
)
