from __future__ import annotations

import pytest
import torch


@pytest.fixture(
    params=[
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is unavailable"
                ),
            ],
        ),
    ]
)
def torch_device(request: pytest.FixtureRequest) -> torch.device:
    return request.param
