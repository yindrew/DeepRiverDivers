from typing import override

import torch
import torch.nn as nn


class OutputMLP(nn.Module):
    """
    OutputHeadMLP layer predicting EV from concatenated attention outputs
    Forward input shape: (B, 64)
    Forward output shape: (B, )
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 1),
        )

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
