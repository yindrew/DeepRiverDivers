from typing import override

import torch
import torch.nn as nn

from models.model_config import ModelConfig


class OutputMLP(nn.Module):
    """
    OutputHeadMLP layer predicting EV from concatenated attention outputs
    Forward input shape: (B, 2 * d_model)
    Forward output shape: (B, )
    """

    model: nn.Module

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        # default: input 64 -> 64 -> 32 -> 1
        self.model = nn.Sequential(
            nn.Linear(
                2 * config.output_mlp["d_model"], 2 * config.output_mlp["d_model"]
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2 * config.output_mlp["d_model"], config.output_mlp["d_model"]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(config.output_mlp["d_model"], 1),
        )

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
