from typing import override

import torch
import torch.nn as nn

from schemas.model_config import ModelConfig

from .cross_attention_bidirectional import CrossAttentionBidirectional
from .encoders import ActionSequenceEncoder, CardSequenceEncoder
from .output_mlp import OutputMLP


class FullModel(nn.Module):
    """
    FullModel combining encoders, cross attention, outputMLP
    Forward input shapes:
        - (B, T, 5) for action (torch.LongTensor), where T is max seq length
        - (B, 7, 3) for output (torch.LongTensor)
        - (B, T) for output (torch.LongTensor)
    Forward output shape: (B, ) (torch.Tensor)
    """

    moduleDict: nn.ModuleDict

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.moduleDict = nn.ModuleDict(
            {
                "encoder_actions": ActionSequenceEncoder(config=config),
                "encoder_cards": CardSequenceEncoder(config=config),
                "cross_attention": CrossAttentionBidirectional(config=config),
                "output_mlp": OutputMLP(),
            }
        )

    @override
    def forward(
        self,
        x_actions: torch.LongTensor,
        x_cards: torch.LongTensor,
        x_actions_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        x_enc_actions: torch.Tensor = self.moduleDict["encoder_actions"](
            x_actions, x_actions_mask
        )
        x_enc_cards: torch.Tensor = self.moduleDict["encoder_cards"](x_cards)
        x_attended: torch.Tensor = self.moduleDict["cross_attention"](
            x_enc_actions, x_enc_cards, x_actions_mask
        )
        predicted_ev: torch.Tensor = self.moduleDict["output_mlp"](x_attended)
        return predicted_ev
