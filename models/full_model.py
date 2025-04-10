from typing import override

import torch
import torch.nn as nn

from .cross_attention_bidirectional import CrossAttentionBidirectional
from .encoders import ActionSequenceEncoder, CardSequenceEncoder
from .output_mlp import OutputMLP


class FullModel(nn.Module):
    """
    FullModel combining encoders, cross attention, outputMLP
    Forward input shapes: (B, T, 5), (B, 7, 3) for action and output (torch.LongTensor)
    Forward output shape: (B, ) (torch.Tensor)
    """

    def __init__(self) -> None:
        super().__init__()
        self.moduleDict = nn.ModuleDict(
            {
                "encoder_actions": ActionSequenceEncoder(),
                "encoder_cards": CardSequenceEncoder(),
                "cross_attention": CrossAttentionBidirectional(),
                "output_mlp": OutputMLP(),
            }
        )

    @override
    def forward(
        self,
        x_actions: torch.LongTensor,
        x_cards: torch.LongTensor,
    ) -> torch.Tensor:
        x_enc_actions = self.moduleDict["encoder_actions"](x_actions)
        x_enc_cards = self.moduleDict["encoder_cards"](x_cards)
        x_attended = self.moduleDict["cross_attention"](x_enc_actions, x_enc_cards)
        predicted_ev = self.moduleDict["output_mlp"](x_attended)
        return predicted_ev
