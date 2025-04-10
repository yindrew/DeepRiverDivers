from typing import override

import torch
import torch.nn as nn


class CrossAttentionBidirectional(nn.Module):
    """
    Bidirectional cross attention layer for attending both ways the outputs from the encoders
    Forward input shapes: (B, T, 32), (B, 7, 32)
    Forward output shape: (B, T, 64)
    """

    def __init__(self, num_heads=1) -> None:
        super().__init__()
        self.modelList: nn.ModuleDict = nn.ModuleDict(
            {
                "self_attn_to_actions": nn.MultiheadAttention(
                    32, num_heads=num_heads, batch_first=True
                ),
                "self_attn_to_cards": nn.MultiheadAttention(
                    32, num_heads=num_heads, batch_first=True
                ),
                "cross_attn_to_actions": nn.MultiheadAttention(
                    32, num_heads=num_heads, batch_first=True
                ),
                "cross_attn_to_cards": nn.MultiheadAttention(
                    32, num_heads=num_heads, batch_first=True
                ),
            }
        )

    @override
    def forward(
        self,
        x_actions: torch.LongTensor,
        x_cards: torch.LongTensor,
    ) -> torch.Tensor:
        (x_actions_self_attn, x_cards_self_attn) = self._call_self_attn(
            x_actions, x_cards
        )
        (y_actions, y_cards) = self._call_self_attn(
            x_actions_self_attn, x_cards_self_attn
        )
        y_actions_pooled = y_actions.mean(dim=1)
        y_cards_pooled = y_cards.mean(dim=1)
        return torch.cat((y_actions_pooled, y_cards_pooled), dim=-1)

    def _call_self_attn(self, x_actions, x_cards):
        return (
            self.modelList["self_attn_to_actions"](x_actions, x_actions, x_actions)[0],
            self.modelList["self_attn_to_cards"](x_cards, x_cards, x_cards)[0],
        )

    def _call_cross_attn(self, x_actions, x_cards):
        return (
            self.modelList["cross_attn_to_actions"](x_actions, x_cards, x_cards)[0],
            self.modelList["cross_attn_to_cards"](x_cards, x_actions, x_actions)[0],
        )
