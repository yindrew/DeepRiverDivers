from typing import override

import torch
import torch.nn as nn

from models.model_config import ModelConfig


class CrossAttentionBidirectional(nn.Module):
    # NOTE: the following interpretation of the d_model param might be wrong
    """
    Bidirectional cross attention layer for attending between action and card sequences.
    Actions attend to cards and cards attend to actions.
    Forward input shapes:
        - x_actions: (B, T, d_model), torch.LongTensor
        - x_cards: (B, 7, d_model), torch.LongTensor
        - x_actions_mask: (B, T), torch.BoolTensor
    Forward output shape: (B, 2 * d_model), torch.Tensor
    """

    modelList: nn.ModuleDict

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.modelList = nn.ModuleDict(
            {
                "cross_attn_to_actions": nn.MultiheadAttention(
                    config.cross_attention["d_model"],
                    num_heads=config.cross_attention["num_heads"],
                    batch_first=True,
                ),
                "cross_attn_to_cards": nn.MultiheadAttention(
                    config.cross_attention["d_model"],
                    num_heads=config.cross_attention["num_heads"],
                    batch_first=True,
                ),
            }
        )

    @override
    def forward(
        self,
        x_actions: torch.LongTensor,
        x_cards: torch.LongTensor,
        x_actions_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        # Actions attend to cards
        y_actions: torch.Tensor = self.modelList["cross_attn_to_actions"](
            x_actions, x_cards, x_cards
        )[0]

        # Cards attend to actions
        y_cards: torch.Tensor = self.modelList["cross_attn_to_cards"](
            x_cards, x_actions, x_actions, key_padding_mask=x_actions_mask
        )[0]

        # Pool and concatenate
        y_actions_pooled: torch.Tensor = y_actions.mean(dim=1)  # (B, 32)
        y_cards_pooled: torch.Tensor = y_cards.mean(dim=1)  # (B, 32)
        return torch.cat((y_actions_pooled, y_cards_pooled), dim=-1)  # (B, 64)
