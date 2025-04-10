from typing import override

import torch
import torch.nn as nn
from models.model_config import ModelConfig


class CrossAttentionBidirectional(nn.Module):
    """
    Bidirectional cross attention layer for attending between action and card sequences.
    Actions attend to cards and cards attend to actions.
    Forward input shapes: (B, T, 32), (B, 7, 32)
    Forward output shape: (B, 64)
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.modelList: nn.ModuleDict = nn.ModuleDict(
            {
                "cross_attn_to_actions": nn.MultiheadAttention(
                    config.cross_attention['d_model'], 
                    num_heads=config.cross_attention['num_heads'], 
                    batch_first=True
                ),
                "cross_attn_to_cards": nn.MultiheadAttention(
                    config.cross_attention['d_model'], 
                    num_heads=config.cross_attention['num_heads'], 
                    batch_first=True
                ),
            }
        )

    @override
    def forward(self, x_actions: torch.LongTensor, x_cards: torch.LongTensor) -> torch.Tensor:
        # Actions attend to cards
        y_actions = self.modelList["cross_attn_to_actions"](x_actions, x_cards, x_cards)[0]
        
        # Cards attend to actions
        y_cards = self.modelList["cross_attn_to_cards"](x_cards, x_actions, x_actions)[0]
        
        # Pool and concatenate
        y_actions_pooled = y_actions.mean(dim=1)  # (B, 32)
        y_cards_pooled = y_cards.mean(dim=1)      # (B, 32)
        return torch.cat((y_actions_pooled, y_cards_pooled), dim=-1)  # (B, 64)
