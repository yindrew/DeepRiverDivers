from dataclasses import dataclass, field
from typing import TypedDict


class ActionEncoderConfig(TypedDict):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    num_actions: int
    num_positions: int
    num_streets: int
    action_embedding_dim: int
    position_embedding_dim: int
    street_embedding_dim: int


class CardEncoderConfig(TypedDict):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    num_ranks: int
    num_suits: int
    num_streets: int
    rank_embedding_dim: int
    suit_embedding_dim: int
    street_embedding_dim: int


class CrossAttentionConfig(TypedDict):
    num_heads: int
    d_model: int


@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters."""

    # Action Sequence Encoder
    action_encoder: ActionEncoderConfig = field(
        default_factory=lambda: {
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "num_actions": 5,
            "num_positions": 6,
            "num_streets": 5,
            "action_embedding_dim": 4,
            "position_embedding_dim": 4,
            "street_embedding_dim": 4,
        }
    )

    # Card Sequence Encoder
    card_encoder: CardEncoderConfig = field(
        default_factory=lambda: {
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "num_ranks": 13,
            "num_suits": 4,
            "num_streets": 4,
            "rank_embedding_dim": 8,
            "suit_embedding_dim": 4,
            "street_embedding_dim": 4,
        }
    )

    # Cross Attention
    cross_attention: CrossAttentionConfig = field(
        default_factory=lambda: {"num_heads": 4, "d_model": 32}
    )

