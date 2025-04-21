from dataclasses import dataclass, field
from typing import Literal, TypedDict


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


class OutputMLPConfig(TypedDict):
    d_model: int


class TrainingProcessConfig(TypedDict):
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: Literal["Adam", "SGD", "AdamW"]
    momentum: float
    dataset: Literal["GTO", "Human"]
    warm_start: bool
    p_train_test_split: float


class GeneralConfig(TypedDict):
    device: Literal["cpu", "cuda"]
    seed: int
    checkpoint_name: str


@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters."""

    # General Config
    general: GeneralConfig = field(
        default_factory=lambda: {
            "device": "cpu",
            "seed": 42069,
            "checkpoint_name": "default",
        }
    )

    # Training Process Config
    training_process: TrainingProcessConfig = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_epochs": 50,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "momentum": 0,
            "dataset": "GTO",
            "warm_start": False,
            "p_train_test_split": 0.2,
        }
    )

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

    # OutputMLP
    output_mlp: OutputMLPConfig = field(default_factory=lambda: {"d_model": 32})

    def __post_init__(self):
        self.validate_config()

    def validate_config(self):
        # tests for d_model keys - update to reflect intended usage of d_model
        assert (
            self.action_encoder["d_model"] == self.card_encoder["d_model"]
        ), "expect d_model to be the same across action and card configs"
        assert (
            self.action_encoder["d_model"] == self.cross_attention["d_model"]
        ), "expect d_model to be the same across action and cross attention configs"
        assert (
            self.action_encoder["d_model"] == self.output_mlp["d_model"]
        ), "expect d_model to be the same across action and cross attention configs"
