from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.custom_nn_modules import CombinedEncoder, DoNothingEncoder, OneHotEncoder


class ActionSequenceEncoder(nn.Module):
    """
    Encoder encoding action sequence.
    Forward input shape: (B, T, 5)
    Forward output shape: (B, T, 32)
    """

    def __init__(self) -> None:
        super().__init__()
        actor_encoder = DoNothingEncoder()
        action_type_encoder = nn.Embedding(5, 4)
        bet_size_bucket_encoder = OneHotEncoder(6)
        street_encoder = nn.Embedding(5, 4)
        position_encoder = nn.Embedding(6, 4)
        encoder_list = [
            actor_encoder,
            action_type_encoder,
            bet_size_bucket_encoder,
            street_encoder,
            position_encoder,
        ]
        action_sequence_encoder = CombinedEncoder(encoder_list)
        self.encoder = action_sequence_encoder
        self.fc_layers = nn.Sequential(nn.Linear(19, 32), nn.ReLU(), nn.Linear(32, 32))
        self.model = nn.Sequential(self.encoder, self.fc_layers)

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.model(x)


class CardSequenceEncoder(nn.Module):
    """
    Encoder encoding action sequence.
    Forward input shape: (B, 7, 3)
    Forward output shape: (B, 7, 32)
    """

    def __init__(self) -> None:
        super().__init__()
        rank_encoder = nn.Embedding(13, 8)
        suit_encoder = nn.Embedding(4, 4)
        street_encoder = nn.Embedding(4, 4)
        encoder_list = [rank_encoder, suit_encoder, street_encoder]
        hand_evaluator_encoder = CombinedEncoder(encoder_list)
        self.encoder = hand_evaluator_encoder
        self.fc_layers = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 32))
        self.model = nn.Sequential(self.encoder, self.fc_layers)

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.model(x)
