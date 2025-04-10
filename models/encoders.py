from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.custom_nn_modules import CombinedEncoder, DoNothingEncoder, OneHotEncoder
from models.model_config import ModelConfig


class ActionSequenceEncoder(nn.Module):
    """
    Encoder encoding action sequence with self-attention.
    Forward input shape: (B, T, 5)
    Forward output shape: (B, T, 32)
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()

        actor_encoder = DoNothingEncoder() # 1

        action_type_encoder = nn.Embedding(config.action_encoder['num_actions'], 
                                         config.action_encoder['action_embedding_dim']) # 4
        
        bet_size_bucket_encoder = OneHotEncoder(6) # 6

        street_encoder = nn.Embedding(config.action_encoder['num_streets'], 
                                    config.action_encoder['street_embedding_dim']) # 4
        
        position_encoder = nn.Embedding(config.action_encoder['num_positions'], 
                                      config.action_encoder['position_embedding_dim']) # 4
        encoder_list = [
            actor_encoder,
            action_type_encoder,
            bet_size_bucket_encoder,
            street_encoder,
            position_encoder,
        ]
        action_sequence_encoder = CombinedEncoder(encoder_list)
        self.encoder = action_sequence_encoder
        self.fc_layers = nn.Sequential(nn.Linear(19, config.action_encoder['d_model']), 
                                     nn.ReLU(), 
                                     nn.Linear(config.action_encoder['d_model'], 
                                             config.action_encoder['d_model']))
        
        # Add transformer encoder layer for self-attention
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=config.action_encoder['d_model'],
            nhead=config.action_encoder['nhead'],
            dim_feedforward=config.action_encoder['dim_feedforward'],
            dropout=config.action_encoder['dropout'],
            batch_first=True
        )
        
        self.model = nn.Sequential(
            self.encoder,           # Initial encoding
            self.fc_layers,        # Process encoded features
            self.transformer_encoder  # Self-attention
        )

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.model(x)  # (B, T, 32)
        
        return x


class CardSequenceEncoder(nn.Module):
    """
    Encoder encoding card sequence with self-attention.
    Forward input shape: (B, 7, 3)
    Forward output shape: (B, 7, 32)
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        rank_encoder = nn.Embedding(config.card_encoder['num_ranks'], 
                                    config.card_encoder['rank_embedding_dim']) # 8
        
        suit_encoder = nn.Embedding(config.card_encoder['num_suits'],  
                                    config.card_encoder['suit_embedding_dim']) # 4
        
        street_encoder = nn.Embedding(config.card_encoder['num_streets'],  
                                      config.card_encoder['street_embedding_dim']) # 4
        
        encoder_list = [rank_encoder, suit_encoder, street_encoder]
        hand_evaluator_encoder = CombinedEncoder(encoder_list)
        self.encoder = hand_evaluator_encoder
        self.fc_layers = nn.Sequential(nn.Linear(16, 24), 
                                     nn.ReLU(), 
                                     nn.Linear(24, config.card_encoder['d_model']))
        
        # Add transformer encoder layer for self-attention
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=config.card_encoder['d_model'],
            nhead=config.card_encoder['nhead'],
            dim_feedforward=config.card_encoder['dim_feedforward'],
            dropout=config.card_encoder['dropout'],
            batch_first=True
        )
        
        self.model = nn.Sequential(
            self.encoder,           # Initial encoding
            self.fc_layers,        # Process encoded features
            self.transformer_encoder  # Self-attention
        )

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.model(x)  # (B, 7, 32)
        
        return x
