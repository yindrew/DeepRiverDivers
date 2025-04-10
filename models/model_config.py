from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters."""
    
    # Action Sequence Encoder
    action_encoder: Dict[str, int] = field(default_factory=lambda: {
        'd_model': 32,
        'nhead': 4,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'num_actions': 5,
        'num_positions': 6,
        'num_streets': 5,
        'action_embedding_dim': 4,
        'position_embedding_dim': 4,
        'street_embedding_dim': 4
    })
    
    # Card Sequence Encoder
    card_encoder: Dict[str, int] = field(default_factory=lambda: {
        'd_model': 32,
        'nhead': 4,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'num_ranks': 13,
        'num_suits': 4,
        'num_streets': 4,
        'rank_embedding_dim': 8,
        'suit_embedding_dim': 4,
        'street_embedding_dim': 4
    })
    
    # Cross Attention
    cross_attention: Dict[str, int] = field(default_factory=lambda: {
        'num_heads': 4,
        'd_model': 32
    })    