from collections import OrderedDict
from typing import TypedDict

import torch

from schemas.model_config import ModelConfig
from utils.training_history import TrainingHistory


class CheckpointDict(TypedDict):
    model_state_dict: OrderedDict[str, torch.Tensor]
    optimizer_state_dict: OrderedDict[str, torch.Tensor]
    loss_history: TrainingHistory
    model_config: ModelConfig
