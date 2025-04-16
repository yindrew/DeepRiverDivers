from typing import TypedDict

import torch

from schemas.hand_history import EncodedHandHistoryType


class DatasetBaseType(TypedDict):
    """
    Common base datatype for passing into torch's datasets and dataloaders
    """

    encoded_hand_history: EncodedHandHistoryType
    expected_ev: float


class DatasetBaseCollatedType(TypedDict):
    """
    Common base datatype after collating
    """

    expected_ev: torch.Tensor
    cards: torch.LongTensor
    actions_padded: torch.LongTensor
    actions_mask: torch.BoolTensor
