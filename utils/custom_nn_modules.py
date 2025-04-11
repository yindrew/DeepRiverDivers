from collections.abc import Sequence
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoNothingEncoder(nn.Module):
    """
    Encoder that does nothing except converting LongTensor to Tensor
    and adds a dummy dimension.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return x.unsqueeze(-1).float()


class OneHotEncoder(nn.Module):
    """
    F.one_hot written in the form of an nn.Module
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes: int = num_classes

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Input:
            x: LongTensor of shape (*)
        Output:
            x: Tensor of shape (*, self.num_classes)
        """
        return F.one_hot(x, num_classes=self.num_classes).float()


class CombinedEncoder(nn.Module):
    """
    Encoder used for combining multiple base encoders
    and apply to each slice of tensor.
    """

    def __init__(self, encoder_list: Sequence[nn.Module]) -> None:
        super().__init__()
        self.encoder_module_list: nn.ModuleList = nn.ModuleList(
            [encoder for encoder in encoder_list]
        )

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Apply ith encoder to x.select(-1, i) and concat.
        """
        if x.shape[-1] != len(self.encoder_module_list):
            raise ValueError(
                "Dimension mismatch: Expect x.shape[-1] match len(self.encoder_module_list)"
            )
        return torch.cat(
            [
                encoder(x.select(-1, i))
                for (i, encoder) in enumerate(self.encoder_module_list)
            ],
            dim=-1,
        )
