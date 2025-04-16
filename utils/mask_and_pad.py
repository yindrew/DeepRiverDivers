from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class MaskerPadder:
    # NOTE: Possible improvement: modify to only need to take a list of ints
    """
    Utility function for creating masks and padded tensor from a given list of LongTensors.
    Intended to be called on LongTensors produced by list of actions tensors encoded_hand_history.

    Args:
        validate_input: use True in testing, validates shapes
    Input:
        x: length B list of long tensors of shape (E, ) to create mask and padded versions of x
           (where E = action embedding dim)
    Returns:
        tuple[x_mask, x_padded]
        where:
            x_mask: torch.BoolTensor of shape (B, T)
            x_padded: torch.BoolTensor of shape (B, T, E)
    """

    validate_input: bool = False

    def forward(
        self, x: Sequence[torch.LongTensor]
    ) -> tuple[torch.BoolTensor, torch.LongTensor]:
        if self.validate_input:
            self.validate(x)
        x_lens = list(map(len, x))
        T = max(x_lens)  # same as T
        B = len(x)  # same as B
        x_mask = torch.arange(T)[None, :] >= torch.tensor(x_lens)[:, None]
        x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        return (x_mask, x_padded)

    def __call__(self, x: Sequence[torch.LongTensor]) -> torch.BoolTensor:
        return self.forward(x)

    @staticmethod
    def validate(x: Sequence[torch.LongTensor]):
        assert set([len(xx.shape) for xx in x]) == {2}
        assert len(set([xx.shape[-1] for xx in x])) == 1
