import torch

from utils.create_mask_and_pad import MaskerPadder


class TestCreateMask:
    def test_masker(self):
        masker_padder = MaskerPadder(validate_input=True)
        for _ in range(3):
            perm = torch.randperm(4)
            x = [torch.ones((i, 10)).long() for i in range(1, 5)]
            true_mask = torch.ones(4, 4).triu(diagonal=1).bool()[perm, :]

            x_mask, x_padded = masker_padder([x[i] for i in perm])
            assert x_mask.shape == true_mask.shape
            assert x_mask.dtype == true_mask.dtype
            assert torch.equal(x_mask, true_mask)
