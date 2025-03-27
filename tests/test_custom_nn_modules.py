import torch
import torch.nn as nn

from utils.custom_nn_modules import CombinedEncoder, DoNothingEncoder, OneHotEncoder


class TestCustomNnModules:
    def test_do_nothing_encoder(self):
        x = torch.randperm(40).reshape((2, 4, 5))
        encoder = DoNothingEncoder()
        y: torch.Tensor = encoder(x)
        assert y.shape == (*x.shape, 1)
        assert torch.equal(y.select(-1, 0), x)
        assert y.dtype != torch.int64

    def test_action_sequence_encoder_shape(self):
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
        # 2D tensor test
        x = torch.tensor([[0, 4, 5, 2, 5], [1, 2, 3, 1, 0]])
        y: torch.Tensor = action_sequence_encoder(x)
        assert y.shape == (2, 1 + 4 + 6 + 4 + 4)
        # 3D tensor test
        x = x.unsqueeze(1).repeat(1, 8, 1)
        y: torch.Tensor = action_sequence_encoder(x)
        assert y.shape == (2, 8, 1 + 4 + 6 + 4 + 4)

    def test_hand_evaluator_encoder_shape(self):
        rank_encoder = nn.Embedding(13, 8)
        suit_encoder = nn.Embedding(4, 4)
        street_encoder = nn.Embedding(4, 4)
        encoder_list = [rank_encoder, suit_encoder, street_encoder]
        hand_evaluator_encoder = CombinedEncoder(encoder_list)

        # 2D tensor shape test
        x = torch.tensor([[2, 3, 1], [3, 1, 3]])
        y: torch.Tensor = hand_evaluator_encoder(x)
        assert y.shape == (2, 8 + 4 + 4)
        # 3D tensor shape test
        x = x.unsqueeze(1).repeat(1, 8, 1)
        y: torch.Tensor = hand_evaluator_encoder(x)
        assert y.shape == (2, 8, 8 + 4 + 4)
