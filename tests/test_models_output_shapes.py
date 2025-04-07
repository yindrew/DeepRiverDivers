import torch

from models.cross_attention_bidirectional import CrossAttentionBidirectional
from models.encoders import ActionSequenceEncoder, CardSequenceEncoder
from models.full_model import FullModel
from models.output_mlp import OutputMLP

B = 32
T = 17


class TestModelsOutputShapes:
    def test_shape_action_sequence_encoder(self):
        x = torch.ones(B, T, 5).long()
        model = ActionSequenceEncoder()
        y = model(x)
        assert y.dtype != torch.int64
        assert y.shape == (B, T, 32)

    def test_shape_action_sequence_encoder(self):
        x = torch.ones(B, 7, 3).long()
        model = CardSequenceEncoder()
        y = model(x)
        assert y.dtype != torch.int64
        assert y.shape == (B, 7, 32)

    def test_shape_cross_attention_bidirectional(self):
        x_actions = torch.randn((B, T, 32))
        x_cards = torch.randn((B, 7, 32))
        model = CrossAttentionBidirectional()
        y = model(x_actions, x_cards)
        assert y.shape == (B, 64)

    def test_shape_output_mlp(self):
        x = torch.randn((B, 64))
        model = OutputMLP()
        y = model(x)
        assert y.shape == (B,)

    def test_shape_full_model(self):
        x_actions = torch.ones(B, T, 5).long()
        x_cards = torch.ones(B, 7, 3).long()
        model = FullModel()
        y = model(x_actions, x_cards)
        assert y.shape == (B,)
