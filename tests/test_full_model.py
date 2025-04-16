import numpy as np
import pytest
import torch

from models.encoded_handhistory import EncodedHandHistory
from models.full_model import FullModel
from schemas.hand_history import Action, Actor, GameAction, HandHistory, Player, Street
from schemas.model_config import ModelConfig


class TestFullModel:
    @pytest.fixture
    def model(self):
        return FullModel()

    @pytest.fixture
    def sample_hand(self):
        """Create a sample hand history for testing."""
        game_log = [
            GameAction(
                action=Action.RAISE,
                amount=2.5,
                player=Player.DEALER,
                street=Street.PREFLOP,
                actor=Actor.HERO,
            ),
            GameAction(
                action=Action.CALL,
                amount=2.5,
                player=Player.BIG_BLIND,
                street=Street.PREFLOP,
                actor=Actor.VILLAIN,
            ),
            GameAction(
                action=Action.CHECK,
                amount=0,
                player=Player.BIG_BLIND,
                street=Street.FLOP,
                actor=Actor.VILLAIN,
            ),
            GameAction(
                action=Action.BET,
                amount=1.0,
                player=Player.DEALER,
                street=Street.FLOP,
                actor=Actor.HERO,
            ),
        ]

        return HandHistory(
            hand=["Ts", "9s"], board=["Jh", "Qh", "Kh"], gameLog=game_log
        )

    def test_model_initialization(self, model):
        """Test that the model initializes correctly with all components."""
        assert isinstance(model, FullModel)
        assert "encoder_actions" in model.moduleDict
        assert "encoder_cards" in model.moduleDict
        assert "cross_attention" in model.moduleDict
        assert "output_mlp" in model.moduleDict

    def test_forward_shape(self, model, sample_hand):
        """Test that the forward pass produces correct output shapes."""
        # Encode the hand
        encoded = EncodedHandHistory.encode_hand_history(sample_hand)

        # Convert to tensors and add batch dimension
        x_actions = torch.LongTensor(encoded["actions"]).unsqueeze(0)  # Add batch dim
        x_cards = torch.LongTensor(encoded["cards"]).unsqueeze(0)  # Add batch dim

        # Forward pass
        output = model(x_actions, x_cards)

        # Check output shape
        assert output.shape == (1,)  # (batch_size,)
        assert isinstance(output, torch.Tensor)

    def test_batch_processing(self, model):
        """Test that the model can process multiple hands in a batch."""
        # Create two different hands
        hand1 = HandHistory(
            hand=["Ah", "Kh"],
            board=["Jh", "Qh", "Kh"],
            gameLog=[
                GameAction(
                    action=Action.BET,
                    amount=3.0,
                    player=Player.UTG,
                    street=Street.PREFLOP,
                    actor=Actor.HERO,
                )
            ],
        )

        hand2 = HandHistory(
            hand=["Qd", "Qc"],
            board=["Qs", "Qh", "2c"],
            gameLog=[
                GameAction(
                    action=Action.CHECK,
                    amount=0.0,
                    player=Player.BIG_BLIND,
                    street=Street.FLOP,
                    actor=Actor.VILLAIN,
                )
            ],
        )

        # Encode both hands
        encoded1 = EncodedHandHistory.encode_hand_history(hand1)
        encoded2 = EncodedHandHistory.encode_hand_history(hand2)

        # Convert to numpy arrays first, then stack, then convert to tensor
        actions_array = np.array([encoded1["actions"], encoded2["actions"]])
        cards_array = np.array([encoded1["cards"], encoded2["cards"]])

        # Convert to tensors
        x_actions = torch.LongTensor(actions_array)
        x_cards = torch.LongTensor(cards_array)

        # Forward pass
        output = model(x_actions, x_cards)

        # Check output shape
        assert output.shape == (2,)  # (batch_size,)
        assert isinstance(output, torch.Tensor)

    def test_model_config(self):
        """Test that the model can be initialized with different configs."""
        config = ModelConfig()

        # Modify some config values
        config.action_encoder["d_model"] = 64
        config.card_encoder["d_model"] = 64
        config.cross_attention["d_model"] = 64

        # Initialize model with config
        model = FullModel()

        # Test forward pass with new config
        sample_hand = HandHistory(
            hand=["Ah", "Kh"],
            board=["Jh", "Qh", "Kh"],
            gameLog=[
                GameAction(
                    action=Action.BET,
                    amount=3.0,
                    player=Player.UTG,
                    street=Street.PREFLOP,
                    actor=Actor.HERO,
                )
            ],
        )

        encoded = EncodedHandHistory.encode_hand_history(sample_hand)
        x_actions = torch.LongTensor(encoded["actions"]).unsqueeze(0)
        x_cards = torch.LongTensor(encoded["cards"]).unsqueeze(0)

        output = model(x_actions, x_cards)
        assert output.shape == (1,)
        assert isinstance(output, torch.Tensor)

    def test_real_hand(self, model):
        """Test the model with a real hand from hand1.json."""
        encoded = EncodedHandHistory.from_json("tests/sample-hand/hand1.json")

        # Convert to tensors and add batch dimension
        x_actions = torch.LongTensor(encoded["actions"]).unsqueeze(0)
        x_cards = torch.LongTensor(encoded["cards"]).unsqueeze(0)

        # Forward pass
        output = model(x_actions, x_cards)

        # Check output
        assert output.shape == (1,)
        assert isinstance(output, torch.Tensor)
        assert not torch.isnan(output).any()  # Check for NaN values
        assert not torch.isinf(output).any()  # Check for Inf values

