from pathlib import Path

import numpy as np
import pytest

from models.encoded_handhistory import EncodedHandHistory
from models.handhistory import Action, Actor, GameAction, HandHistory, Player, Street


class TestEncodedHandHistory:
    def test_encode_actions_preflop(self):
        # Create a preflop action sequence
        game_log = [
            GameAction(
                action=Action.BET,  # RFI
                amount=3.0,
                player=Player.UTG,
                street=Street.PREFLOP,
                actor=Actor.HERO,
            ),
            GameAction(
                action=Action.CALL,
                amount=3.0,
                player=Player.BIG_BLIND,
                street=Street.PREFLOP,
                actor=Actor.VILLAIN,
            ),
        ]

        hand_history = HandHistory(hand=["Ah", "Kh"], board=[], gameLog=game_log)

        encoded = EncodedHandHistory.encode_hand_history(hand_history)

        # Check action encoding
        actions = encoded["actions"]
        assert actions.shape == (2, 5)  # 2 actions, 5 features each

        # First action (RFI)
        assert actions[0][0] == Actor.HERO.value  # actor
        assert actions[0][1] == Action.BET.value  # action
        assert actions[0][2] == 0  # bet size bucket (ignored for preflop)
        assert actions[0][3] == Street.PREFLOP.value  # street
        assert actions[0][4] == Player.UTG.value  # position

        # Second action (Call)
        assert actions[1][0] == Actor.VILLAIN.value
        assert actions[1][1] == Action.CALL.value
        assert actions[1][2] == 0  # bet size bucket (ignored for preflop)
        assert actions[1][3] == Street.PREFLOP.value
        assert actions[1][4] == Player.BIG_BLIND.value

    def test_encode_actions_postflop(self):
        # Create a postflop action sequence
        game_log = [
            GameAction(
                action=Action.BET,
                amount=0.5,  # 50% of pot
                player=Player.UTG,  # Fixed: Using position instead of HERO
                street=Street.FLOP,
                actor=Actor.HERO,
            ),
            GameAction(
                action=Action.RAISE,
                amount=1.5,  # 150% of pot
                player=Player.BIG_BLIND,  # Fixed: Using position instead of VILLAIN
                street=Street.FLOP,
                actor=Actor.VILLAIN,
            ),
        ]

        hand_history = HandHistory(
            hand=["Ah", "Kh"], board=["Jh", "Th", "2c"], gameLog=game_log
        )

        encoded = EncodedHandHistory.encode_hand_history(hand_history)

        # Check action encoding
        actions = encoded["actions"]
        assert actions.shape == (2, 5)

        # First action (50% pot bet)
        assert actions[0][0] == Actor.HERO.value
        assert actions[0][1] == Action.BET.value
        assert actions[0][2] == 1  # bet size bucket (0.5 pot = bucket 1)
        assert actions[0][3] == Street.FLOP.value
        assert actions[0][4] == Player.UTG.value  # Fixed: Using position value

        # Second action (150% pot raise)
        assert actions[1][0] == Actor.VILLAIN.value
        assert actions[1][1] == Action.RAISE.value
        assert actions[1][2] == 4  # bet size bucket (1.5 pot = bucket 4)
        assert actions[1][3] == Street.FLOP.value
        assert actions[1][4] == Player.BIG_BLIND.value  # Fixed: Using position value

    def test_encode_cards(self):
        hand_history = HandHistory(
            hand=["Ah", "Kh"],  # Ace and King of hearts
            board=["Jh", "Th", "2c", "Qd", "As"],  # Flush draw board
            gameLog=[],
        )

        encoded = EncodedHandHistory.encode_hand_history(hand_history)
        cards = encoded["cards"]

        # Check shape (7 cards total: 2 hole + 5 board)
        assert cards.shape == (7, 3)  # 7 cards, 3 features each (rank, suit, street)

        # Check hole cards
        # Ah
        assert cards[0][0] == EncodedHandHistory.RANK_MAP["A"]  # rank
        assert cards[0][1] == EncodedHandHistory.SUIT_MAP["h"]  # suit
        assert cards[0][2] == EncodedHandHistory.CARD_STREET_MAP["hole"]  # street

        # Kh
        assert cards[1][0] == EncodedHandHistory.RANK_MAP["K"]
        assert cards[1][1] == EncodedHandHistory.SUIT_MAP["h"]
        assert cards[1][2] == EncodedHandHistory.CARD_STREET_MAP["hole"]

        # Check board cards
        # Jh (flop)
        assert cards[2][0] == EncodedHandHistory.RANK_MAP["J"]
        assert cards[2][1] == EncodedHandHistory.SUIT_MAP["h"]
        assert cards[2][2] == EncodedHandHistory.CARD_STREET_MAP["flop"]

        # Th (flop)
        assert cards[3][0] == EncodedHandHistory.RANK_MAP["T"]
        assert cards[3][1] == EncodedHandHistory.SUIT_MAP["h"]
        assert cards[3][2] == EncodedHandHistory.CARD_STREET_MAP["flop"]

        # 2c (flop)
        assert cards[4][0] == EncodedHandHistory.RANK_MAP["2"]
        assert cards[4][1] == EncodedHandHistory.SUIT_MAP["c"]
        assert cards[4][2] == EncodedHandHistory.CARD_STREET_MAP["flop"]

        # Qd (turn)
        assert cards[5][0] == EncodedHandHistory.RANK_MAP["Q"]
        assert cards[5][1] == EncodedHandHistory.SUIT_MAP["d"]
        assert cards[5][2] == EncodedHandHistory.CARD_STREET_MAP["turn"]

        # As (river)
        assert cards[6][0] == EncodedHandHistory.RANK_MAP["A"]
        assert cards[6][1] == EncodedHandHistory.SUIT_MAP["s"]
        assert cards[6][2] == EncodedHandHistory.CARD_STREET_MAP["river"]

    def test_from_json(self):
        # Create a test JSON file
        import json
        import tempfile

        test_data = {
            "hand": ["Ah", "Kh"],
            "board": ["Jh", "Th", "2c"],
            "gameLog": [
                {
                    "action": "BET",
                    "amount": 3.0,
                    "player": "UTG",
                    "street": "PREFLOP",
                    "actor": "HERO",
                },
                {
                    "action": "CALL",
                    "amount": 3.0,
                    "player": "BIG_BLIND",
                    "street": "PREFLOP",
                    "actor": "VILLAIN",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            encoded = EncodedHandHistory.from_json(temp_path)

            # Check action encoding
            actions = encoded["actions"]
            assert actions.shape == (2, 5)

            # First action (RFI)
            assert actions[0][0] == Actor.HERO.value
            assert actions[0][1] == Action.BET.value
            assert actions[0][2] == 0  # bet size bucket (ignored for preflop)
            assert actions[0][3] == Street.PREFLOP.value
            assert actions[0][4] == Player.UTG.value

            # Check card encoding
            cards = encoded["cards"]
            assert cards.shape == (5, 3)  # 2 hole + 3 board cards

            # Check hole cards
            assert cards[0][0] == EncodedHandHistory.RANK_MAP["A"]
            assert cards[0][1] == EncodedHandHistory.SUIT_MAP["h"]
            assert cards[0][2] == EncodedHandHistory.CARD_STREET_MAP["hole"]

        finally:
            # Clean up temporary file
            import os

            os.unlink(temp_path)

    def test_encode_batch(self):
        # Create multiple hand histories
        hand_histories = [
            HandHistory(
                hand=["Ah", "Kh"],
                board=["Jh", "Th", "2c"],
                gameLog=[
                    GameAction(
                        action=Action.BET,
                        amount=3.0,
                        player=Player.UTG,
                        street=Street.PREFLOP,
                        actor=Actor.HERO,
                    )
                ],
            ),
            HandHistory(
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
            ),
        ]

        encoded_batch = EncodedHandHistory.encode_batch(hand_histories)

        # Check batch encoding
        assert len(encoded_batch) == 2

        # Check first hand
        first_hand = encoded_batch[0]
        assert first_hand["actions"].shape == (1, 5)  # 1 action
        assert first_hand["cards"].shape == (5, 3)  # 2 hole + 3 board

        # Check second hand
        second_hand = encoded_batch[1]
        assert second_hand["actions"].shape == (1, 5)  # 1 action
        assert second_hand["cards"].shape == (5, 3)  # 2 hole + 3 board

    def test_sample_hand(self):
        """Test encoding of the sample hand from hand1.json"""
        test_dir = Path(__file__).parent
        sample_json_file = str((test_dir / "sample-hand") / "hand1.json")
        encoded = EncodedHandHistory.from_json(sample_json_file)

        # Check action encoding
        actions = encoded["actions"]
        assert actions.shape == (10, 5)  # 10 actions in the hand

        # Check first action (preflop raise)
        assert actions[0][0] == Actor.HERO.value
        assert actions[0][1] == Action.RAISE.value
        assert actions[0][2] == 0  # bet size bucket (ignored for preflop)
        assert actions[0][3] == Street.PREFLOP.value
        assert actions[0][4] == Player.DEALER.value

        # Check second action (preflop call)
        assert actions[1][0] == Actor.VILLAIN.value
        assert actions[1][1] == Action.CALL.value
        assert actions[1][2] == 0  # bet size bucket (ignored for preflop)
        assert actions[1][3] == Street.PREFLOP.value
        assert actions[1][4] == Player.BIG_BLIND.value

        # Check flop actions
        assert actions[2][0] == Actor.VILLAIN.value  # Check
        assert actions[2][1] == Action.CHECK.value
        assert actions[2][3] == Street.FLOP.value

        assert actions[3][0] == Actor.HERO.value  # Bet
        assert actions[3][1] == Action.BET.value
        assert actions[3][2] == 3  # bet size bucket (1.0 pot = bucket 3)
        assert actions[3][3] == Street.FLOP.value

        assert actions[4][0] == Actor.VILLAIN.value  # Call
        assert actions[4][1] == Action.CALL.value
        assert actions[4][2] == 3  # bet size bucket (1.0 pot = bucket 3)
        assert actions[4][3] == Street.FLOP.value

        # Check turn actions
        assert actions[5][0] == Actor.VILLAIN.value  # Check
        assert actions[5][1] == Action.CHECK.value
        assert actions[5][3] == Street.TURN.value

        assert actions[6][0] == Actor.HERO.value  # Bet
        assert actions[6][1] == Action.BET.value
        assert actions[6][2] == 3  # bet size bucket (1.0 pot = bucket 3)
        assert actions[6][3] == Street.TURN.value

        assert actions[7][0] == Actor.VILLAIN.value  # Raise
        assert actions[7][1] == Action.RAISE.value
        assert actions[7][2] == 5  # bet size bucket (3.0 pot = bucket 5, >1.5 pot)
        assert actions[7][3] == Street.TURN.value

        assert actions[8][0] == Actor.HERO.value  # Call
        assert actions[8][1] == Action.CALL.value
        assert actions[8][2] == 5  # bet size bucket (3.0 pot = bucket 5, >1.5 pot)
        assert actions[8][3] == Street.TURN.value

        # Check river action
        assert actions[9][0] == Actor.VILLAIN.value  # Bet
        assert actions[9][1] == Action.BET.value
        assert (
            actions[9][2] == 4
        )  # bet size bucket (1.25 pot = bucket 4, between 1.0-1.5 pot)
        assert actions[9][3] == Street.RIVER.value

        # Check card encoding
        cards = encoded["cards"]
        assert cards.shape == (7, 3)  # 2 hole + 5 board cards

        # Check hole cards
        assert cards[0][0] == EncodedHandHistory.RANK_MAP["T"]  # Ts
        assert cards[0][1] == EncodedHandHistory.SUIT_MAP["s"]
        assert cards[0][2] == EncodedHandHistory.CARD_STREET_MAP["hole"]

        assert cards[1][0] == EncodedHandHistory.RANK_MAP["9"]  # 9s
        assert cards[1][1] == EncodedHandHistory.SUIT_MAP["s"]
        assert cards[1][2] == EncodedHandHistory.CARD_STREET_MAP["hole"]

        # Check board cards
        assert cards[2][0] == EncodedHandHistory.RANK_MAP["J"]  # Jh
        assert cards[2][1] == EncodedHandHistory.SUIT_MAP["h"]
        assert cards[2][2] == EncodedHandHistory.CARD_STREET_MAP["flop"]

        assert cards[3][0] == EncodedHandHistory.RANK_MAP["Q"]  # Qh
        assert cards[3][1] == EncodedHandHistory.SUIT_MAP["h"]
        assert cards[3][2] == EncodedHandHistory.CARD_STREET_MAP["flop"]

        assert cards[4][0] == EncodedHandHistory.RANK_MAP["K"]  # Kh
        assert cards[4][1] == EncodedHandHistory.SUIT_MAP["h"]
        assert cards[4][2] == EncodedHandHistory.CARD_STREET_MAP["flop"]

        assert cards[5][0] == EncodedHandHistory.RANK_MAP["A"]  # Ah
        assert cards[5][1] == EncodedHandHistory.SUIT_MAP["h"]
        assert cards[5][2] == EncodedHandHistory.CARD_STREET_MAP["turn"]

        assert cards[6][0] == EncodedHandHistory.RANK_MAP["2"]  # 2s
        assert cards[6][1] == EncodedHandHistory.SUIT_MAP["s"]
        assert cards[6][2] == EncodedHandHistory.CARD_STREET_MAP["river"]

