import json

import torch

from schemas.hand_history import (
    Action,
    Actor,
    EncodedHandHistoryType,
    GameAction,
    HandHistory,
    Player,
    Street,
)


class EncodedHandHistory:
    """
    Functional class for encoding HandHistory objects into neural network input format.
    All methods are static or class methods, so no instantiation is required.
    """

    # Preflop bet size buckets

    # Postflop bet size buckets (as percentages of pot)
    POSTFLOP_BET_BUCKETS = [0.25, 0.5, 0.75, 1.0, 1.5, float("inf")]

    # Only need mappings for card notation
    RANK_MAP = {
        "2": 0,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 4,
        "7": 5,
        "8": 6,
        "9": 7,
        "T": 8,
        "J": 9,
        "Q": 10,
        "K": 11,
        "A": 12,
    }

    SUIT_MAP = {"s": 0, "h": 1, "d": 2, "c": 3}  # Spade  # Heart  # Diamond  # Club

    CARD_STREET_MAP = {"hole": 0, "flop": 1, "turn": 2, "river": 3}

    @classmethod
    def encode_hand_history(cls, hand_history: HandHistory) -> EncodedHandHistoryType:
        """
        Encode a HandHistory object into the format needed for the neural network.

        Args:
            hand_history: A HandHistory object containing the raw hand data

        Returns:
            A dictionary containing encoded actions and cards
        """
        encoded_actions = cls._encode_actions(hand_history.gameLog)
        encoded_cards = cls._encode_cards(hand_history.hand, hand_history.board)

        return {"actions": encoded_actions, "cards": encoded_cards}

    @classmethod
    def encode_batch(
        cls, hand_histories: list[HandHistory]
    ) -> list[EncodedHandHistoryType]:
        """
        Encode a batch of HandHistory objects.

        Args:
            hand_histories: A list of HandHistory objects

        Returns:
            A list of dictionaries, each containing encoded actions and cards
        """
        return [cls.encode_hand_history(hh) for hh in hand_histories]

    @staticmethod
    def _encode_actions(game_log: list[GameAction]) -> torch.LongTensor:
        """
        Encode the game actions into the format needed for the neural network.

        Args:
            game_log: A list of GameAction objects

        Returns:
            A torch.LongTensor of shape (T, 5) where T is the number of actions
            Each row contains [actor_idx, action_idx, bet_size_bucket_idx, street_idx, position_idx]
        """
        encoded_actions: list[torch.LongTensor] = []

        for action in game_log:
            # Get indices directly from enum values
            actor_idx = action.actor.value
            action_idx = action.action.value
            street_idx = action.street.value
            position_idx = action.player.value

            # For preflop, ignore bet sizing
            if action.street == Street.PREFLOP:
                bet_size_bucket_idx = 0  # Default value for preflop
            else:
                # Postflop uses pot percentage buckets
                bet_size = action.amount
                bet_size_bucket_idx = 0
                for i, threshold in enumerate(EncodedHandHistory.POSTFLOP_BET_BUCKETS):
                    if bet_size <= threshold:
                        bet_size_bucket_idx = i
                        break

            action_vector = torch.LongTensor(
                [actor_idx, action_idx, bet_size_bucket_idx, street_idx, position_idx]
            )

            encoded_actions.append(action_vector)

        if len(encoded_actions) == 0:  # torch stack needs non-empty input
            return torch.LongTensor([])
        else:
            return torch.stack(encoded_actions)

    @staticmethod
    def _encode_cards(hand: list[str], board: list[str]) -> torch.LongTensor:
        """
        Encode the hole cards and board cards into the format needed for the neural network.

        Args:
            hand: A list of strings representing the hole cards
            board: A list of strings representing the board cards

        Returns:
            A torch.LongTensor of shape (7, 3) where 7 is the total number of cards
            Each row contains [rank_idx, suit_idx, street_idx]
        """
        encoded_cards = []

        # Encode hole cards
        for card in hand:
            rank = card[0]
            suit = card[1]

            # Get indices directly
            rank_idx = EncodedHandHistory.RANK_MAP[rank]
            suit_idx = EncodedHandHistory.SUIT_MAP[suit]
            street_idx = EncodedHandHistory.CARD_STREET_MAP["hole"]

            card_encoding = torch.LongTensor([rank_idx, suit_idx, street_idx])
            encoded_cards.append(card_encoding)

        # Encode board cards
        for i, card in enumerate(board):
            rank = card[0]
            suit = card[1]

            # Get indices directly
            rank_idx = EncodedHandHistory.RANK_MAP[rank]
            suit_idx = EncodedHandHistory.SUIT_MAP[suit]

            # Determine street based on position in board
            if i < 3:
                street_idx = EncodedHandHistory.CARD_STREET_MAP["flop"]
            elif i == 3:
                street_idx = EncodedHandHistory.CARD_STREET_MAP["turn"]
            else:
                street_idx = EncodedHandHistory.CARD_STREET_MAP["river"]

            card_encoding = torch.LongTensor([rank_idx, suit_idx, street_idx])
            encoded_cards.append(card_encoding)

        if len(encoded_cards) == 0:  # torch stack needs non-empty input
            return torch.LongTensor([])
        else:
            return torch.stack(encoded_cards)

    @classmethod
    def from_json(cls, json_file: str) -> EncodedHandHistoryType:
        """
        Create encoded hand history from a JSON file.

        Args:
            json_file: Path to the JSON file

        Returns:
            A dictionary containing encoded actions and cards
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        # Convert string values to enum values
        hand = data["hand"]
        board = data["board"]

        game_log = []
        for action_data in data["gameLog"]:
            # Convert string values to enum values
            action = Action[action_data["action"].upper()]
            player = Player[action_data["player"].replace(" ", "_").upper()]
            street = Street[action_data["street"].upper()]
            actor = Actor[action_data["actor"].upper()]

            game_action = GameAction(
                action=action,
                amount=action_data["amount"],
                player=player,
                street=street,
                actor=actor,
            )
            game_log.append(game_action)

        hand_history = HandHistory(hand=hand, board=board, gameLog=game_log)
        return cls.encode_hand_history(hand_history)

    @classmethod
    def decode_to_string(cls, encoded_hand: EncodedHandHistoryType) -> str:
        """
        Convert encoded tensors back to a human-readable string format.

        Args:
            encoded_hand: Dictionary containing encoded actions and cards tensors

        Returns:
            A formatted string showing the decoded hand history
        """
        # Initialize street-based card and action storage
        street_cards = {
            "hole": [],
            "flop": [],
            "turn": [],
            "river": []
        }
        street_actions = {
            "PREFLOP": [],
            "FLOP": [],
            "TURN": [],
            "RIVER": []
        }

        # Decode cards and group by street
        for card in encoded_hand["cards"]:
            rank_idx, suit_idx, street_idx = card.tolist()

            # Convert indices back to human-readable format
            rank_map_reverse = {v: k for k, v in cls.RANK_MAP.items()}
            suit_map_reverse = {v: k for k, v in cls.SUIT_MAP.items()}
            street_map_reverse = {v: k for k, v in cls.CARD_STREET_MAP.items()}

            rank = rank_map_reverse[rank_idx]
            suit = suit_map_reverse[suit_idx]
            street = street_map_reverse[street_idx]

            street_cards[street].append(f"{rank}{suit}")

        # Decode actions and group by street
        mask = encoded_hand.get("mask", None)
        
        for i, action in enumerate(encoded_hand["actions"]):
            # Skip padded actions using mask
            if mask is not None and mask[i]:
                continue
                
            actor_idx, action_idx, bet_size_idx, street_idx, position_idx = (
                action.tolist()
            )

            # Convert indices back to human-readable format
            actor = "Hero" if actor_idx == 0 else "Villain"
            action_map_reverse = {v.value: v.name for v in Action}
            action_name = action_map_reverse[action_idx]
            street_map_reverse = {v.value: v.name for v in Street}
            street = street_map_reverse[street_idx]
            position_map_reverse = {v.value: v.name.replace("_", " ") for v in Player}
            position = position_map_reverse[position_idx]

            # Format bet size
            if street_idx == 0:  # Preflop
                bet_size = "standard sizing"
            else:  # Postflop
                if bet_size_idx < len(cls.POSTFLOP_BET_BUCKETS):
                    bucket = cls.POSTFLOP_BET_BUCKETS[bet_size_idx]
                    bet_size = (
                        f"{bucket*100:.0f}% pot" if bucket != float("inf") else "over 2x pot"
                    )
                else:
                    bet_size = "unknown sizing"

            action_str = f"{position} {action_name} ({bet_size}) as {actor}"
            street_actions[street].append(action_str)

        # Build the output string
        output = []
        
        # Hole cards
        output.append(f"hole cards: {', '.join(street_cards['hole'])}")
        
        # Preflop actions
        if street_actions["PREFLOP"]:
            output.append(f"preflop: ({'; '.join(street_actions['PREFLOP'])})")
        
        # Flop
        flop_str = f"flop: {' '.join(street_cards['flop'])}"
        if street_actions["FLOP"]:
            flop_str += f" | ({'; '.join(street_actions['FLOP'])})"
        output.append(flop_str)
        
        # Turn
        turn_str = f"turn: {', '.join(street_cards['turn'])}"
        if street_actions["TURN"]:
            turn_str += f" | ({'; '.join(street_actions['TURN'])})"
        output.append(turn_str)
        
        # River
        river_str = f"river: {', '.join(street_cards['river'])}"
        if street_actions["RIVER"]:
            river_str += f" | ({'; '.join(street_actions['RIVER'])})"
        output.append(river_str)

        return "\n".join(output)
