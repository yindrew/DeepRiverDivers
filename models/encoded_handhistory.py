from typing import List, Dict
import numpy as np
import json
from models.handhistory import HandHistory, Action, Player, Street, Actor, GameAction

class EncodedHandHistory:
    """
    Functional class for encoding HandHistory objects into neural network input format.
    All methods are static or class methods, so no instantiation is required.
    """

    # Preflop bet size buckets
    
    # Postflop bet size buckets (as percentages of pot)
    POSTFLOP_BET_BUCKETS = [0.25, 0.5, 0.75, 1.0, 1.5, float('inf')]
    
    # Only need mappings for card notation
    RANK_MAP = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
        '9': 7, 'T': 8, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }
    
    SUIT_MAP = {
        's': 0,  # Spade
        'h': 1,  # Heart
        'd': 2,  # Diamond
        'c': 3   # Club
    }
    
    CARD_STREET_MAP = {
        'hole': 0,
        'flop': 1,
        'turn': 2,
        'river': 3
    }

    
    @classmethod
    def encode_hand_history(cls, hand_history: HandHistory) -> Dict[str, np.ndarray]:
        """
        Encode a HandHistory object into the format needed for the neural network.
        
        Args:
            hand_history: A HandHistory object containing the raw hand data
            
        Returns:
            A dictionary containing encoded actions and cards
        """
        encoded_actions = cls._encode_actions(hand_history.gameLog)
        encoded_cards = cls._encode_cards(hand_history.hand, hand_history.board)
        
        return {
            'actions': encoded_actions,
            'cards': encoded_cards
        }
    
    @classmethod
    def encode_batch(cls, hand_histories: List[HandHistory]) -> List[Dict[str, np.ndarray]]:
        """
        Encode a batch of HandHistory objects.
        
        Args:
            hand_histories: A list of HandHistory objects
            
        Returns:
            A list of dictionaries, each containing encoded actions and cards
        """
        return [cls.encode_hand_history(hh) for hh in hand_histories]
    
    @staticmethod
    def _encode_actions(game_log: List[GameAction]) -> np.ndarray:
        """
        Encode the game actions into the format needed for the neural network.
        
        Args:
            game_log: A list of GameAction objects
            
        Returns:
            A numpy array of shape (T, 5) where T is the number of actions
            Each row contains [actor_idx, action_idx, bet_size_bucket_idx, street_idx, position_idx]
        """
        encoded = []
        
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
            
            action_vector = np.array([
                actor_idx,
                action_idx,
                bet_size_bucket_idx,
                street_idx,
                position_idx
            ])
            
            encoded.append(action_vector)
        
        return np.array(encoded)
    
    @staticmethod
    def _encode_cards(hand: List[str], board: List[str]) -> np.ndarray:
        """
        Encode the hole cards and board cards into the format needed for the neural network.
        
        Args:
            hand: A list of strings representing the hole cards
            board: A list of strings representing the board cards
            
        Returns:
            A numpy array of shape (7, 3) where 7 is the total number of cards
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
            street_idx = EncodedHandHistory.CARD_STREET_MAP['hole']
            
            card_encoding = np.array([rank_idx, suit_idx, street_idx])
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
                street_idx = EncodedHandHistory.CARD_STREET_MAP['flop']
            elif i == 3:
                street_idx = EncodedHandHistory.CARD_STREET_MAP['turn']
            else:
                street_idx = EncodedHandHistory.CARD_STREET_MAP['river']
            
            card_encoding = np.array([rank_idx, suit_idx, street_idx])
            encoded_cards.append(card_encoding)
        
        return np.array(encoded_cards)
    
    @classmethod
    def from_json(cls, json_file: str) -> Dict[str, np.ndarray]:
        """
        Create encoded hand history from a JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Returns:
            A dictionary containing encoded actions and cards
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert string values to enum values
        hand = data['hand']
        board = data['board']
        
        game_log = []
        for action_data in data['gameLog']:
            # Convert string values to enum values
            action = Action[action_data['action'].upper()]
            player = Player[action_data['player'].replace(' ', '_').upper()]
            street = Street[action_data['street'].upper()]
            actor = Actor[action_data['actor'].upper()]
            
            game_action = GameAction(
                action=action,
                amount=action_data['amount'],
                player=player,
                street=street,
                actor=actor
            )
            game_log.append(game_action)
        
        hand_history = HandHistory(hand=hand, board=board, gameLog=game_log)
        return cls.encode_hand_history(hand_history) 