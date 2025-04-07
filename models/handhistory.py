from dataclasses import dataclass
from typing import List, Dict, Any
import json
from enum import Enum
import numpy as np

class Actor(Enum):
    HERO = "hero"
    VILLAIN = "villain"

class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

class Player(Enum):
    UTG = "UTG"
    UTG_PLUS_1 = "UTG+1"
    UTG_PLUS_2 = "UTG+2"
    DEALER = "Dealer"
    SMALL_BLIND = "Small Blind"
    BIG_BLIND = "Big Blind"

class Action(Enum):
    FOLD = "Fold"
    RAISE = "Raise"
    BET = "Bet"
    CALL = "Call"
    CHECK = "Check"

@dataclass
class GameAction:
    action: Action       
    amount: float      
    player: Player       
    street: Street        
    actor: Actor         

@dataclass
class HandHistory:
    hand: List[str]                
    board: List[str]              
    gameLog: List[GameAction]     

class EncodedHandHistory:
    """
    Functional class for encoding HandHistory objects into neural network input format.
    All methods are static or class methods, so no instantiation is required.
    """
    
    # Bet size buckets (as percentages of pot)
    BET_SIZE_BUCKETS = [0.2, 0.5, 0.8, 1.0, 2.0, float('inf')]
    
    # Mapping dictionaries
    RANK_MAP = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
        'T': 8, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }
    
    SUIT_MAP = {
        's': 0,  # Spade
        'h': 1,  # Heart
        'd': 2,  # Diamond
        'c': 3   # Club
    }
    
    PLAYER_MAP = {
        Player.UTG: 0,
        Player.UTG_PLUS_1: 1,
        Player.UTG_PLUS_2: 2,
        Player.DEALER: 3,
        Player.SMALL_BLIND: 4,
        Player.BIG_BLIND: 5
    }
    
    ACTION_MAP = {
        Action.FOLD: 0,
        Action.CHECK: 1,
        Action.CALL: 2,
        Action.RAISE: 3,
        Action.BET: 4
    }
    
    STREET_MAP = {
        Street.PREFLOP: 0,
        Street.FLOP: 1,
        Street.TURN: 2,
        Street.RIVER: 3
    }
    
    CARD_STREET_MAP = {
        'hole': 0,
        'flop': 1,
        'turn': 2,
        'river': 3
    }

    # Action encoding dimensions
    ACTION_DIM = 22 
    
    # Card encoding dimensions
    RANK_DIM = len(RANK_MAP)    # Rank encoding dimension (2-10, J, Q, K, A)
    SUIT_DIM = len(SUIT_MAP)    # Suit encoding dimension
    CARD_STREET_DIM = len(CARD_STREET_MAP)  # Street encoding dimension for cards
    
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
            A numpy array of shape (T, 22) where T is the number of actions
        """
        encoded = []
        
        for action in game_log:
            # Actor: binary flag (1 for hero, 0 for villain)
            actor_encoding = 1 if action.actor == Actor.HERO else 0
            
            # Action Type: 5-way one-hot vector
            action_encoding = np.zeros(5)
            action_idx = EncodedHandHistory.ACTION_MAP[action.action]
            action_encoding[action_idx] = 1
            
            # Bet Size Bucket: 6-dimensional one-hot vector
            bet_size_encoding = np.zeros(6)
            bet_size = action.amount
            for i, threshold in enumerate(EncodedHandHistory.BET_SIZE_BUCKETS):
                if bet_size <= threshold:
                    bet_size_encoding[i] = 1
                    break
            
            # Street: 4-way one-hot vector
            street_encoding = np.zeros(4)
            street_idx = EncodedHandHistory.STREET_MAP[action.street]
            street_encoding[street_idx] = 1
            
            # Position: 6-way one-hot vector
            position_encoding = np.zeros(6)
            position_idx = EncodedHandHistory.PLAYER_MAP[action.player]
            position_encoding[position_idx] = 1
            
            # Concatenate all encodings into a 22-dimensional vector
            action_vector = np.concatenate([
                [actor_encoding],
                action_encoding,
                bet_size_encoding,
                street_encoding,
                position_encoding
            ])
            
            encoded.append(action_vector)
        
        return np.array(encoded)
    
    @staticmethod
    def _encode_cards(hand: List[str], board: List[str]) -> Dict[str, np.ndarray]:
        """
        Encode the hole cards and board cards into the format needed for the neural network.
        
        Args:
            hand: A list of strings representing the hole cards
            board: A list of strings representing the board cards
            
        Returns:
            A dictionary containing encoded hole cards and board cards
        """
        # Encode hole cards
        hole_cards = []
        for card in hand:
            # Handle multi-character ranks like '10'
            if len(card) > 2:
                rank = card[:-1]  # Everything except the last character (suit)
                suit = card[-1]   # Last character (suit)
            else:
                rank = card[0]
                suit = card[1]
                
            rank_encoding = np.zeros(EncodedHandHistory.RANK_DIM)
            rank_idx = EncodedHandHistory.RANK_MAP[rank]
            rank_encoding[rank_idx] = 1
            
            suit_encoding = np.zeros(EncodedHandHistory.SUIT_DIM)
            suit_idx = EncodedHandHistory.SUIT_MAP[suit]
            suit_encoding[suit_idx] = 1
            
            street_encoding = np.zeros(EncodedHandHistory.CARD_STREET_DIM)
            street_encoding[EncodedHandHistory.CARD_STREET_MAP['hole']] = 1
            
            # Add a dimension to indicate this is a hole card (1)
            is_hole_card = 1
            
            card_encoding = np.concatenate([rank_encoding, suit_encoding, street_encoding, [is_hole_card]])
            hole_cards.append(card_encoding)
        
        # Encode board cards
        board_cards = []
        for i, card in enumerate(board):
            # Handle multi-character ranks like '10'
            if len(card) > 2:
                rank = card[:-1]  # Everything except the last character (suit)
                suit = card[-1]   # Last character (suit)
            else:
                rank = card[0]
                suit = card[1]
                
            rank_encoding = np.zeros(EncodedHandHistory.RANK_DIM)
            rank_idx = EncodedHandHistory.RANK_MAP[rank]
            rank_encoding[rank_idx] = 1
            
            suit_encoding = np.zeros(EncodedHandHistory.SUIT_DIM)
            suit_idx = EncodedHandHistory.SUIT_MAP[suit]
            suit_encoding[suit_idx] = 1
            
            street_encoding = np.zeros(EncodedHandHistory.CARD_STREET_DIM)
            # Determine which street this card belongs to
            if i < 3:
                street_encoding[EncodedHandHistory.CARD_STREET_MAP['flop']] = 1
            elif i == 3:
                street_encoding[EncodedHandHistory.CARD_STREET_MAP['turn']] = 1
            else:
                street_encoding[EncodedHandHistory.CARD_STREET_MAP['river']] = 1
            
            # Add a dimension to indicate this is a board card (0)
            is_hole_card = 0
            
            card_encoding = np.concatenate([rank_encoding, suit_encoding, street_encoding, [is_hole_card]])
            board_cards.append(card_encoding)
        
        return {
            'hole_cards': np.array(hole_cards),
            'board_cards': np.array(board_cards)
        }
    
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