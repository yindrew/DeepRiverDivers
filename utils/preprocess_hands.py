import json
from pathlib import Path
from models.encoded_handhistory import EncodedHandHistory
from schemas.hand_history import Action, Actor, GameAction, HandHistory, Player, Street

def sort_cards(cards: list[str]) -> list[str]:
    """Sort cards from highest to lowest rank, with suits as tiebreaker."""
    # Define rank order (Ace high)
    rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, 
                  '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    # Define suit order (spades, hearts, diamonds, clubs)
    suit_order = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
    
    return sorted(cards, 
                 key=lambda x: (-rank_order[x[0]], suit_order[x[1]]), 
                 reverse=False)

def normalize_hand(hand_history: HandHistory) -> HandHistory:
    """Normalize a hand by sorting cards in canonical order."""
    # Sort hole cards
    sorted_hand = sort_cards(hand_history.hand)
    
    # Sort flop cards
    sorted_board = sort_cards(hand_history.board[:3]) + hand_history.board[3:]
    
    return HandHistory(
        hand=sorted_hand,
        board=sorted_board,
        gameLog=hand_history.gameLog
    )

def preprocess_hands():
    # Setup paths
    root_dir = Path(__file__).parent.parent  # Go up one level to project root
    input_path = root_dir / "data" / "human" / "hands.json"
    output_path = root_dir / "data" / "human" / "hands_processed.json"
    
    print(f"Reading hands from {input_path}...")
    with open(input_path, 'r') as f:
        hands_data = json.load(f)
    
    processed_hands = []
    total_hands = len(hands_data['hands'])
    
    print(f"Processing {total_hands} hands...")
    for i, hand in enumerate(hands_data['hands'], 1):
        if i % 100 == 0:
            print(f"Processed {i}/{total_hands} hands...")
            
        # Convert dictionary back to HandHistory object
        hand_dict = hand['hand_history']
        game_log = [
            GameAction(
                action=Action(action['action']),
                amount=action['amount'],
                player=Player(action['player']),
                street=Street(action['street']),
                actor=Actor(action['actor'])
            )
            for action in hand_dict['gameLog']
        ]
        
        hand_history = HandHistory(
            hand=hand_dict['hand'],
            board=hand_dict['board'],
            gameLog=game_log
        )
        
        # Normalize the hand (sort cards)
        normalized_hand = normalize_hand(hand_history)
        
        # Encode the hand history
        encoded_hand = EncodedHandHistory.encode_hand_history(normalized_hand)
        
        # Create processed hand entry
        processed_hand = {
            'expected_ev': hand['expected_ev'],
            'encoded_hand_history': {
                'actions': encoded_hand['actions'].tolist(),
                'cards': encoded_hand['cards'].tolist()
            }
        }
        processed_hands.append(processed_hand)
    
    print(f"Writing processed hands to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump({'hands': processed_hands}, f)
    
    print("Done!")

if __name__ == "__main__":
    preprocess_hands() 