import json
from pathlib import Path
from models.encoded_handhistory import EncodedHandHistory
from schemas.hand_history import Action, Actor, GameAction, HandHistory, Player, Street

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
        
        # Encode the hand history
        encoded_hand = EncodedHandHistory.encode_hand_history(hand_history)
        
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