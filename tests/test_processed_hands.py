import json
from pathlib import Path
import torch
from models.encoded_handhistory import EncodedHandHistory

def test_processed_hands():
    # Setup paths
    root_dir = Path(__file__).parent.parent  # Go up one level from tests directory
    processed_path = root_dir / "data" / "human" / "hands_processed.json"
    
    print(f"Reading processed hands from {processed_path}...")
    with open(processed_path, 'r') as f:
        hands_data = json.load(f)
    
    print("\nDisplaying first 10 hands:\n")
    for i, hand in enumerate(hands_data['hands'][:10], 1):
        print(f"Hand {i}:")
        print(f"Expected EV: {hand['expected_ev']:.4f}")
        
        # Convert the stored lists back to tensors
        encoded_hand = {
            'actions': torch.tensor(hand['encoded_hand_history']['actions']),
            'cards': torch.tensor(hand['encoded_hand_history']['cards'])
        }
        
        # Decode and display the hand
        decoded_str = EncodedHandHistory.decode_to_string(encoded_hand)
        print(decoded_str)
        print("-" * 80 + "\n")

if __name__ == "__main__":
    test_processed_hands() 