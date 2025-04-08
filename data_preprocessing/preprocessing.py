import re
import os
import json
from glob import glob
import numpy as np

# Predefined bet size buckets (as percentages of the pot)
BET_SIZE_BUCKETS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # 25%, 50%, 75%, 100%, 150%, 200%

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

def get_bucket_index(bet_size_pct):
    """Convert a bet size percentage to the nearest bucket index."""
    if bet_size_pct == 0:  # Checks or folds
        return 0
    # Find the bucket with the smallest absolute difference
    return int(np.argmin([abs(bet_size_pct - bucket) for bucket in BET_SIZE_BUCKETS]))

def parse_hand_history(file_content, filename=None):
    """Parse a single poker hand history file with bet size bucketing."""
    hands = file_content.strip().split("\n\n")
    parsed_hands = []

    for hand in hands:
        lines = hand.split("\n")
        board = None
        hand_cards = None
        game_log = []
        big_blind = None
        hand_id = None
        timestamp = None
        current_pot = 0

        for line in lines:
            # Extract hand ID and timestamp
            if line.startswith("Ignition Hand #"):
                parts = line.split()
                hand_id = parts[2]
                timestamp = " ".join(parts[-2:])
            
            # Track pot size changes
            if "Total Pot($" in line:
                pot_match = re.search(r'Total Pot\(\$(\d+\.?\d*)', line)
                if pot_match:
                    current_pot = float(pot_match.group(1))
            
            # Extract big blind amount
            if "Big blind $" in line:
                bb_match = re.search(r'Big blind \$(\d+\.?\d*)', line)
                if bb_match:
                    big_blind = float(bb_match.group(1))
            
            # Extract board cards
            if "*** FLOP ***" in line:
                board = re.findall(r'\[(.*?)\]', line)[0].split()
            elif "*** TURN ***" in line or "*** RIVER ***" in line:
                new_cards = re.findall(r'\[(.*?)\]', line)[0].split()
                if board is not None:
                    board.extend(new_cards)
            
            # Extract player's hole cards
            elif "Card dealt to a spot" in line and "[ME]" in line:
                cards = re.findall(r'\[(.*?)\]', line)
                if cards:
                    hand_cards = cards[0].split()
            
            # Process player actions with bucketing
            elif ("Raises" in line or "Calls" in line or "Checks" in line or 
                  "Bets" in line or "Folds" in line or "All-in" in line) and "[ME]" in line:
                action = re.findall(r'(Raises|Calls|Checks|Bets|Folds|All-in)', line, re.IGNORECASE)[0].lower()
                amount = re.findall(r'\$\d+\.?\d*', line)
                amount = float(amount[0][1:]) if amount else 0.0
                
                # Calculate bet size as percentage of pot
                bet_pct = amount / current_pot if current_pot > 0 else 0
                bucket_idx = get_bucket_index(bet_pct)
                
                player = re.findall(r'Seat \d+: (\w+)', line)
                player = player[0] if player else None
                
                game_log.append({
                    "action": action,
                    "amount": amount,
                    "amount_bb": amount / big_blind if big_blind and big_blind > 0 else amount,
                    "pot_pct": bet_pct,
                    "bucket": bucket_idx,
                    "bucket_value": BET_SIZE_BUCKETS[bucket_idx] if action in ["bets", "raises"] else 0,
                    "player": player
                })

                # Update pot for raises/bets (calls don't increase the pot)
                if action in ["raises", "bets"]:
                    current_pot += amount

        if hand_cards:
            parsed_hand = {
                "hand_id": hand_id,
                "timestamp": timestamp,
                "filename": filename,
                "board": board,
                "hand": hand_cards,
                "game_log": game_log,
                "big_blind": big_blind,
                "buckets": BET_SIZE_BUCKETS
            }
            parsed_hands.append(convert_to_serializable(parsed_hand))

    return parsed_hands

def parse_hand_history_folder(folder_path):
    """Parse all poker hand history files in a folder."""
    all_hands = []
    file_paths = glob(os.path.join(folder_path, "**/*.txt"), recursive=True)
    
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                file_content = file.read()
            filename = os.path.basename(file_path)
            parsed_hands = parse_hand_history(file_content, filename)
            all_hands.extend(parsed_hands)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    return all_hands

def save_to_json(data, output_file):
    """Save parsed data to a JSON file with proper type conversion."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(data), f, indent=2, ensure_ascii=False)
    print(f"Data successfully saved to {output_file}")

if __name__ == "__main__":
    # Test bucketing function
    test_cases = [
        (0.32, 0),    # 32% → 25% bucket (index 0)
        (0.45, 1),    # 45% → 50% bucket (index 1)
        (1.25, 3),    # 125% → 100% bucket (index 3)
        (1.8, 4)      # 180% → 150% bucket (index 4)
    ]
    
    print("Bucket Testing:")
    for bet_pct, expected_idx in test_cases:
        idx = get_bucket_index(bet_pct)
        print(f"{bet_pct*100:.1f}% → Bucket {idx} ({BET_SIZE_BUCKETS[idx]*100}%) {'✓' if idx == expected_idx else '✗'}")
    
    # Process real data
    folder_path = "Ignition2.0"
    if os.path.exists(folder_path):
        all_hands = parse_hand_history_folder(folder_path)
        save_to_json(all_hands, "bucketed_hands.json")
        print(f"\nTotal hands parsed: {len(all_hands)}")
        
        # Show example of bucketed actions
        if all_hands:
            example_hand = all_hands[0]
            print("\nExample hand with bucketed actions:")
            for action in example_hand["game_log"]:
                if action["action"] in ["bets", "raises"]:
                    print(f"{action['action']}: {action['amount']} into {action['amount']/action['pot_pct'] if action['pot_pct']>0 else 0:.1f} pot → {action['pot_pct']*100:.1f}% → Bucket {action['bucket']} ({BET_SIZE_BUCKETS[action['bucket']]*100}%)")
    else:
        print(f"Folder not found: {folder_path}")
