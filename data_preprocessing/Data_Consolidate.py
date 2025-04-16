import re
import os
import json
from glob import glob
import numpy as np

# Predefined bet size buckets (as percentages of the pot)
BET_SIZE_BUCKETS = [0.2, 0.5, 0.8, 1.0, 2.0, float('inf')]  # 0-20%, 20-50%, 50-80%, 80-100%, 100-200%, >200%

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
    #Convert a bet size percentage to the nearest bucket index.
    if bet_size_pct == 0:  # Checks or folds
        return 0
    # Find the first bucket that the bet size is less than or equal to
    for i, threshold in enumerate(BET_SIZE_BUCKETS):
        if bet_size_pct <= threshold:
            return i
    # If bet size is larger than all thresholds, return the last bucket
    return len(BET_SIZE_BUCKETS) - 1

def calculate_bet_pct(amount, current_pot, last_bet=0, is_raise=False):
    """
    Calculate bet size as percentage of the pot.
    For raises: X = (Raise Size - Bet) / (2 × Bet + Pot)
    For bets: X = Bet / Pot
    """
    if is_raise:
        denominator = (2 * last_bet + current_pot)
        if denominator > 0:
            return (amount - last_bet) / denominator
        return 0
    else:
        return amount / current_pot if current_pot > 0 else 0

def parse_hand_history(file_content, filename=None):
    #Parse a single poker hand history file with bet size bucketing.
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
        last_bet = 0  # Track the last bet amount for raise calculations

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
                is_raise = action == "raises"
                bet_pct = calculate_bet_pct(amount, current_pot, last_bet, is_raise)
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

                # Update pot and last bet for raises/bets
                if action == "bets":
                    last_bet = amount
                    current_pot += amount
                elif action == "raises":
                    current_pot += (amount - last_bet)
                    last_bet = amount - last_bet  # The new bet amount is the raise size

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
    #Parse all poker hand history files in a folder.
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
    #Save parsed data to a JSON file with correct type conversion.
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(data), f, indent=2, ensure_ascii=False)
    print(f"Data successfully saved to {output_file}")

if __name__ == "__main__":
    # Debugging: Show current directory and contents
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    # Test bucketing function
    test_cases = [
        (0.15, 0),    # 15% → 0-20% bucket (index 0)
        (0.35, 1),    # 35% → 20-50% bucket (index 1)
        (0.65, 2),    # 65% → 50-80% bucket (index 2)
        (0.85, 3),    # 85% → 80-100% bucket (index 3)
        (1.5, 4),     # 150% → 100-200% bucket (index 4)
        (2.5, 5)      # 250% → >200% bucket (index 5)
    ]
    
    print("\nBucket Testing:")
    for bet_pct, expected_idx in test_cases:
        idx = get_bucket_index(bet_pct)
        bucket_range = f"0-20%" if idx == 0 else f"{BET_SIZE_BUCKETS[idx-1]*100}-{BET_SIZE_BUCKETS[idx]*100}%" if idx < 5 else ">200%"
        print(f"{bet_pct*100:.1f}% → Bucket {idx} ({bucket_range}) {'✓' if idx == expected_idx else '✗'}")
    
    # Test raise calculation
    print("\nRaise Calculation Testing:")
    test_raises = [
        (20, 15, 5, 0.6),   # Raise to 20bb into 15bb pot after 5bb bet → 60%
        (12, 10, 2, 0.5),    # Raise to 12bb into 10bb pot after 2bb bet → 50%
        (8, 6, 2, 0.4)       # Raise to 8bb into 6bb pot after 2bb bet → 40%
    ]
    for raise_size, pot, last_bet, expected_pct in test_raises:
        pct = calculate_bet_pct(raise_size, pot, last_bet, True)
        print(f"Raise to {raise_size} into {pot} pot after {last_bet} bet → {pct*100:.1f}% {'✓' if abs(pct - expected_pct) < 0.01 else '✗'}")
    
    # Process real data
    folder_path = "Ignition2.0"
    print(f"\nLooking for hand history folder at: {os.path.abspath(folder_path)}")
    
    if os.path.exists(folder_path):
        all_hands = parse_hand_history_folder(folder_path)
        save_to_json(all_hands, "bucketed_hands_1.json")
        print(f"\nTotal hands parsed: {len(all_hands)}")
        
        # Show example of bucketed actions
        if all_hands:
            example_hand = all_hands[0]
            print("\nExample hand with bucketed actions:")
            for action in example_hand["game_log"]:
                if action["action"] in ["bets", "raises"]:
                    print(f"{action['action']}: {action['amount']} → {action['pot_pct']*100:.1f}% → Bucket {action['bucket']}")
    else:
        print(f"\nError: Folder not found at {os.path.abspath(folder_path)}")
        print("Please ensure:")
        print("1. The 'Ignition2.0' folder exists in the same directory as this script")
        print("2. The folder contains .txt hand history files")
        print(f"Current directory contents: {os.listdir('.')}")