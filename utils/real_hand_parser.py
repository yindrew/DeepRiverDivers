import re
import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


""" Used in conjunction with extract_river_hands_from_human_hands.py to generate the river_all_in_hands.json file"""
class RealHandParser:
    """Parser for Ignition poker hand histories that extracts only hands with river all-ins"""
    
    def __init__(self):
        self.river_tag = "*** RIVER ***"
        self.flop_tag = "*** FLOP ***"
        # Updated regex to handle all positions and proper spacing
        self.hero_pattern = r'(?:UTG(?:\+[12])?|BB|SB|Dealer|Small Blind|Big Blind)\s*:\s*Card dealt to a spot \[(.*?)\]'
    
    def has_river_all_in(self, hand_raw: str) -> bool:
        """Check if there was an all-in on the river in a hand"""
        # Find the river section
        sections = hand_raw.split(self.river_tag)
        if len(sections) < 2:
            return False
            
        # Get the river section (right after the river tag)
        river_section = sections[1].strip().split('\n')
        
        # Check for "All-in" in the river section
        for line in river_section:
            if "All-in" in line:
                return True
        
        return False
    
    def has_two_players_on_flop(self, hand_raw: str) -> bool:
        """verify only two players to the flop"""
        # Find the flop section
        flop_sections = hand_raw.split(self.flop_tag)
        if len(flop_sections) < 2:
            return False
            
        # Get the section after the flop tag
        after_flop = flop_sections[1]
        
        # Find the turn tag to get only the flop section
        turn_sections = after_flop.split("*** TURN ***")
        if "*** TURN ***" not in after_flop:
            return False
            
        # Get the flop section (between flop tag and turn tag)
        flop_section = turn_sections[0].strip().split('\n')
        
        # Count players involved in flop action
        players_involved = set()
        for line in flop_section:
            if ":" in line:  # Only count lines with player actions
                player_name = line.split(":")[0].strip()
                players_involved.add(player_name)
        
        # Return True if exactly 2 players were involved
        return len(players_involved) == 2

    def parse_hand_file(self, file_path: str) -> List[HandHistory]:
        """Parse a hand history file and extract only hands with river all-ins"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split the content into individual hands (separated by two newlines)
        hands_raw = content.strip().split("\n\n")
        parsed_hands = []
        
        for hand_raw in hands_raw:
            # Skip empty hands
            if not hand_raw.strip():
                continue
            
            # Check if this hand reaches the river
            if self.river_tag not in hand_raw:
                continue
            
            # Check if there was an all-in on the river
            river_all_in = self.has_river_all_in(hand_raw)
            if not river_all_in:
                continue
            
            two_players_on_flop = self.has_two_players_on_flop(hand_raw)
            if not two_players_on_flop:
                continue

            # Extract hand ID and timestamp
            hand_id = None
            timestamp = None
            hero_cards = None
            board = []
            pot_size = 0.0
            
            lines = hand_raw.split("\n")
            for line in lines:
                # Extract hand ID and timestamp
                if line.startswith("Ignition Hand #"):
                    parts = line.split()
                    hand_id = parts[2]
                    timestamp = " ".join(parts[-2:])
                
                # Extract board cards
                if "*** FLOP ***" in line:
                    board_match = re.search(r'\[(.*?)\]', line)
                    if board_match:
                        board = board_match.group(1).split()
                elif "*** TURN ***" in line:
                    turn_match = re.search(r'\[(.*?)\]', line)
                    if turn_match:
                        board.append(turn_match.group(1))
                elif "*** RIVER ***" in line:
                    river_match = re.search(r'\[(.*?)\]', line)
                    if river_match:
                        board.append(river_match.group(1))
                
                # Extract hero's hole cards
                elif "Card dealt to a spot" in line and "[ME]" in line:
                    cards_match = re.search(self.hero_pattern, line)
                    if cards_match:
                        hero_cards = cards_match.group(1).split()
                
                # Extract pot size
                elif "Total Pot($" in line:
                    pot_match = re.search(r'Total Pot\(\$(\d+\.?\d*)', line)
                    if pot_match:
                        pot_size = float(pot_match.group(1))
            
            # Create hand history object
            if hand_id:
                hand_history = HandHistory(
                    hand_id=hand_id,
                    timestamp=timestamp,
                    content=hand_raw,
                    board=board,
                    hero_cards=hero_cards,
                    pot_size=pot_size,
                    river_all_in=river_all_in
                )
                parsed_hands.append(hand_history)
        
        return parsed_hands
    
    def parse_hand_folder(self, folder_path: str) -> List[HandHistory]:
        """Parse all hand history files in a folder and extract only hands with river all-ins"""
        all_hands = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        hands = self.parse_hand_file(file_path)
                        all_hands.extend(hands)
                        print(f"Processed {file_path}: Found {len(hands)} hands with river all-ins")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
        
        return all_hands
    
    def save_to_json(self, hands: List[HandHistory], output_file: str):
        """Save parsed hands to a JSON file, only saving the content and river_all_in fields"""
        # Convert to serializable format - only include content and river_all_in
        serializable_hands = []
        for hand in hands:
            hand_dict = {
                "content": hand.content,
                "river_all_in": hand.river_all_in
            }
            serializable_hands.append(hand_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_hands, f, indent=2)
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total hands with river all-ins: {len(hands)}")
        print(f"Saved all hands to {output_file}")

# Example usage
if __name__ == "__main__":
    parser = RealHandParser()
    
    # Parse a folder of hand histories
    folder_path = "Ignition2.0"
    if os.path.exists(folder_path):
        hands = parser.parse_hand_folder(folder_path)
        print(f"Total hands with river all-ins: {len(hands)}")
        
        # Save to JSON
        parser.save_to_json(hands, "river_all_in_hands.json")
    else:
        print(f"Folder not found: {folder_path}")
