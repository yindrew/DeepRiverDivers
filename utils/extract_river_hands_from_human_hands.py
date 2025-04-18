#!/usr/bin/env python3
"""
Extract only hands that reach the river from Ignition hand history files. generated the river_all_in_hands.json file
"""

import os
import argparse
from utils.real_hand_parser import RealHandParser

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract hands that reach the river from Ignition hand histories')
    parser.add_argument('--input', '-i', type=str, default='Ignition2.0',
                        help='Input folder containing hand history files (default: Ignition2.0)')
    parser.add_argument('--output', '-o', type=str, default='river_hands.json',
                        help='Output JSON file for river hands (default: river_hands.json)')
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' not found.")
        return
    
    # Create parser and process hand histories
    hand_parser = RealHandParser()
    print(f"Processing hand histories from '{args.input}'...")
    hands = hand_parser.parse_hand_folder(args.input)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total hands reaching the river: {len(hands)}")
    
    # Save to JSON
    hand_parser.save_to_json(hands, args.output)
    print(f"\nDone! River hands saved to '{args.output}'")

if __name__ == "__main__":
    main() 