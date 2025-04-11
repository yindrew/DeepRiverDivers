#!/usr/bin/env python3
"""
Script to convert a hand history JSON file to an EncodedHandHistory object.
"""

import os
import sys

from models.encoded_handhistory import EncodedHandHistory


def main():
    # Get the JSON file path from command line arguments or use default
    json_file = sys.argv[1] if len(sys.argv) > 1 else "../sample-hand/hand1.json"

    # Ensure the file path is relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, json_file)

    # Check if the file exists
    if not os.path.exists(json_path):
        print(f"Error: File '{json_path}' not found.")
        return 1

    try:
        # Convert the JSON file to encoded format
        print(f"Converting {json_path} to encoded format...")
        encoded_data = EncodedHandHistory.from_json(json_path)

        # Get the encoded data
        action_sequence = encoded_data["actions"]
        card_encodings = encoded_data["cards"]

        # Print information about the encoded data
        print("\nEncoded Hand History Information:")
        print(f"Number of actions: {len(action_sequence)}")
        print(f"Action sequence shape: {action_sequence.shape}")
        print(f"Hole cards shape: {card_encodings['hole_cards'].shape}")
        print(f"Board cards shape: {card_encodings['board_cards'].shape}")

        # Print the first action as an example
        if len(action_sequence) > 0:
            print("\nFirst action encoding:")
            print(action_sequence[0])

        # Print the hole cards as an example
        if len(card_encodings["hole_cards"]) > 0:
            print("\nHole cards encoding:")
            for i, card in enumerate(card_encodings["hole_cards"]):
                print(f"Card {i+1}: {card}")

        print("\nConversion successful!")
        return 0

    except Exception as e:
        print(f"Error converting hand history: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

