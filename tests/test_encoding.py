#!/usr/bin/env python3
"""
Script to convert a hand history JSON file to an EncodedHandHistory object.
"""

import os
import sys
from pathlib import Path

# add git repo base path to sys.path so python3 {path to this file}
# can correctly import models
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.encoded_handhistory import EncodedHandHistory


def main():
    if len(sys.argv) > 1:
        json_path = Path(__file__).parent / sys.argv[1]
    else:
        json_path = Path(__file__).parent / "sample-hand" / "hand1.json"
    json_path = str(json_path)

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
        hole_card_encodings = card_encodings[:2, :]  # (2, 3)
        board_card_encodings = card_encodings[2:, :]  # (5, 3)

        # Print information about the encoded data
        print("\nEncoded Hand History Information:")
        print(f"Number of actions: {len(action_sequence)}")
        print(f"Action sequence shape: {action_sequence.shape}")
        print(f"Hole cards shape: {hole_card_encodings.shape}")
        print(f"Board cards shape: {board_card_encodings.shape}")

        # Print the first action as an example
        if len(action_sequence) > 0:
            print("\nFirst action encoding:")
            print(action_sequence[0])

        # Print the hole cards as an example
        if len(hole_card_encodings) > 0:
            print("\nHole cards encoding:")
            for i, card in enumerate(hole_card_encodings):
                print(f"Card {i+1}: {card}")

        print("\nConversion successful!")
        return 0

    except Exception as e:
        print(f"Error converting hand history: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
