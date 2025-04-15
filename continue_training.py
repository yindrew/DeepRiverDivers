import argparse
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.full_model import FullModel
from train_model import PokerDataset, load_gto_hands, train_model


def load_checkpoint(checkpoint_path, device):
    """Load a model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Continue training a poker model from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for fine-tuning"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model and checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    # Load data
    print("Loading GTO hands...")
    hand_histories = load_gto_hands("gto-hands")
    print(f"Loaded {len(hand_histories)} hand histories")

    # Find the maximum sequence length
    max_seq_length = max(encoded["actions"].shape[0] for encoded, _ in hand_histories)
    print(f"Maximum sequence length: {max_seq_length}")

    # Split into train/val sets (80/20)
    random.shuffle(hand_histories)
    split_idx = int(len(hand_histories) * 0.8)
    train_hands = hand_histories[:split_idx]
    val_hands = hand_histories[split_idx:]

    print(f"Training set: {len(train_hands)} hands")
    print(f"Validation set: {len(val_hands)} hands")

    # Create datasets
    train_dataset = PokerDataset(train_hands, max_seq_length=max_seq_length)
    val_dataset = PokerDataset(val_hands, max_seq_length=max_seq_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Generate a unique run name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"continued_training_{timestamp}"

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Continue training
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
    )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f"{run_name}_final.pth")
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "train_dataset_size": len(train_hands),
            "val_dataset_size": len(val_hands),
            "max_seq_length": max_seq_length,
            "continued_from": args.checkpoint,
        },
        final_model_path,
    )
    print(f"Training complete! Final model saved as '{final_model_path}'")


if __name__ == "__main__":
    main()

