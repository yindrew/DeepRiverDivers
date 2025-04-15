import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from load_gto_data import load_gto_hands
from models.full_model import FullModel


class PokerDataset(Dataset):
    """Dataset class for poker hand histories."""

    def __init__(
        self, hand_histories: list[tuple[dict, float]], max_seq_length: int = 20
    ):
        """
        Initialize the dataset.

        Args:
            hand_histories: list of (encoded_hand_dict, expected_value) tuples
            max_seq_length: Maximum sequence length for padding
        """
        self.hand_histories = hand_histories
        self.max_seq_length = max_seq_length

        # The data is already encoded, so we just need to extract it
        self.encoded_hands = [encoded for encoded, _ in hand_histories]
        self.evs = torch.FloatTensor([ev for _, ev in hand_histories])

    def __len__(self) -> int:
        return len(self.hand_histories)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single hand history and its expected value.

        Args:
            idx: Index of the hand to get

        Returns:
            tuple of (encoded_hand, expected_value)
        """
        encoded = self.encoded_hands[idx]

        # Pad actions tensor to max_seq_length
        actions = encoded["actions"]
        if actions.shape[0] < self.max_seq_length:
            # Create a padding tensor
            padding = torch.zeros(
                (self.max_seq_length - actions.shape[0], actions.shape[1]),
                dtype=actions.dtype,
            )
            # Concatenate the original tensor with the padding
            actions = torch.cat([actions, padding], dim=0)
        elif actions.shape[0] > self.max_seq_length:
            # Truncate to max_seq_length
            actions = actions[: self.max_seq_length]

        return {"actions": actions, "cards": encoded["cards"]}, self.evs[idx]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str,
    run_name: str,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """
    Train the model.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        checkpoint_dir: Directory to save model checkpoints
        run_name: Name of the training run (used in saved model names)
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")

    for epoch in range(num_epochs):
        # NOTE: can maybe abstract
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            inputs, targets = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs["actions"], inputs["cards"])
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        # NOTE: can maybe abstract
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                inputs, targets = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)

                outputs = model(inputs["actions"], inputs["cards"])
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"  Saved new best model with validation loss: {val_loss:.4f}")

    return model


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Generate a unique run name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"poker_model_{timestamp}"

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    model = FullModel()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="checkpoints",
        run_name=run_name,
        num_epochs=100,
        learning_rate=0.001,
        device=device,
    )

    # Save final model
    final_model_path = os.path.join("checkpoints", f"{run_name}_final.pth")
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "train_dataset_size": len(train_hands),
            "val_dataset_size": len(val_hands),
            "max_seq_length": max_seq_length,
        },
        final_model_path,
    )
    print(f"Training complete! Final model saved as '{final_model_path}'")


if __name__ == "__main__":
    main()

