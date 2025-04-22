from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class TrainingHistory:
    stage: Literal["Train", "Validate"] = "Train"
    training_loss_in_epochs: list[float] = field(default_factory=lambda: [0.0])
    validation_loss_in_epochs: list[float] = field(default_factory=lambda: [])
    training_accuracy_in_epochs: list[float] = field(default_factory=lambda: [])
    validation_accuracy_in_epochs: list[float] = field(default_factory=lambda: [])
    best_validation_loss: float = float('inf')
    best_validation_epoch: int = 0
    current_epoch: int = 0
    @classmethod
    def from_checkpoint(cls, checkpoint_dict: dict | None = None) -> "TrainingHistory":
        """Create a TrainingHistory instance, optionally initializing from a checkpoint."""
        if checkpoint_dict is None:
            return cls()
            
        # Create instance with default values
        instance = cls()
        
        # Load the training history from checkpoint
        if "loss_history" in checkpoint_dict:
            history = checkpoint_dict["loss_history"]
            instance.training_loss_in_epochs = history.training_loss_in_epochs
            instance.validation_loss_in_epochs = history.validation_loss_in_epochs
            instance.training_accuracy_in_epochs = history.training_accuracy_in_epochs
            instance.validation_accuracy_in_epochs = history.validation_accuracy_in_epochs
            
            # Set best validation loss from previous history
            if instance.validation_loss_in_epochs:
                min_loss = min(instance.validation_loss_in_epochs)
                min_epoch = instance.validation_loss_in_epochs.index(min_loss)
                instance.best_validation_loss = min_loss
                instance.best_validation_epoch = min_epoch
                
        return instance

    def update_loss_in_batch(self, loss_in_batch: torch.Tensor):
        match self.stage:
            case "Train":
                self.training_loss_in_epochs[-1] += loss_in_batch.item()
            case "Validate":
                self.validation_loss_in_epochs[-1] += loss_in_batch.item()

    def update_accuracy_in_epoch(self, accuracy: float):
        match self.stage:
            case "Train":
                self.training_accuracy_in_epochs.append(accuracy)
            case "Validate":
                self.validation_accuracy_in_epochs.append(accuracy)

    def update_loss_in_epoch(self):
        match self.stage:
            case "Train":
                self.validation_loss_in_epochs.append(0.0)
                self.stage = "Validate"
            case "Validate":
                self.training_loss_in_epochs.append(0.0)
                self.stage = "Train"

    def log_loss_in_epoch(self):
        match self.stage:
            case "Train":
                print(
                    f"Epoch {self.epoch} Training loss: {self.training_loss_in_epochs[-1]}"
                )
            case "Validate":
                print(
                    f"Epoch {self.epoch} Validation loss: {self.validation_loss_in_epochs[-1]}"
                )

    @property
    def epoch(self) -> int:
        return len(self.training_loss_in_epochs)
