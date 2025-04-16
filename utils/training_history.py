from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class TrainingHistory:
    stage: Literal["Train", "Validate"] = "Train"
    training_loss_in_epochs: list[float] = field(default_factory=lambda: [0.0])
    validation_loss_in_epochs: list[float] = field(default_factory=lambda: [])

    def update_loss_in_batch(self, loss_in_batch: torch.Tensor):
        match self.stage:
            case "Train":
                self.training_loss_in_epochs[-1] += loss_in_batch.item()
            case "Validate":
                self.validation_loss_in_epochs[-1] += loss_in_batch.item()

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
