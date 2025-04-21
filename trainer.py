import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

# can correctly import models
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.full_model import FullModel
from schemas.checkpoint_dict import CheckpointDict
from schemas.dataloader_datatypes import DatasetBaseCollatedType, DatasetBaseType
from schemas.model_config import ModelConfig
from utils.custom_torch_data_utils import (
    GTODataset,
    HumanDataset,
    collate_fn_datasetbasetype,
    three_way_split,
)
from utils.training_history import TrainingHistory

torch.serialization.add_safe_globals([TrainingHistory, ModelConfig])


class Trainer:
    config_filename: str | None
    config: ModelConfig
    optim: torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD
    model: FullModel
    loss_fn: nn.MSELoss | nn.HuberLoss
    dataloader_training: DataLoader[DatasetBaseType]
    dataloader_validation: DataLoader[DatasetBaseType]
    dataloader_test: DataLoader[DatasetBaseType]
    training_history: TrainingHistory
    base_path: Path
    start_time: float
    writer: SummaryWriter

    def __init__(self, config_filename: str | None = None) -> None:
        self.start_time = time.time()
        self.base_path = Path(__file__).parent
        self.config_filename = config_filename
        self.config = self._build_config()
        self._init_seed()
        self._set_device()
        self.model = FullModel(config=self.config)
        self.optim = self._build_optim()
        
        # Use MSE for GTO data, Huber for human data
        if self.config.training_process["dataset"] == "GTO":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.HuberLoss(delta=1.0)  # delta=1.0 is a good default for regression
            
        self.dataloader_training, self.dataloader_validation, self.dataloader_test = self._build_dataloaders()
        
        # Initialize training history from checkpoint if doing warm start
        if self.config.training_process["warm_start"]:
            checkpoint_dict = self._load_checkpoint_dict()
            self.training_history = TrainingHistory.from_checkpoint(checkpoint_dict)
            print(f"Loaded previous model from checkpoint: {self.config.general['load_checkpoint_name']}")
        else:
            self.training_history = TrainingHistory()
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(f'runs/{self.config.general["save_checkpoint_name"]}')
                
        # Log hyperparameters
        self.writer.add_hparams(
            {
                "learning_rate": self.config.training_process["learning_rate"],
                "batch_size": self.config.training_process["batch_size"],
                "optimizer": self.config.training_process["optimizer"],
                "weight_decay": self.config.training_process["weight_decay"],
                "num_epochs": self.config.training_process["num_epochs"],
                "dataset": self.config.training_process["dataset"],
            },
            {"hparam/val_loss": float('inf')}  # Start with infinity, will be updated during training
        )

    def run(self):
        self._init_params()
        self._log_training_setup()
        self._forward_n_epochs()
        self._log_training_result()
        
        # Add final evaluation on test set
        test_accuracy = self.evaluate_on_test_set()
        print(f"Final Test Set Accuracy: {test_accuracy:.2%}")

    def _init_seed(self) -> None:
        _ = torch.manual_seed(self.config.general["seed"])

    def _set_device(self) -> None:
        _ = torch.set_default_device(self.config.general["device"])

    def _init_params(self) -> None:
        if self.config.training_process["warm_start"]:
            checkpoint_dict = self._load_checkpoint_dict()
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            self.optim.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    def _forward_n_epochs(self):
        patience = self.config.training_process["patience"]
        min_epochs = self.config.training_process["min_epochs"]
        patience_counter = 0
        
        for _ in range(self.config.training_process["num_epochs"]):
            self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            
            # Check if this is a new best validation loss
            if val_loss < self.training_history.best_validation_loss:
                self.training_history.best_validation_loss = val_loss
                self.training_history.best_validation_epoch = self.training_history.epoch
                self._save_checkpoint(postfix="best")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping check - only after min_epochs
            if self.training_history.epoch >= min_epochs and patience_counter >= patience:
                print(f"\nEarly stopping triggered")
                break
        
        self._save_checkpoint(postfix="final")
        print(f"\nTraining completed:")
        print(f"- Total epochs trained: {self.training_history.epoch}")
        print(f"- Best validation loss: {self.training_history.best_validation_loss:.6f} at epoch {self.training_history.best_validation_epoch}")
        
        # Close TensorBoard writer
        self.writer.close()

    def _train_one_epoch(self) -> None:
        _ = self.model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, data in enumerate(self.dataloader_training):
            data: DatasetBaseCollatedType
            data = {k: v.to(self.config.general["device"]) for k, v in data.items()}
            self.optim.zero_grad()
            loss_in_batch = self._gen_prediction_and_loss(data) / len(
                self.dataloader_training
            )
            loss_in_batch.backward()
            self.optim.step()
            self.training_history.update_loss_in_batch(loss_in_batch)
            epoch_loss += loss_in_batch.item()
            
            # Calculate accuracy metrics
            with torch.no_grad():
                y_pred = self.model(data["actions_padded"], data["cards"], data["actions_mask"])
                y_target = data["expected_ev"]
                correct_predictions += ((y_pred > 0) == (y_target > 0)).sum().item()
                total_predictions += y_pred.size(0)
        
        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / len(self.dataloader_training)
        epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        self.writer.add_scalar('Training/EpochLoss', avg_epoch_loss, self.training_history.epoch)
        self.writer.add_scalar('Training/EpochAccuracy', epoch_accuracy, self.training_history.epoch)
        
        self.training_history.log_loss_in_epoch()
        self.training_history.update_loss_in_epoch()

    def _validate_one_epoch(self) -> float:
        _ = self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataloader_validation):
                data: DatasetBaseCollatedType
                data = {k: v.to(self.config.general["device"]) for k, v in data.items()}
                loss_in_batch = self._gen_prediction_and_loss(data) / len(
                    self.dataloader_validation
                )
                val_loss += loss_in_batch.item()
                self.training_history.update_loss_in_batch(loss_in_batch)
                
                # Calculate accuracy metrics
                y_pred = self.model(data["actions_padded"], data["cards"], data["actions_mask"])
                y_target = data["expected_ev"]
                correct_predictions += ((y_pred > 0) == (y_target > 0)).sum().item()
                total_predictions += y_pred.size(0)
            
            # Log epoch-level validation metrics
            avg_val_loss = val_loss / len(self.dataloader_validation)
            val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            self.writer.add_scalar('Validation/EpochLoss', avg_val_loss, self.training_history.epoch)
            self.writer.add_scalar('Validation/EpochAccuracy', val_accuracy, self.training_history.epoch)
            
            # Update hyperparameter metrics with validation loss
            self.writer.add_scalar('hparam/val_loss', avg_val_loss, self.training_history.epoch)
            
            self.training_history.log_loss_in_epoch()
            self.training_history.update_loss_in_epoch()
        return val_loss

    def _gen_prediction_and_loss(self, data: DatasetBaseCollatedType) -> torch.Tensor:
        x_actions = data["actions_padded"]
        x_cards = data["cards"]
        x_actions_mask = data["actions_mask"]
        y_target = data["expected_ev"]
        y_pred = self.model(x_actions, x_cards, x_actions_mask)
        loss_in_batch = self.loss_fn(y_pred, y_target)
        return loss_in_batch

    def _build_optim(self) -> torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD:
        optim_str = self.config.training_process["optimizer"]
        match optim_str:
            case "Adam":
                return torch.optim.Adam(
                    params=self.model.parameters(),
                    lr=self.config.training_process["learning_rate"],
                    weight_decay=self.config.training_process["weight_decay"],
                )
            case "AdamW":
                return torch.optim.AdamW(
                    params=self.model.parameters(),
                    lr=self.config.training_process["learning_rate"],
                    weight_decay=self.config.training_process["weight_decay"],
                )
            case "SGD":
                return torch.optim.SGD(
                    params=self.model.parameters(),
                    lr=self.config.training_process["learning_rate"],
                    momentum=self.config.training_process["momentum"],
                    weight_decay=self.config.training_process["weight_decay"],
                )

    def _build_config(self) -> ModelConfig:
        config_dir = self.base_path / "configs"
        config = ModelConfig()
        if self.config_filename is None:
            return config
        # else: update dict keys based on params existing in the JSON
        json_file = config_dir / self.config_filename
        with open(json_file, "r") as f:
            config_json_dict = json.load(f)
        for base_key, values in config_json_dict.items():
            if base_key == "$schema":
                continue
            config.__setattr__(
                base_key,
                config.__getattribute__(base_key) | values,
            )
        return config

    def _build_dataloaders(self):
        dataset = {"GTO": GTODataset, "Human": HumanDataset}[
            self.config.training_process["dataset"]
        ]()
        
        # Use three_way_split instead of train_test_split
        train_dataset, val_dataset, test_dataset = three_way_split(
            dataset,
            device=self.config.general["device"],
            p_train_test_split=self.config.training_process["p_train_test_split"],
        )
        
        # Create data loaders for each set
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_process["batch_size"],
            generator=torch.Generator(device=self.config.general["device"]),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_datasetbasetype,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_process["batch_size"],
            generator=torch.Generator(device=self.config.general["device"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_datasetbasetype,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training_process["batch_size"],
            generator=torch.Generator(device=self.config.general["device"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_datasetbasetype,
        )
        
        return train_loader, val_loader, test_loader

    def _load_checkpoint_dict(self) -> dict:
        """Load the checkpoint dictionary without applying it to the model."""
        checkpoint_path = (
            self.base_path
            / "checkpoints"
            / self.config.general["load_checkpoint_name"]
            / f"{self.config.general['type_of_checkpoint']}.pth"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No previous checkpoints found.")
        
        return torch.load(
            checkpoint_path, 
            map_location=self.config.general["device"],
            weights_only=False  
        )

    def _save_checkpoint(self, postfix: Literal["best", "final"] = "final") -> None:
        checkpoint_dir = (
            self.base_path / "checkpoints" / self.config.general["save_checkpoint_name"]
        )
        checkpoint_path = checkpoint_dir / f"{postfix}.pth"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_dict: CheckpointDict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "loss_history": self.training_history,
            "model_config": self.config,
        }
        torch.save(checkpoint_dict, checkpoint_path)

    def _log_training_setup(self):
        print(
            f"""
Training general setup:
    Device: {self.config.general["device"]}
    Warm Start: {self.config.training_process["warm_start"]} 
        """
        )
        if self.config.training_process["warm_start"]:
            print(f"Loading checkpoint_name={self.config.general['load_checkpoint_name']}")
        else:
            print(f"Starting training from scratch")
        print()
    def _log_training_result(self):
        print(
            f"""
Training result:
    Ellapsed time: {time.time() - self.start_time} seconds
    Best validation loss: {min(self.training_history.validation_loss_in_epochs)}
            """
        )

    def evaluate_on_test_set(self):
        """Evaluate model performance on the held-out test set"""
        self.model.eval()
        correct_decisions = 0
        total_decisions = 0
        
        with torch.no_grad():
            for data in self.dataloader_test:
                data = {k: v.to(self.config.general["device"]) for k, v in data.items()}
                
                # Get model predictions
                predicted_ev = self.model(
                    data["actions_padded"],
                    data["cards"],
                    data["actions_mask"]
                )
                
                # Get actual outcomes
                actual_outcomes = data["expected_ev"] > 0
                
                # Model decisions
                model_decisions = predicted_ev > 0
                
                # Calculate correct decisions
                correct_decisions += (
                    (model_decisions & actual_outcomes) |  # Correct calls
                    (~model_decisions & ~actual_outcomes)  # Correct folds
                ).sum().item()
                
                total_decisions += len(predicted_ev)
        
        accuracy = correct_decisions / total_decisions
        
        # Log final test accuracy to TensorBoard
        self.writer.add_scalar('Test/FinalAccuracy', accuracy, 0)
        
        return accuracy


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Trainer",
        description="Class for training models",
    )
    arg_parser.add_argument(
        "--config",
        default="example.json",
        type=str,
        help="name of config file to load from the ./configs directory",
    )
    args = arg_parser.parse_args()
    trainer = Trainer(args.config)
    trainer.run()
