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
from utils.visualization import plot_all_metrics
from models.encoded_handhistory import EncodedHandHistory

torch.serialization.add_safe_globals([TrainingHistory, ModelConfig])


class Trainer:
    config_filename: str | None
    config: ModelConfig
    optim: torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD
    model: FullModel
    loss_fn: nn.MSELoss | nn.HuberLoss | nn.BCEWithLogitsLoss
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
        
        # Use MSE for GTO data, BCEWithLogitsLoss for human data
        if self.config.training_process["dataset"] == "GTO":
            self.loss_fn = nn.MSELoss()
        else:
            # Calculate class weights for BCEWithLogitsLoss
            dataset = HumanDataset()
            ev_values = torch.tensor(dataset.expected_ev_list)
            num_calls = (ev_values > 0).sum().item()
            num_folds = (ev_values <= 0).sum().item()
            total = len(ev_values)
            
            # Calculate class weights inversely proportional to class frequencies
            pos_weight = torch.tensor([num_folds / num_calls if num_calls > 0 else 1.0])
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
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
                "loss_function": "MSE" if self.config.training_process["dataset"] == "GTO" else "BCE",
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
        
        # Create plots directory and save visualizations
        plots_dir = self.base_path / "plots" / self.config.general["save_checkpoint_name"]
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_all_metrics(self.training_history, str(plots_dir))

    def _init_seed(self) -> None:
        _ = torch.manual_seed(self.config.general["seed"])

    def _set_device(self) -> None:
        _ = torch.set_default_device(self.config.general["device"])

    def _init_params(self) -> None:
        if self.config.training_process["warm_start"]:
            checkpoint_dict = self._load_checkpoint_dict()
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            
            # For human data fine-tuning, we need to handle the optimizer state differently
            if self.config.training_process["dataset"] == "Human":
                # Create a new optimizer state with only OutputMLP parameters
                self.optim = self._build_optim()
            else:
                # Load the full optimizer state for GTO training
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
            
            # Calculate accuracy metrics based on EV sign
            with torch.no_grad():
                y_pred = self.model(data["actions_padded"], data["cards"], data["actions_mask"])
                y_target = data["expected_ev"]
                predictions = (y_pred > 0).float()
                correct_predictions += (predictions == (y_target > 0).float()).sum().item()
                total_predictions += y_pred.size(0)
        
        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / len(self.dataloader_training)
        epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        self.writer.add_scalar('Training/EpochLoss', avg_epoch_loss, self.training_history.epoch)
        self.writer.add_scalar('Training/EpochAccuracy', epoch_accuracy, self.training_history.epoch)
        
        self.training_history.update_accuracy_in_epoch(epoch_accuracy)
        self.training_history.log_loss_in_epoch()
        self.training_history.update_loss_in_epoch()  # This will switch to validation stage

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
                
                # Calculate accuracy metrics based on EV sign
                y_pred = self.model(data["actions_padded"], data["cards"], data["actions_mask"])
                y_target = data["expected_ev"]
                
                # Convert to binary predictions and targets (call/fold decisions)
                predictions = (y_pred > 0).float()
                targets = (y_target > 0).float()
                
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += y_pred.size(0)
                
                # Log detailed information for first hand in batch
                if batch_idx == 0:  # Only log first batch to avoid spam
                    # Get a random hand from the batch
                    batch_size = data["actions_padded"].size(0)
                    random_idx = torch.randint(0, batch_size, (1,)).item()
                    
                    hand_data = {
                        "actions": data["actions_padded"][random_idx].cpu(),
                        "cards": data["cards"][random_idx].cpu(),
                        "mask": data["actions_mask"][random_idx].cpu() if data["actions_mask"] is not None else None
                    }
                    
                    # Decode the hand history to readable format
                    hand_str = EncodedHandHistory.decode_to_string(hand_data)
                    
                    # Get predicted and actual values
                    pred_ev = y_pred[random_idx].item()
                    actual_ev = y_target[random_idx].item()
                    
                    # Convert predictions to probabilities and decisions
                    if self.config.training_process["dataset"] == "Human":
                        pred_prob = torch.sigmoid(torch.tensor(pred_ev)).item()
                    else:
                        pred_prob = 1.0 if pred_ev > 0 else 0.0
                    
                    actual_decision = float(actual_ev > 0)
                    model_decision = pred_prob > 0.5
                    
                    print(f"\nExample Hand from Validation Batch {self.training_history.epoch}:")
                    print(f"Hand History:\n{hand_str}")
                    print(f"Model Decision: {'Call' if model_decision else 'Fold'}")
                    print(f"Correct Decision: {'Call' if actual_decision == 1 else 'Fold'}")
                    print(f"Decision Confidence: {abs(pred_prob - 0.5) * 2:.2%}")
                    if self.config.training_process["dataset"] == "GTO":
                        print(f"Predicted EV: {pred_ev:.2f}")
                        print(f"Actual EV: {actual_ev:.2f}")
                    print("-" * 50)
            
            # Log epoch-level validation metrics
            avg_val_loss = val_loss / len(self.dataloader_validation)
            val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            self.writer.add_scalar('Validation/EpochLoss', avg_val_loss, self.training_history.epoch)
            self.writer.add_scalar('Validation/EpochAccuracy', val_accuracy, self.training_history.epoch)
            
            # Update hyperparameter metrics with validation loss
            self.writer.add_scalar('hparam/val_loss', avg_val_loss, self.training_history.epoch)
            
            self.training_history.update_accuracy_in_epoch(val_accuracy)
            self.training_history.log_loss_in_epoch()
            self.training_history.update_loss_in_epoch()
        return val_loss

    def _gen_prediction_and_loss(self, data: DatasetBaseCollatedType) -> torch.Tensor:
        x_actions = data["actions_padded"]
        x_cards = data["cards"]
        x_actions_mask = data["actions_mask"]
        y_target = data["expected_ev"]
        
        # For human data, convert EV to binary target (0 for negative EV, 1 for positive EV)
        if self.config.training_process["dataset"] == "Human":
            # Convert to binary target (0 for negative EV, 1 for positive EV)
            y_target = (y_target > 0).float()
        
        y_pred = self.model(x_actions, x_cards, x_actions_mask)
        
        # BCEWithLogitsLoss handles the sigmoid internally
        loss_in_batch = self.loss_fn(y_pred, y_target)
        return loss_in_batch

    def _build_optim(self) -> torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD:
        optim_str = self.config.training_process["optimizer"]
        
        # For human data fine-tuning with warm loading, only update OutputMLP parameters
        if self.config.training_process["dataset"] == "Human" and self.config.training_process["warm_start"]:
            params = self.model.moduleDict["output_mlp"].parameters()
        else:
            params = self.model.parameters()
            
        match optim_str:
            case "Adam":
                return torch.optim.Adam(
                    params=params,
                    lr=self.config.training_process["learning_rate"],
                    weight_decay=self.config.training_process["weight_decay"],
                )
            case "AdamW":
                return torch.optim.AdamW(
                    params=params,
                    lr=self.config.training_process["learning_rate"],
                    weight_decay=self.config.training_process["weight_decay"],
                )
            case "SGD":
                return torch.optim.SGD(
                    params=params,
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
        
        # Log data distribution
        if self.config.training_process["dataset"] == "Human":
            ev_values = torch.tensor(dataset.expected_ev_list)
            num_calls = (ev_values > 0).sum().item()
            num_folds = (ev_values <= 0).sum().item()
            total = len(ev_values)
            print(f"\nHuman Data Distribution:")
            print(f"Total hands: {total}")
            print(f"Calls (EV > 0): {num_calls} ({num_calls/total:.1%})")
            print(f"Folds (EV <= 0): {num_folds} ({num_folds/total:.1%})")
        
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
    Loss Function: {"MSE" if self.config.training_process["dataset"] == "GTO" else "BCEWithLogitsLoss"}
    Learning Rate: {self.config.training_process["learning_rate"]}
    Batch Size: {self.config.training_process["batch_size"]}
    Optimizer: {self.config.training_process["optimizer"]}
    Weight Decay: {self.config.training_process["weight_decay"]}
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
    Best validation loss: {self.training_history.best_validation_loss:.6f} at epoch {self.training_history.best_validation_epoch}
    Final training accuracy: {self.training_history.training_accuracy_in_epochs[-1]:.2%}
    Final validation accuracy: {self.training_history.validation_accuracy_in_epochs[-1]:.2%}
    Final training loss: {self.training_history.training_loss_in_epochs[-1]:.6f}
    Final validation loss: {self.training_history.validation_loss_in_epochs[-1]:.6f}
            """
        )

    def evaluate_on_test_set(self):
        """Evaluate model performance on the held-out test set"""
        self.model.eval()
        correct_decisions = 0
        total_decisions = 0
        
        # Create test results directory if it doesn't exist
        test_results_dir = self.base_path / "test_results" / self.config.general["save_checkpoint_name"]
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Open file to save test hands
        test_hands_file = test_results_dir / "test_hands.txt"
        with open(test_hands_file, "w") as f:
            with torch.no_grad():
                for batch_idx, data in enumerate(self.dataloader_test):
                    data = {k: v.to(self.config.general["device"]) for k, v in data.items()}
                    
                    # Get model predictions for the entire batch
                    predicted_ev = self.model(
                        data["actions_padded"],
                        data["cards"],
                        data["actions_mask"]
                    )
                    
                    # Get actual outcomes based on EV sign
                    actual_outcomes = data["expected_ev"] > 0
                    model_decisions = predicted_ev > 0
                    
                    # Process each hand in the batch
                    batch_size = data["actions_padded"].size(0)
                    for i in range(batch_size):
                        # Create a single hand dictionary for decoding
                        single_hand = {
                            "actions": data["actions_padded"][i],  # Changed from actions_padded to actions
                            "cards": data["cards"][i],  # Remove the extra dimension
                            "mask": data["actions_mask"][i] if data["actions_mask"] is not None else None
                        }
                        
                        try:
                            hand_str = EncodedHandHistory.decode_to_string(single_hand)
                            f.write(f"Hand {batch_idx * batch_size + i + 1}: {hand_str}\n")
                            f.write(f"Model Decision: {'Call' if predicted_ev[i] > 0 else 'Fold'}\n")
                            f.write(f"Actual Decision: {'Call' if actual_outcomes[i] > 0 else 'Fold'}\n")
                            f.write(f"Decision Correct: {model_decisions[i] == actual_outcomes[i]}\n")
                            f.write("-" * 50 + "\n")
                        except Exception as e:
                            print(f"Error decoding hand {batch_idx * batch_size + i + 1}: {str(e)}")
                            continue
                    
                    # Calculate correct decisions
                    correct_decisions += (
                        (model_decisions & actual_outcomes) |  # Correct calls
                        (~model_decisions & ~actual_outcomes)  # Correct folds
                    ).sum().item()
                    
                    total_decisions += len(predicted_ev)
                    
                    # Print progress
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Processed {(batch_idx + 1) * batch_size} hands...")
        
        accuracy = correct_decisions / total_decisions
        
        # Log final test accuracy to TensorBoard
        self.writer.add_scalar('Test/FinalAccuracy', accuracy, 0)
        
        print(f"\nTest hands have been saved to: {test_hands_file}")
        print(f"Final Test Accuracy: {accuracy:.2%}")
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
    arg_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="only run evaluation on test set without training",
    )
    args = arg_parser.parse_args()
    trainer = Trainer(args.config)
    
    if args.eval_only:
        print("Running evaluation only...")
        # Initialize model parameters from checkpoint
        trainer._init_params()
        test_accuracy = trainer.evaluate_on_test_set()
        print(f"Test Set Accuracy: {test_accuracy:.2%}")
    else:
        trainer.run()
