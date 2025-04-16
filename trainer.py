import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

# add git repo base path to sys.path so python3 {path to this file}
# can correctly import models
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.full_model import FullModel
from schemas.checkpoint_dict import CheckpointDict
from schemas.dataloader_datatypes import DatasetBaseCollatedType, DatasetBaseType
from schemas.model_config import ModelConfig
from utils.custom_torch_data_utils import (
    GTODataset,
    HumanDataset,
    collate_fn_datasetbasetype,
    train_test_split,
)
from utils.training_history import TrainingHistory

torch.serialization.add_safe_globals([TrainingHistory, ModelConfig])


class Trainer:
    config_filename: str | None
    config: ModelConfig
    optim: torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD
    model: FullModel
    loss_fn: nn.MSELoss
    dataloader_training: DataLoader[DatasetBaseType]
    dataloader_validation: DataLoader[DatasetBaseType]
    training_history: TrainingHistory
    base_path: Path
    start_time: float

    def __init__(self, config_filename: str | None = None) -> None:
        self.start_time = time.time()
        self.base_path = Path(__file__).parent
        self.config_filename = config_filename
        self.config = self._build_config()
        self._init_seed()
        self._set_device()
        self.model = FullModel(config=self.config)
        self.optim = self._build_optim()
        self.loss_fn = nn.MSELoss()
        self.dataloader_training, self.dataloader_validation = self._build_dataloaders()
        self.training_history = TrainingHistory()

    def run(self):
        self._init_params()
        self._log_training_setup()
        self._forward_n_epochs()
        self._log_training_result()

    def _init_seed(self) -> None:
        _ = torch.manual_seed(self.config.general["seed"])

    def _set_device(self) -> None:
        _ = torch.set_default_device(self.config.general["device"])

    def _init_params(self) -> None:
        if self.config.training_process["warm_start"]:
            (
                previous_model_state_dict,
                previous_optimizer_state_dict,
                previous_training_history,
            ) = self._load_checkpoint()
            self.model.load_state_dict(previous_model_state_dict)
            self.optim.load_state_dict(previous_optimizer_state_dict)
            self.training_history = previous_training_history

    def _forward_n_epochs(self):
        for _ in range(self.config.training_process["num_epochs"]):
            self._train_one_epoch()
            self._validate_one_epoch()
        self._save_checkpoint(postfix="final")

    def _train_one_epoch(self) -> None:
        _ = self.model.train()
        for _, data in enumerate(self.dataloader_training):
            data: DatasetBaseCollatedType
            data = {k: v.to(self.config.general["device"]) for k, v in data.items()}
            self.optim.zero_grad()
            loss_in_batch = self._gen_prediction_and_loss(data) / len(
                self.dataloader_training
            )
            _ = loss_in_batch.backward()
            _ = self.optim.step()
            self.training_history.update_loss_in_batch(loss_in_batch)
        self.training_history.log_loss_in_epoch()
        self.training_history.update_loss_in_epoch()

    def _validate_one_epoch(self) -> None:
        _ = self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(self.dataloader_validation):
                data: DatasetBaseCollatedType
                data = {k: v.to(self.config.general["device"]) for k, v in data.items()}
                loss_in_batch = self._gen_prediction_and_loss(data) / len(
                    self.dataloader_validation
                )
                self.training_history.update_loss_in_batch(loss_in_batch)
            self.training_history.log_loss_in_epoch()
            self.training_history.update_loss_in_epoch()

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
            config.__setattr__(
                base_key,
                config.__getattribute__(base_key) | values,
            )
        return config

    def _build_dataloaders(self):
        dataset = {"GTO": GTODataset, "Human": HumanDataset}[
            self.config.training_process["dataset"]
        ]()
        return tuple(
            DataLoader(
                dataset,
                batch_size=self.config.training_process["batch_size"],
                generator=torch.Generator(device=self.config.general["device"]),
                shuffle=True,
                num_workers=0,  # hardcoded, otherwise would sometimes throw errors
                collate_fn=collate_fn_datasetbasetype,
            )
            for dataset_subset in train_test_split(
                dataset,
                device=self.config.general["device"],
                p_train_test_split=self.config.training_process["p_train_test_split"],
            )
        )

    def _load_checkpoint(self):
        checkpoint_path = (
            self.base_path
            / "checkpoints"
            / self.config.general["checkpoint_name"]
            / "final.pth"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No previous checkpoints found.")
        checkpoint_dict: CheckpointDict = torch.load(
            checkpoint_path, map_location=self.config.general["device"]
        )
        (
            previous_model_state_dict,
            previous_optimizer_state_dict,
            previous_training_history,
            previous_model_config,  # not returned
        ) = (
            checkpoint_dict["model_state_dict"],
            checkpoint_dict["optimizer_state_dict"],
            checkpoint_dict["loss_history"],
            checkpoint_dict["model_config"],
        )
        return (
            previous_model_state_dict,
            previous_optimizer_state_dict,
            previous_training_history,
        )

    def _save_checkpoint(self, postfix: Literal["best", "final"] = "final") -> None:
        checkpoint_dir = (
            self.base_path / "checkpoints" / self.config.general["checkpoint_name"]
        )
        checkpoint_path = checkpoint_dir / f"{postfix}.pth"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_path)
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
        (if set to true, loads from [checkpoint_name]__final.pth, where checkpoint_name={self.config.general["checkpoint_name"]})
        """
        )

    def _log_training_result(self):
        print(
            f"""
Training result:
    Ellapsed time: {time.time() - self.start_time} seconds
    Best validation loss: {min(self.training_history.validation_loss_in_epochs)}

Models (best and final) saved in ./checkpoints/{self.config.general["checkpoint_name"]} directory.
            """
        )


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
