import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.full_model import FullModel
from models.model_config import ModelConfig
from utils.custom_torch_data_utils import (
    CollateFnForJSONDatasetType,
    JSONDataset,
    JSONDatasetType,
    JSONDatasetTypeCollated,
)


@dataclass
class TrainingHistory:
    training_batch_loss: list[float] = field(default_factory=lambda: [0.0])
    validation_batch_loss: list[float] = field(default_factory=lambda: [0.0])

    def _update_loss_in_batch(
        self,
        loss_in_batch: torch.Tensor,
        train_or_validate: Literal["Training", "Validation"],
    ):
        match train_or_validate:
            case "Training":
                self.training_batch_loss[-1] += loss_in_batch.item()
            case "Validation":
                self.training_batch_loss[-1] += loss_in_batch.item()

    def _update_loss_in_epoch(
        self,
        loss_in_batch: torch.Tensor,
        train_or_validate: Literal["Training", "Validation"],
    ):
        pass


class Trainer:
    config_filename: str | None
    config: ModelConfig
    optim: torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD
    model: FullModel
    loss_fn: nn.MSELoss
    dataloader_training: DataLoader[JSONDatasetType]
    dataloader_validation: DataLoader[JSONDatasetType]
    training_history: TrainingHistory

    def __init__(self, config_filename: str | None = None) -> None:
        self.config_filename = config_filename
        self.config = self._build_config()
        self.model = FullModel(config=self.config)
        self.optim = self._build_optim()
        self.loss_fn = nn.MSELoss()
        self.dataloader_training = self._build_dataloader(train_or_validate="Training")
        self.dataloader_validation = self._build_dataloader(
            train_or_validate="Validation"
        )
        self.training_history = TrainingHistory()

    def run(self):
        self._forward_n_epochs()

    def _forward_n_epochs(self):
        for _ in range(self.config.training_process["num_epochs"]):
            self._train_one_epoch()
            self._validate_one_epoch()

    def _train_one_epoch(self) -> None:
        _ = self.model.train()
        for _, data in enumerate(self.dataloader_training):
            data: JSONDatasetTypeCollated
            self.optim.zero_grad()
            loss_in_batch = self._gen_prediction_and_loss(data)
            _ = loss_in_batch.backward()
            _ = self.optim.step()

    def _validate_one_epoch(self) -> None:
        _ = self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(self.dataloader_validation):
                data: JSONDatasetTypeCollated
                loss_in_batch = self._gen_prediction_and_loss(data)

    def _gen_prediction_and_loss(self, data: JSONDatasetTypeCollated) -> torch.Tensor:
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
        config_dir = Path(__file__).parent.parent / "config"
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
                self.config.__getattribute__(base_key) | values,
            )
        return config

    def _build_dataloader(self, train_or_validate: Literal["Training", "Validation"]):
        dataset = JSONDataset(self.config, train_or_validate=train_or_validate)
        return DataLoader(
            dataset,
            batch_size=self.config.training_process["batch_size"],
            shuffle=True,
            num_workers=2,
            collate_fn=CollateFnForJSONDatasetType,
        )
