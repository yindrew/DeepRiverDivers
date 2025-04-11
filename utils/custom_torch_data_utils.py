from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, override

import torch
from torch.utils.data import Dataset

from models.encoded_handhistory import EncodedHandHistory, EncodedHandHistoryType
from models.model_config import ModelConfig
from utils.create_mask_and_pad import MaskerPadder


class JSONDatasetType(TypedDict):
    encoded_hand_history: EncodedHandHistoryType
    expected_ev: float


class JSONDatasetTypeCollated(TypedDict):
    expected_ev: torch.Tensor
    cards: torch.LongTensor
    actions_padded: torch.LongTensor
    actions_mask: torch.BoolTensor


class JSONDatasetBase(Dataset[JSONDatasetType]):
    """
    torch.utils.Dataset for loading json files, assuming each
    json contains one action and on hand seq (like hand1.json).

    This basic one loads from the same file 500 times.
    torch.utils.data has other variants of Dataset classes.

    Attributes:
        root_dir (pathlib.Path): root directory for json files
        expected_ev_list (list[float]): list of "ground truth" expected values
        training_files_list (list[str]): list of training file names
    """

    root_dir: Path
    expected_ev_list: list[float]
    training_files_list: list[str]

    def __init__(self, config: ModelConfig = ModelConfig()):
        self.root_dir = Path(__file__).parent.parent / "tests" / "sample-hand"
        self.expected_ev_list = [
            -0.42069 + i * 0.002 for i in range(500)
        ]  # should depend on config (which phase of training)
        self.training_files_list = ["hand1.json"] * 500

    def __len__(self):
        return len(self.expected_ev_list)

    @override
    def __getitem__(self, idx: int) -> JSONDatasetType:
        json_path = str(self.root_dir / self.training_files_list[idx])
        return {
            "encoded_hand_history": EncodedHandHistory.from_json(json_path),
            "expected_ev": self.expected_ev_list[idx],
        }


def CollateFnForJSONDatasetType(
    xx: Sequence[JSONDatasetType],
) -> JSONDatasetTypeCollated:
    """
    util function for collating JSONDatasetType outputs
    useful for combining sequence of JSONDatasetType
    to be passed into torch.utils.data.DataLoader
    """
    masker_padder = MaskerPadder()
    actions_mask, actions_padded = masker_padder(
        [x["encoded_hand_history"]["actions"] for x in xx]
    )
    return {
        "expected_ev": torch.Tensor([x["expected_ev"] for x in xx]),
        "cards": torch.stack([x["encoded_hand_history"]["cards"] for x in xx], dim=0),
        "actions_padded": actions_padded,
        "actions_mask": actions_mask,
    }
