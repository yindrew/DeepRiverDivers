from collections.abc import Sequence
from math import floor
from pathlib import Path
from typing import Literal, override

import polars as pl
import torch
from torch.utils.data import Dataset, Subset

from models.encoded_handhistory import EncodedHandHistory
from schemas.dataloader_datatypes import DatasetBaseCollatedType, DatasetBaseType
from schemas.hand_history import EncodedHandHistoryType
from utils.gto_hand_parser import GTOHandParser
from utils.mask_and_pad import MaskerPadder


def collate_fn_datasetbasetype(
    xx: Sequence[DatasetBaseType],
) -> DatasetBaseCollatedType:
    """
    util function for collating DatasetBaseType outputs
    useful for combining sequence of DatasetBaseType
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


class HumanDataset(Dataset[DatasetBaseType]):
    """
    torch.utils.Dataset for loading json files, assuming each
    json contains one action and on hand seq (like hand1.json).

    This class utilizes the data directory (in this repository).

    Input:
        train_or_validate (Literal["Training", "Validation"])

    Attributes:
        root_dir (pathlib.Path): root directory for json files
        expected_ev_list (list[float]): list of "ground truth" expected values
        training_files_list (list[str]): list of training file names
    """

    root_dir: Path
    expected_ev_list: list[float]
    training_files_list: list[str]

    def __init__(self):
        self.root_dir = Path(__file__).parent.parent / "data" / "human"
        df: pl.DataFrame = pl.read_csv(
            source=str(self.root_dir / "metadata.csv"),
            separator=",",
            has_header=True,
            quote_char='"',
            schema=pl.Schema(
                {
                    "filename": pl.String,
                    "ev": pl.Float64,
                }
            ),
        )
        self.expected_ev_list = df["ev"].to_list()
        self.training_files_list = df["filename"].to_list()

    def __len__(self):
        return len(self.expected_ev_list)

    @override
    def __getitem__(self, idx: int) -> DatasetBaseType:
        json_path = str(self.root_dir / self.training_files_list[idx])
        return {
            "encoded_hand_history": EncodedHandHistory.from_json(json_path),
            "expected_ev": self.expected_ev_list[idx],
        }


class GTODataset(Dataset[DatasetBaseType]):
    """Dataset class for poker hand histories."""

    root_dir: Path
    hand_histories: list[DatasetBaseType]
    expected_ev_list: list[float]
    encoded_hands: list[EncodedHandHistoryType]

    def __init__(self):
        """
        Initialize the dataset.

        Args:
            hand_histories: list of (encoded_hand_dict, expected_value) tuples
        """
        parser = GTOHandParser()
        self.root_dir = Path(__file__).parent.parent / "data" / "gto"
        self.hand_histories = []
        for txt_file in self.root_dir.glob("*.txt"):
            self.hand_histories.extend(
                parser.parse_hand_file(str(self.root_dir / txt_file))
            )

        # The data is already encoded, so we just need to extract it
        self.expected_ev_list = [
            hand_history["expected_ev"] for hand_history in self.hand_histories
        ]
        self.encoded_hands = [
            hand_history["encoded_hand_history"] for hand_history in self.hand_histories
        ]

    def __len__(self) -> int:
        return len(self.hand_histories)

    @override
    def __getitem__(self, idx: int) -> DatasetBaseType:
        return {
            "encoded_hand_history": self.encoded_hands[idx],
            "expected_ev": self.expected_ev_list[idx],
        }


def train_test_split(
    dataset: Dataset[DatasetBaseType],
    device: Literal["cpu", "cuda"],
    p_train_test_split: float = 0.1,
) -> tuple[Dataset[DatasetBaseType], Dataset[DatasetBaseType]]:
    tot_len = len(dataset)
    cut_off = floor(tot_len * p_train_test_split)
    perm = torch.randperm(tot_len)
    train_inds = perm[cut_off:]
    test_inds = perm[:cut_off]
    return Subset(dataset, train_inds), Subset(dataset, test_inds)
