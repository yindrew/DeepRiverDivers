from torch.utils.data import DataLoader

from models.full_model import FullModel
from models.model_config import ModelConfig
from utils.custom_torch_data_utils import (
    CollateFnForJSONDatasetType,
    JSONDatasetBase,
    JSONDatasetType,
)


class TestCustomTorchDatasets:
    def test_dataloader_and_forward_on_base_data(self):
        # NOTE: Need to update
        config = ModelConfig()
        config.training_process["batch_size"] = 16
        dataset = JSONDatasetBase(config)
        # dataloader: DataLoader[JSONDatasetType] = DataLoader(
        #     dataset,
        #     batch_size=config.training_process["batch_size"],
        #     shuffle=True,
        #     num_workers=2,
        #     collate_fn=CollateFnForJSONDatasetType,
        # )

        # # test if forward works
        # model = FullModel(config=config)
        # for i_batch, sample_batched in enumerate(dataloader):
        #     if i_batch == 0:
        #         print({k: v.shape for k, v in sample_batched.items()})
        #     [expected_ev, x_cards, x_actions, x_actions_mask] = [
        #         sample_batched[k]
        #         for k in ("expected_ev", "cards", "actions_padded", "actions_mask")
        #     ]
        #     model.forward(x_actions, x_cards, x_actions_mask)
