import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import CustomDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CustomDataset(self.hparams, "train")
            self.valid_dataset = CustomDataset(self.hparams, "valid")
        if stage == "test" or stage is None:
            self.test_dataset = CustomDataset(self.hparams, "test")

    def train_dataloader(self):
        return self._shared_loader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._shared_loader(self.valid_dataset, "valid")

    def test_dataloader(self):
        return self._shared_loader(self.test_dataset, "test")

    def _shared_loader(self, dataset, phase):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams["Data"]["batch_size"],
            num_workers=torch.cuda.device_count() * 4,
            shuffle=True if phase == "train" else False,
            pin_memory=True,
        )
