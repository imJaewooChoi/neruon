import torch
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score
from .helper import Helper


class LitModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.helper = Helper(self.hparams)
        self.model = model
        self.criterion = self.helper.loss_fn()
        self.optimizer = self.helper.optimizer(self.model)
        self.scheduler = self.helper.scheduler(self.optimizer)
        self.train_accurcay = Accuracy(
            task="multiclass", num_classes=self.hparams["num_classes"]
        )
        self.train_f1 = F1Score(
            task="multiclass", num_classes=self.hparams["num_classes"]
        )

        self.valid_accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams["num_classes"]
        )
        self.valid_f1 = F1Score(
            task="multiclass", num_classes=self.hparams["num_classes"]
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams["num_classes"]
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=self.hparams["num_classes"]
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = {
            "scheduler": self.scheduler,
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss, labels, predict_labels = self._shared_step(batch)
        self.train_accurcay(predict_labels, labels)
        self.train_f1(predict_labels, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(
            "train_acc",
            self.train_accurcay,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predict_labels = self._shared_step(batch)
        self.valid_accuracy(predict_labels, labels)
        self.valid_f1(predict_labels, labels)
        self.log("valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(
            "valid_acc",
            self.valid_accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "valid_f1",
            self.valid_f1,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        _, labels, predict_labels = self._shared_step(batch)
        self.test_accuracy(predict_labels, labels)
        self.test_f1(predict_labels, labels)
        self.log(
            "test_acc", self.test_accuracy, on_epoch=True, on_step=False, prog_bar=True
        )
        self.log("test_f1", self.test_f1, on_epoch=True, on_step=False, prog_bar=True)

    def _shared_step(self, batch):
        images, labels = batch
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        predict_labels = torch.argmax(logits, dim=-1)
        return loss, labels, predict_labels
