import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        litmodule: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        config,
    ):
        self.config = config
        self.litmodule = litmodule
        self.datamodule = datamodule

    def _set_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            save_top_k=1,
            dirpath=self.config["Logger"]["dir"] + "/" + self.config["Model"]["name"],
            filename="{epoch:03d}-{train_loss:.4f}-{train_f1:.4f}-{valid_loss:.4f}-{valid_f1:.4f}",
            verbose=True,
        )
        rich_model_summary = RichProgressBar()

        learningrate_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_callback, rich_model_summary, learningrate_monitor]

    def _set_loggers(self):

        wandblogger = WandbLogger(
            project=self.config["Logger"]["project"],
            name=self.config["Model"]["name"],
            save_dir=self.config["Logger"]["dir"],
        )
        return [wandblogger]

    def _set_trainer(self):
        callbacks = self._set_callbacks()
        loggers = self._set_loggers()
        trainer = pl.Trainer(
            max_epochs=self.config["Train"]["epoch"],
            accelerator=self.config["Train"]["device"],
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=1,
        )
        return trainer

    def run(self):
        pl.seed_everything(self.config["seed"])
        self.datamodule.setup()
        trainer = self._set_trainer()
        trainer.fit(self.litmodule, self.datamodule)
        trainer.test(self.litmodule, self.datamodule)
