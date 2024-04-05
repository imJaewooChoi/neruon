import torch


class Helper:
    def __init__(self, config):
        self.config = config["Helper"]

    def loss_fn(self):
        if self.config["loss"] == "cross_entropy":
            return torch.nn.CrossEntropyLoss()

    def optimizer(self, model):
        if self.config["optimizer"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )

    def scheduler(self, optimizer):
        if self.config["scheduler"] == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
        elif self.config["scheduler"] == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        elif self.config["scheduler"] == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=1e-6, last_epoch=-1
            )
        elif self.config["scheduler"] == "cosine_warmup":
            from utils import CosineAnnealingWarmUpRestarts

            return CosineAnnealingWarmUpRestarts(
                optimizer,
                T_0=100,
                T_mult=1,
                eta_max=0.1,
                T_up=10,
                gamma=0.5,
                last_epoch=-1,
            )
        elif self.config["scheduler"] == "none":
            return None
