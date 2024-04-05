from engine import Trainer, DataModule, LitModule
from network.backbone import BackBone
from utils import load_yaml


def main():

    config_path = "/workspace/neruon/vision/configs/classification.yaml"
    config = load_yaml(config_path)
    backbone = BackBone(config)
    backbone = backbone.build()
    datamodule = DataModule(config)
    lit_module = LitModule(backbone, config)

    train = Trainer(
        model=backbone, litmodule=lit_module, datamodule=datamodule, config=config
    )
    train.run()


if __name__ == "__main__":
    main()
