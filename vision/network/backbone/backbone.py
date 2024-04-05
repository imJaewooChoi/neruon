from torchinfo import summary
from .resnet import ResNet
from .resnext import ResNeXt
from utils import weight_initialize, load_yaml


class BackBone:
    def __init__(self, config):
        self.config = config

    def build(self):
        if self.config["Model"]["name"] == "resnet50":
            config = load_yaml(
                "/workspace/neruon/vision/configs/backbone/resnet50.yaml"
            )
            model = ResNet(config)

        elif self.config["Model"]["name"] == "resnet101":
            config = load_yaml(
                "/workspace/neruon/vision/configs/backbone/resnet101.yaml"
            )
            model = ResNet(config)

        elif self.config["Model"]["name"] == "resnet152":
            config = load_yaml(
                "/workspace/neruon/vision/configs/backbone/resnet150.yaml"
            )
            model = ResNet(config)
        elif self.config["Model"]["name"] == "resnext50":
            config = load_yaml(
                "/workspace/neruon/vision/configs/backbone/resnext50.yaml"
            )
            model = ResNeXt(config)

        summary(model, input_size=(1, 3, 224, 224))

        return model.apply(weight_initialize)
