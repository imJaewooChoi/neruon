from torchinfo import summary
from .resnet import ResNet
from .resnext import ResNeXt
from utils import weight_initialize, load_yaml


class BackBone:
    def __init__(self, config):
        self.config = config
        self.model_name = config["Model"]["name"]
        self.model_config = load_yaml(
            f"/workspace/neuron/vision/configs/backbone/{self.model_name}.yaml"
        )

    def build(self):
        if self.model_name == "resnet50":
            model = ResNet(self.model_config)
        elif self.model_name == "resnet101":
            model = ResNet(self.model_config)
        elif self.model_name == "resnet152":
            model = ResNet(self.model_config)
        elif self.model_name == "resnext50":
            model = ResNeXt(self.model_config)
        elif self.model_name == "resnext101":
            model = ResNeXt(self.model_config)
        elif self.model_name == "resnext152":
            model = ResNeXt(self.model_config)
        summary(model, input_size=(1, 3, 224, 224))

        return model.apply(weight_initialize)
