import torch.nn as nn
from module import BasicStem, BasicHead
from .body import ResNetBody


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stem = BasicStem(
            config["Stem"]["in_channel"], config["Stem"]["out_channel"]
        )
        self.body = ResNetBody(
            config["Body"]["in_channel"],
            config["Body"]["out_channel"],
            config["Body"]["num_blocks"],
        )
        self.head = BasicHead(
            config["Head"]["in_channel"], config["Head"]["out_channel"]
        )

    def forward(self, x):
        stem_tensor = self.stem(x)
        body_tensor = self.body(stem_tensor)
        head_tensor = self.head(body_tensor)
        return head_tensor


if __name__ == "__main__":
    import yaml
    import torch
    from torchinfo import summary
    from utils import export_onnx, set_device

    DEVICE = set_device()
    onnx_path = "/workspace/neruon/vision/network/backbone/resnet/resnet.onnx"
    with open("/workspace/neruo/vision/configs/backbone/resnet101.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dummy_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)
    resnet = ResNet(config).to(DEVICE)

    summary(resnet, input_size=(1, 3, 224, 224))

    export_onnx(resnet, dummy_tensor, onnx_path)
