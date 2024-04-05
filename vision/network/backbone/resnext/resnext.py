import torch.nn as nn
from module import BasicStem, BasicHead
from .body import ResNeXtBody


class ResNeXt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stem = BasicStem(
            config["Stem"]["in_channel"], config["Stem"]["out_channel"]
        )
        self.body = ResNeXtBody(
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
    onnx_path = "/workspace/neruon/vision/network/backbone/resnext/resnext50.onnx"
    with open("/workspace/neruon/vision/configs/backbone/resnext50.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dummy_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)
    resnext = ResNeXt(config).to(DEVICE)
    output_tensor = resnext(dummy_tensor)
    print("Output Tensor Shape: ", output_tensor.shape)
    summary(resnext, input_size=(1, 3, 224, 224))

    export_onnx(resnext, dummy_tensor, onnx_path)
