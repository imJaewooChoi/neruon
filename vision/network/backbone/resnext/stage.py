import torch.nn as nn
from module import ResNeXtBlock


class ResNeXtStage(nn.Module):
    def __init__(self, c1, c2, stride=1, num_blocks=1, expansion=2, cardinality=32):
        super().__init__()

        blocks = [
            ResNeXtBlock(c1, c2, stride, cardinality=cardinality, expansion=expansion)
        ]

        blocks.extend(
            [
                ResNeXtBlock(
                    expansion * c2, c2, 1, cardinality=cardinality, expansion=expansion
                )
                for _ in range(num_blocks - 1)
            ]
        )
        self.stage = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stage(x)


if __name__ == "__main__":

    import torch
    from torchinfo import summary
    from utils import export_onnx, set_device

    DEVICE = set_device()
    onnx_path = "/workspace/neruon/vision/network/backbone/resnext/resnext_stage.onnx"

    c1 = 128
    c2 = 128
    num_blocks = 3
    dummy_tensor = torch.randn(1, 128, 56, 56).to(DEVICE)

    stage = ResNeXtStage(c1, c2, num_blocks=num_blocks).to(DEVICE)

    output_tensor = stage(dummy_tensor)
    print("Output Tensor Shape: ", output_tensor.shape)
    summary(stage, input_size=(1, 128, 56, 56))

    export_onnx(stage, dummy_tensor, onnx_path)
