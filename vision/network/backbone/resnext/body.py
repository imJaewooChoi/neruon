import torch.nn as nn
from typing import List
from .stage import ResNeXtStage


class ResNeXtBody(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels_list: List[int],
        num_blocks_list: List[int],
    ):
        super().__init__()

        self.body = nn.ModuleList()

        for index in range(len(in_channels_list)):
            stride = 1 if index == 0 else 2
            self.body.append(
                ResNeXtStage(
                    in_channels_list[index],
                    out_channels_list[index],
                    stride,
                    num_blocks_list[index],
                )
            )

    def forward(self, x):
        for stage in self.body:
            x = stage(x)
        return x
