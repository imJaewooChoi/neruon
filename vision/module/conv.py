import torch.nn as nn


def auto_padding(kernel, padding=None, dilation=1):
    if dilation > 1:
        kernel = (
            dilation * (kernel - 1) + 1
            if isinstance(kernel, int)
            else [dilation * (x - 1) + 1 for x in kernel]
        )
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding


class Conv(nn.Module):
    default_activation = nn.ReLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        dilation: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_padding(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_activation
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
