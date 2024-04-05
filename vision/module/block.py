import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv


class BasicStem(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = Conv(c1, c2, kernel_size=7, stride=2, padding=3, act=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.conv1(x))


class BasicHead(nn.Module):
    def __init__(self, in_channles, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channles, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, expansion=4):
        super().__init__()
        c3 = expansion * c2
        self.conv1 = Conv(c1, c2, kernel_size=1, act=True)
        self.conv2 = Conv(c2, c2, kernel_size=3, stride=stride, act=True)
        self.conv3 = Conv(c2, c3, kernel_size=1, act=False)
        self.projection = (
            nn.Sequential(Conv(c1, c3, kernel_size=1, stride=stride, act=False))
            if stride != 1 or c1 != c3
            else nn.Identity()
        )

    def forward(self, x):
        projection = self.projection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return F.relu(x + projection)


class ResNeXtBlock(ResNetBlock):
    def __init__(self, c1, c2, stride=1, cardinality=32, expansion=2):
        super().__init__(c1, c2, stride, expansion)
        self.conv2 = Conv(
            c2, c2, kernel_size=3, stride=stride, groups=cardinality, act=True
        )


if __name__ == "__main__":
    c1 = 64
    c2 = 64
    block = ResNetBlock(c1, c2)
    print(block)
