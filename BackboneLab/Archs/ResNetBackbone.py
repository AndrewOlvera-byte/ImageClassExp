import torch
import torch.nn as nn
from .BasicBlock import BasicBlock

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Initial convolution layer (no big 7x7 conv or maxpool, since images are small)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)
        # Stage 1: 16 -> 16 channels
        self.layer1 = nn.Sequential(
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1)
        )
        # Stage 2: 16 -> 32 channels (first block with stride 2 to downsample)
        self.layer2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1)
        )
        # Stage 3: 32 -> 64 channels (downsample)
        self.layer3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1)
        )
        # Global average pool to get 1x1 output per channel
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 64  # after global pool, feature vector length

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)          # output shape (batch, 64, 1, 1)
        x = torch.flatten(x, 1)         # flatten to (batch, 64)
        return x

