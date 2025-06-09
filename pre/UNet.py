import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """基本残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)

class UNet(nn.Module):
    def __init__(self, in_channels=7, base_channels=128):
        super().__init__()
        # 编码器
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)

        # 低维表示
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4)

        # 解码器
        self.dec3 = ResidualBlock(base_channels * 4, base_channels * 2)
        self.dec2 = ResidualBlock(base_channels * 2, base_channels)
        self.dec1 = ResidualBlock(base_channels, base_channels)

        # 输出层
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.enc3(F.avg_pool2d(x2, 2))

        x = self.bottleneck(x3)
        x = x + x3
        x = self.dec3(x)
        x = F.interpolate(x, scale_factor=2) + x2
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=2) + x1
        x = self.dec1(x)

        return self.out_conv(x)
