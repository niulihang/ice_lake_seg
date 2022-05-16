import torch
import torch.nn as nn


class SenetBlock(nn.Module):
    """SENet注意力机制模块"""
    def __init__(self, in_channels, ratio=16):
        super(SenetBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        scale = self.fc(avg).view(b, c, 1, 1)
        return x * scale


if __name__ == '__main__':
    tensor = torch.randn((1, 13, 128, 128))
    print(tensor.shape)
    senet_blk = SenetBlock(tensor.shape[1])
    out_tensor = senet_blk(tensor)
    print(out_tensor.shape)
