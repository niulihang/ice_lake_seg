import math

import torch
import torch.nn as nn

# ECA-Net改进了SENet


class EcaBlock(nn.Module):
    """ECA-Net注意力模块"""
    def __init__(self, in_channels, b=1, gamma=2):
        super(EcaBlock, self).__init__()
        # |(log2(channel) + b) / gamma|
        kernel_size = int(abs((math.log(in_channels, 2) + 2) / 2))
        # kernel_size取最近的奇数
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = (kernel_size - 1) // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # print(f'eca x shape {x.shape}')
        avg_pool_out = self.avg_pool(x).view(b, 1, c)
        # print(f'eca avg pool shape {avg_pool_out.shape}')
        out = self.conv(avg_pool_out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        # print(f'eca out shape {out.shape}')
        return out * x


if __name__ == '__main__':
    tensor = torch.randn((2, 64, 128, 128))
    print(tensor.shape)
    cbam_blk = EcaBlock(64)
    print(cbam_blk)
    out_tensor = cbam_blk(tensor)
    print(out_tensor.shape)

