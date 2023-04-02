import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, in_channels, ration=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ration, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // ration),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ration, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out = self.fc(self.max_pool(x))
        avg_pool_out = self.fc(self.avg_pool(x))
        max_avg_sum = max_pool_out + avg_pool_out
        return self.sigmoid(max_avg_sum)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f'x shape {x.shape}')
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(f'max pool shape {max_pool_out.shape}')
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        # print(f'avg pool shape {avg_pool_out.shape}')
        max_avg_cat = torch.cat((max_pool_out, avg_pool_out), dim=1)
        # print(f'max_avg_cat shape {max_avg_cat.shape}')
        out = self.conv(max_avg_cat)
        # print(f'after conv out shape {out.shape}')
        return self.sigmoid(out)


class CbamBlock(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CbamBlock, self).__init__()
        self.cha_att = ChannelAttention(in_channels, ratio)
        self.spa_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.cha_att(x)
        x = x * self.spa_att(x)
        return x


if __name__ == '__main__':
    tensor = torch.randn((2, 64, 128, 128))
    print(tensor.shape)
    cbam_blk = CbamBlock(64)
    # cbam_blk = ChannelAttention(64)
    out_tensor = cbam_blk(tensor)
    print(out_tensor.shape)
