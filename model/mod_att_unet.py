from .unet_parts import *
from .attention.eca import EcaBlock
from .attention.cbam import SpatialAttention


class AttentionBlock(nn.Module):
    """注意力模块"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.w_l = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.channel_att = EcaBlock(in_channels // 2)
        self.spatial_att = SpatialAttention()

    def forward(self, xl, g):
        """xl是编码部分的输出，g是上采样后的输出"""
        xl_conv_out = self.w_l(xl)
        g_conv_out = self.w_g(g)
        xl_g_conv_sum = self.relu(xl_conv_out + g_conv_out)
        out = self.channel_att(xl_g_conv_sum)
        out = xl * out
        out = self.spatial_att(out)
        out = xl * out
        return out


class AttentionUp(nn.Module):
    """用于处理解码部分输入注意力模块前的上采样操作"""
    def __init__(self, in_channels, out_channels):
        super(AttentionUp, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class ModAttUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ModAttUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = False

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.att_up1 = AttentionUp(1024, 512)
        self.att1 = AttentionBlock(512)
        self.up1 = Up(1024, 512, bilinear)
        self.att_up2 = AttentionUp(512, 256)
        self.att2 = AttentionBlock(256)
        self.up2 = Up(512, 256, bilinear)
        self.att_up3 = AttentionUp(256, 128)
        self.att3 = AttentionBlock(128)
        self.up3 = Up(256, 128, bilinear)
        self.att_up4 = AttentionUp(128, 64)
        self.att4 = AttentionBlock(64)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)

        g1 = self.att_up1(x5)
        x4 = self.att1(xl=x4, g=g1)
        x = self.up1(x5, x4)

        g2 = self.att_up2(x)
        x3 = self.att2(xl=x3, g=g2)
        x = self.up2(x, x3)

        g3 = self.att_up3(x)
        x2 = self.att3(xl=x2, g=g3)
        x = self.up3(x, x2)

        g4 = self.att_up4(x)
        x1 = self.att4(xl=x1, g=g4)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    tensor = torch.randn((2, 13, 128, 128))
    att_unet = ModAttUnet(n_channels=13, n_classes=2)
    # print(att_unet)
    out_tensor = att_unet(tensor)
    print(out_tensor.shape)
