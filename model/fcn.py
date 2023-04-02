import torch
import torch.nn as nn
import numpy as np



class VGG_fcn32s(nn.Module):
    '''
    将 VGG model 改变成 FCN-32s
    '''

    def __init__(self, n_classes=2, in_channels=13):
        super(VGG_fcn32s, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=100)  # padding = 100，传统 VGG 为1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])  # 第一组 Stage
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 降采样 /2
        self.layer2 = Layer(64, [128, 128])  # 第二组 Stage
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 降采样 /4
        self.layer3 = Layer(128, [256, 256, 256, 256])  # 第三组 Stage
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 降采样 /8
        self.layer4 = Layer(256, [512, 512, 512, 512])  # 第四组 Stage
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 降采样 /16
        self.layer5 = Layer(512, [512, 512, 512, 512])  # 第五组 Stage
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 降采样 /32

        # modify to be compatible with segmentation and classification
        # self.fc6 = nn.Linear(512*7*7, 4096) # 全连接层 VGG
        self.fc6 = nn.Conv2d(512, 4096, 7)  # padding = 0
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        # self.fc7 = nn.Linear(4096, 4096) # 全连接层 VGG
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()

        # self.score = nn.Linear(4096, n_class) # 全连接层 VGG
        self.score = nn.Conv2d(4096, n_classes, 1)

        self.upscore = nn.ConvTranspose2d(n_classes, n_classes, 64, 32)  # 上采样 32 倍

    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        # f5 = f5.view(f5.size(0), -1)
        print('f5.shape:', f5.shape)
        f6 = self.drop6(self.relu6(self.fc6(f5)))
        print('f6.shape:', f6.shape)
        f7 = self.drop7(self.relu7(self.fc7(f6)))
        print('f7.shape:', f7.shape)
        score = self.score(f7)
        upscore = self.upscore(score)
        # crop 19 再相加融合 [batchsize, channel, H, W ] 要对 H、W 维度裁剪
        upscore = upscore[:, :, 19:19 + x.size(2), 19:19 + x.size(3)].contiguous()
        return upscore

# Block 包含：conv-bn-relu
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


# 建立 layer 加入很多 Block
def make_layers(in_channels, layer_list):
    layers = []
    for out_channels in layer_list:
        layers += [Block(in_channels, out_channels)]
        in_channels = out_channels
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out


if __name__ == '__main__':
    fcn = VGG_fcn32s()
    x = torch.randn((2, 13, 128, 128), dtype=torch.float32)
    y = fcn(x)
    print(y.shape)