import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, in_channels=13, n_classes=2):
        super(SegNet, self).__init__()
        self.n_classes = n_classes
        self.enconv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enconv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enconv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enconv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.enconv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.enconv1(x)
        out, idx1 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv2(out)
        out, idx2 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv3(out)
        out, idx3 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv4(out)
        out, idx4 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv5(out)
        out, idx5 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = F.max_unpool2d(out, indices=idx5, kernel_size=2, stride=2)
        out = self.deconv1(out)
        out = F.max_unpool2d(out, indices=idx4, kernel_size=2, stride=2)
        out = self.deconv2(out)
        out = F.max_unpool2d(out, indices=idx3, kernel_size=2, stride=2)
        out = self.deconv3(out)
        out = F.max_unpool2d(out, indices=idx2, kernel_size=2, stride=2)
        out = self.deconv4(out)
        out = F.max_unpool2d(out, indices=idx1, kernel_size=2, stride=2)
        out = self.deconv5(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    segnet = SegNet()
    x = torch.randn((2, 13, 128, 128), dtype=torch.float32)
    y = segnet(x)
    print(y.shape)
