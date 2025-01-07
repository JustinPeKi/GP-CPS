import torch
import torch.nn as nn
import numpy as np



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            # 一般后面要接BN操作就设置bias=False，因为BN会使bias无效化
            nn.InstanceNorm2d(out_channels),  # 这里参数是指需要BN的通道数
            nn.ReLU(inplace=False),  # 参数表示relu会改变输入值
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        nn.init.normal_(self.double_conv[0].weight, mean=0.0, std=np.sqrt(2/(3*3*in_channels)))
        nn.init.normal_(self.double_conv[0].bias, mean=0.0, std=1.0)
        nn.init.normal_(self.double_conv[3].weight, mean=0, std=np.sqrt(2/(3*3*out_channels)))
        nn.init.normal_(self.double_conv[3].bias, mean=0.0, std=1.0)

    def forward(self, x):
        return self.double_conv(x)


class DownDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
            # 这里kernel_size到底应该等于多少，想不明白
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        nn.init.normal_(self.up[0].weight, mean=0, std=np.sqrt(2 / (2 * 2 * in_channels)))
        nn.init.normal_(self.up[0].bias)
        #nn.init.normal_(self.model[1].weight, mean=0, std=np.sqrt(2/(3*3*in_channels)))
        #nn.init.normal_(self.model[1].bias)

    def forward(self, x):
        return self.up(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.conv.weight, mean=0, std=np.sqrt(2/(1*1*in_channels)))
        nn.init.normal_(self.conv.bias)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.sigmoid(x1)
        return x2

class MyUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        n1 = 64
        channels = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.double_conv1 = DoubleConv(in_channels, channels[0])
        self.down_double_conv1 = DownDoubleConv(channels[0], channels[1])
        self.down_double_conv2 = DownDoubleConv(channels[1], channels[2])
        self.down_double_conv3 = DownDoubleConv(channels[2], channels[3])
        self.down_double_conv4 = DownDoubleConv(channels[3], channels[4])
        self.up1 = Up(channels[4], channels[3])
        self.double_conv2 = DoubleConv(channels[4], channels[3])
        self.up2 = Up(channels[3], channels[2])
        self.double_conv3 = DoubleConv(channels[3], channels[2])
        self.up3 = Up(channels[2], channels[1])
        self.double_conv4 = DoubleConv(channels[2], channels[1])
        self.up4 = Up(channels[1], channels[0])
        self.double_conv5 = DoubleConv(channels[1], channels[0])
        self.out = OutConv(channels[0], out_channels)

    def forward(self, x):
        x = x.float()
        x1 = self.double_conv1(x)
        x2 = self.down_double_conv1(x1)
        x3 = self.down_double_conv2(x2)
        x4 = self.down_double_conv3(x3)
        x5 = self.down_double_conv4(x4)
        x6 = self.up1(x5)
        x7 = torch.cat((x4,x6), dim=1)
        x8 = self.double_conv2(x7)
        x9 = self.up2(x8)
        x10 = torch.cat((x3, x9), dim=1)
        x11 = self.double_conv3(x10)
        x12 = self.up3(x11)
        x13 = torch.cat((x2, x12), dim=1)
        x14 = self.double_conv4(x13)
        x15 = self.up4(x14)
        x16 = torch.cat((x1, x15), dim=1)
        x17 = self.double_conv5(x16)
        x18 = self.out(x17)

        return x18, x5
