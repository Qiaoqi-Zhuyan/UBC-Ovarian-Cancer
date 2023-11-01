import torch.nn as nn
import math
import einops

class SEModule(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=4):
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(out_channel, in_channel // reduction, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.se(x)
        y = self.sigmoid(y).view(b, c, 1, 1)

        return x * y




class Conv_3x3_bn(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Conv_3x3_bn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


    def forward(self, x):
        return self.conv_bn(x)


class Conv_1x1_bn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_1x1_bn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.conv_bn(x)


class MBConv(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, downsample, expansion=4):
        super(MBConv, self).__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(in_channel * expansion)

        if self.downsample:
            self.max_pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channel, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                #pw
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channel, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                #dw
                nn.Conv2d(hidden_dim, out_channel, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                SEModule(in_channel, hidden_dim),

                # linear
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        self.norm = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.max_pool(x)) + self.conv(self.norm(x))
        else:
            return x + self.conv(self.norm(x))

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion=4):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(in_channel * expansion)
        self.use_res_connect = self.stride == 1 and in_channel == out_channel

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channel, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # pw
                nn.Conv2d(self.hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv = nn.Sequential(
                #pw
                nn.Conv2d(in_channel, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                #pw-linear

            )