import torch
import torch.nn as nn
import numpy as np

class conv_3x3_bn(nn.Module):
    def __init__(self, input, output, img_size, downsample=False):
        super(conv_3x3_bn, self).__init__()
        self.stride = 1 if downsample == False else 2
        self.conv2d = nn.Conv2d(input, output, 3, stride=self.stride, bias=False)
        self.bn2d = nn.BatchNorm2d(output)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.gelu(x)

        return x


class pre_norm(nn.Module):
    def __init__(self, dim, fn, norm):
        super.__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, input, output, expansion=0.25):
        super.__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(output, (input * expansion), bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear((input * expansion), output, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.sigmoid(self.fc2(self.gelu(self.fc1(y))))
        y = x.view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class MBConv(nn.Module):
    def __init__(self, input, output, img_size, downsample=False, expansion=4):
        super(MBConv, self).__init__()
        self.downsample = downsample
        stride = 1  if self.downsample == False else 2

        hidden_dim = int(input * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(input, output, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv_1 =


