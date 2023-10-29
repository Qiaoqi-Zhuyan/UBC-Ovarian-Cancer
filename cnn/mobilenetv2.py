import torch.nn as nn


class Conv_bn(nn.Module):
    def __init__(self, input, output, stride):
        super(Conv_bn, self).__init__()
        self.conv2d = nn.Conv2d(input, output, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(output)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class conv_1x1_bn()