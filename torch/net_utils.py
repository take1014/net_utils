#!/usr/bin/env python3
import torch
import torch.nn as nn

class LinearRelu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearRelu, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)

class LinearBnRelu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearBnRelu, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.bn     = nn.BatchNorm2d(num_features=out_features)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return self.relu(x)

class LinearBn(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.bn     = nn.BatchNorm2d(num_features=out_features)

    def forward(self, x):
        x = self.linear(x)
        return self.bn(x)
class ResLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ResLinearBlock, self).__init__()
        self.linear_relu = LinearRelu(in_features=in_features, out_features=out_features, bias=bias)
        self.linear1 = nn.Linear(in_features=out_features, out_features=out_features, bias=bias)
        self.linear2 = None
        if (in_features != out_features):
            self.linear2 = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x2 = self.linear_relu(x)
        x2 = self.linear1(x2)
        x1 = x if self.linear2 is None else self.lienar2(x)
        return self.relu(torch.add(x1, x2))

class ResLinearBlockWithBn(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ResLinearBlockWithBn, self).__init__()
        self.linear_bn_relu = LinearBnRelu(in_features=in_features, out_features=out_features, bias=bias)
        self.linear_bn1 = LinearBn(in_features=out_features, out_features=out_features, bias=bias)
        self.linear_bn2 = None
        if (in_features != out_features):
            self.linear_bn2 = LinearBn(in_features=in_features, out_features=out_features, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x2 = self.linear_bn_relu(x)
        x2 = self.bn(self.linear1(x2))
        x1 = x if self.linear_bn2 is None else self.linear_bn2(x)
        return self.relu(torch.add(x1, x2))

class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(Conv2dBn, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn     = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        return self.bn(x)

class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(Conv2dBnRelu, self).__init__()
        self.conv2d_bn = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_bn(x)
        return self.relu(x)

class Conv2dRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(Conv2dRelu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        return self.relu(x)

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=(1,1), bias=True):
        super(ResConvBlock, self).__init__()
        self.conv2d_relu = Conv2dRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(1,1), dilation=dilation, bias=bias)
        self.conv2d1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(1,1), padding=padding, dilation=dilation, bias=bias)
        self.conv2d2 = None
        if (stride != (1,1) or (in_channels != out_channels)):
            self.conv2d2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1), stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x2 = self.conv2d_relu1(x)
        x2 = self.conv2d1(x2)
        x1 =  x if self.conv2d2 is None else self.conv2d2(x)
        return self.relu(torch.add(x1, x2))

class ResConvBlockWithBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(ResConvBlockWithBn, self).__init__()
        self.conv2d_bn_relu = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(1,1), dilation=dilation, bias=bias)
        self.conv2d_bn1 = Conv2dBn(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(1,1), padding=(1,1), dilation=dilation, bias=bias)
        self.conv2d_bn2 = None
        if (stride != (1, 1)) or (in_channels != out_channels):
            self.conv2d_bn2 = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=stride, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x2 = self.conv2d_bn_relu(x)
        x2 = self.conv2d_bn1(x2)
        x1 = x if self.conv2d_bn2 is None else self.conv2d_bn2(x)
        return self.relu(torch.add(x1, x2))

class DeConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(DeConvRelu, self).__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.deconv2d(x)
        return self.relu(x)

class DeConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(DeConvRelu, self).__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn     = nn.BatchNorm2d(num_features=out_channels)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.deconv2d(x)
        x = self.bn(x)
        return self.relu(x)

class GlobalAvgPool2D(nn.Module):
    def __init__(self,):
        super(GlobalAvgPool2D, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.gap(x)
        # Convert (N, C, 1, 1) to (N, C)
        return x.view(x.size(0), -1)