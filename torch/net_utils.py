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

class ResLinearBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features, bias=True):
        super(ResLinearBlock, self).__init__()
        self.linear_relu1 = LinearRelu(in_features=in_features, out_features=out_features, bias=bias)
        self.linear_relu2 = LinearRelu(in_features=out_features, out_features=mid_features, bias=bias)
        self.linear = nn.Linear(in_features=mid_features, out_features=out_features, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x1 = self.linear_relu1(x)
        x2 = self.linear_relu2(x)
        x2 = self.linear(x2)
        return self.relu(torch.add(x1, x2))

class ResLinearBnBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features, bias=True):
        super(ResLinearBnBlock, self).__init__()
        self.linear_bn_relu1 = LinearBnRelu(in_features=in_features, out_features=out_features, bias=bias)
        self.linear_bn_relu2 = LinearBnRelu(in_features=out_features, out_features=mid_features, bias=bias)
        self.linear = nn.Linear(in_features=mid_features, out_features=out_features, bias=bias)
        self.bn     = nn.BatchNorm2d(num_features=out_features)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x1 = self.linear_bn_relu1(x)
        x2 = self.linear_bn_relu2(x)
        x2 = self.bn(self.linear(x2))
        return self.relu(torch.add(x1, x2))

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(ConvBnRelu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn     = nn.BatchNorm2d(num_features=out_channels)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(ConvRelu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        return self.relu(x)

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(ResConvBlock, self).__init__()
        self.conv2d_relu1 = ConvRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2d_relu2 = ConvRelu(in_channels=out_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2d = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x1 = self.conv2d_relu1(x)
        x2 = self.conv2d_relu2(x1)
        x2 = self.conv2d(x2)
        return self.relu(torch.add(x1, x2))

class ResConvBlockWithBn(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), bias=True):
        super(ResConvBlockWithBn, self).__init__()
        self.conv2d_bn_relu1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2d_bn_relu2 = ConvBnRelu(in_channels=out_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2d = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn     = nn.BatchNorm2d(num_features=out_channels)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x1 = self.conv2d_bn_relu1(x)
        x2 = self.conv2d_bn_relu2(x1)
        x2 = self.bn(self.conv2d(x2))
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