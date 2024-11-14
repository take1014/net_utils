#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import Conv2dBnRelu, ResConvBlockWithBn, Conv2dBn, GlobalAvgPool2d

# HSwish 活性化関数の実装
class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0

# Depthwise separable convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1), expand_ratio=1):
        super(DepthwiseSeparableConv, self).__init__()

        # 1x1 Pointwise Convolution (Expansion layer)
        self.expand = Conv2dBn(in_channels=in_channels, out_channels=in_channels * expand_ratio, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        self.relu = HSwish()

        # Depthwise Convolution
        self.depthwise = Conv2dBn(in_channels=in_channels * expand_ratio, out_channels=in_channels * expand_ratio, kernel_size=(3,3), stride=stride, padding=(1,1), groups=in_channels * expand_ratio, bias=False)

        # Pointwise Convolution (Projection layer)
        self.project = Conv2dBn(in_channels=in_channels * expand_ratio, out_channels=out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)

    def forward(self, x):
        x = self.relu(self.expand(x))
        x = self.depthwise(x)
        x = self.project(x)
        return x

# MobileNetV3のモデルクラス
class MobileNetV3(nn.Module):
    def __init__(self, config=None, in_channels=3):
        super(MobileNetV3, self).__init__()
        assert config is not None
        self.config = config

        self.stem = nn.Sequential(
            Conv2dBn(**self.config['stem_params']['conv']),
            HSwish()
        )

        self.conv_layers = nn.ModuleList([
            DepthwiseSeparableConv(**params)
            for params in self.config["layers_params"]
        ])

        self.out_conv = nn.Sequential(
            Conv2dBnRelu(in_channels=self.config["layers_params"][-1]['out_channels'], out_channels=self.config['output_channels'], kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),
            HSwish()
        )

        self.gap = GlobalAvgPool2d() if self.config["apply_gap"] else None

    def forward(self, x):
        x = self.stem(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.out_conv(x)
        if self.gap:
            return self.gap(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, config=None, in_channels=3):
        super(ResNet18, self).__init__()
        assert config is not None
        self.config = config

        self.stem = nn.Sequential(
            Conv2dBnRelu(**self.config['stem_params']['conv']),
            nn.MaxPool2d(**self.config['stem_params']['maxpool'])
        )

        self.conv_layers = nn.ModuleList([
            ResConvBlockWithBn(**params)
            for params in self.config["layers_params"]
        ])

        self.out_conv = Conv2dBnRelu(in_channels=self.config["layers_params"][-1]['out_channels'], out_channels=self.config['output_channels'], kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)

        self.gap = GlobalAvgPool2d() if self.config["apply_gap"] else None

    def forward(self, x):
        x = self.stem(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.out_conv(x)
        if self.gap:
            return self.gap(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    import yaml
    with open("./backborn_params.yaml", 'r') as f:
        backborn_params = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 160, 672).to(device=device)
    resnet18 = ResNet18(config=backborn_params['resnet18'], in_channels=3)
    resnet18 = resnet18.to(device=device)
    output = resnet18(x)
    print(output.shape)
    summary(resnet18, input_size=(3, 160, 672))

    mobilenet3 = MobileNetV3(config=backborn_params['mobilenetv3'], in_channels=3)
    mobilenet3 = mobilenet3.to(device=device)
    output = mobilenet3(x)
    print(output.shape)
    summary(mobilenet3, input_size=(3, 160, 672))
