#!/usr/bin/env python3
import torch
import torch.nn as nn
from net_utils import Conv2dBnRelu, ResConvBlockWithBn, GlobalAvgPool2D

class ResNet18(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet18, self).__init__()

        self.conv2d_bn_relu = Conv2dBnRelu(in_channels=in_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

        # Layer 1
        self.layer1_1 = ResConvBlockWithBn(in_channels=64, out_channels=64, stride=(1,1), bias=False)
        self.layer1_2 = ResConvBlockWithBn(in_channels=64, out_channels=64, stride=(1,1), bias=False)

        # Layer 2
        self.layer2_1 = ResConvBlockWithBn(in_channels=64, out_channels=128, stride=(2,2), bias=False)
        self.layer2_2 = ResConvBlockWithBn(in_channels=128, out_channels=128, stride=(1,1), bias=False)

        # Layer 3
        self.layer3_1 = ResConvBlockWithBn(in_channels=128, out_channels=256, stride=(2,2), bias=False)
        self.layer3_2 = ResConvBlockWithBn(in_channels=256, out_channels=256, stride=(1,1), bias=False)

        # Layer 4
        self.layer4_1 = ResConvBlockWithBn(in_channels=256, out_channels=512, stride=(2,2), bias=False)
        self.layer4_2 = ResConvBlockWithBn(in_channels=512, out_channels=512, stride=(1,1), bias=False)

    def forward(self, x):
        out = self.conv2d_bn_relu(x)
        out = self.maxpool2d(out)
        # Layer1
        out = self.layer1_1(out)
        out = self.layer1_2(out)
        # Layer2
        out = self.layer2_1(out)
        out = self.layer2_2(out)
        # Layer3
        out = self.layer3_1(out)
        out = self.layer3_2(out)
        # Layer4
        out = self.layer4_1(out)
        out = self.layer4_2(out)

        return out

if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(in_channels=3)
    model = model.to(device=device)
    x = torch.randn(1, 3, 160, 672).to(device=device)
    output = model(x)
    print(output.shape)
    summary(model, input_size=(3, 160, 672))