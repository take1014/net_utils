#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models
from .net_utils import ResConvBlockWithBn

class ResNet18(models.Model):
    def __init__(self, in_channels=3):
        super(ResNet18, self).__init__()

        self.conv2d_bn_relu = Conv2dBnRelu(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding='same', bias=False)
        self.maxpool2d = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Layer 1
        self.layer1_1 = ResConvBlockWithBn(in_channels=64, out_channels=64, stride=(1, 1), bias=False)
        self.layer1_2 = ResConvBlockWithBn(in_channels=64, out_channels=64, stride=(1, 1), bias=False)

        # Layer 2
        self.layer2_1 = ResConvBlockWithBn(in_channels=64, out_channels=128, stride=(2, 2), bias=False)
        self.layer2_2 = ResConvBlockWithBn(in_channels=128, out_channels=128, stride=(1, 1), bias=False)

        # Layer 3
        self.layer3_1 = ResConvBlockWithBn(in_channels=128, out_channels=256, stride=(2, 2), bias=False)
        self.layer3_2 = ResConvBlockWithBn(in_channels=256, out_channels=256, stride=(1, 1), bias=False)

        # Layer 4
        self.layer4_1 = ResConvBlockWithBn(in_channels=256, out_channels=512, stride=(2, 2), bias=False)
        self.layer4_2 = ResConvBlockWithBn(in_channels=512, out_channels=512, stride=(1, 1), bias=False)

    def call(self, x):
        x = self.conv2d_bn_relu(x)
        x = self.maxpool2d(x)
        # Layer1
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        # Layer2
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        # Layer3
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        # Layer4
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        return x

if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    model = ResNet18(in_channels=3)
    input_tensor = tf.random.normal([1, 160, 672, 3])  # (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)

    # モデルの概要を表示
    model.build(input_shape=(None, 160, 672, 3))
    model.summary()
