#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models
from .net_utils import ResConvBlockWithBn, Conv2dBnRelu, GlobalAvgPool2d, Conv2dBn

# HSwish Activation Function
class HSwish(tf.keras.layers.Layer):
    def __init__(self, name):
        super(HSwish, self).__init__(name=name)

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0

# Depthwise Separable Convolution
class DepthwiseSeparableConv(tf.keras.layers.Layer):
    def __init__(self, in_filters, out_filters, stride=(1, 1), expand_ratio=1, name='_depthwise'):
        super(DepthwiseSeparableConv, self).__init__(name=name)
        # 1x1 Pointwise Convolution (Expansion layer)
        self.expand = tf.keras.Sequential([
            Conv2dBn(filters=in_filters * expand_ratio, kernel_size=(1,1), stride=(1, 1), padding='same', bias=False, name=self.name+"_conv2dbn"),
            HSwish(name=self.name+"_hswith")
        ], name=self.name+'_expand')

        # Depthwise Convolution
        self.depthwise = tf.keras.Sequential([
            layers.DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding='same', use_bias=False, name=self.name+"depthwiseconv2d"),
            layers.BatchNormalization(name=self.name+"bn")
        ], name=self.name+'_depthwise')

        # Pointwise Convolution (Projection layer)
        self.project = Conv2dBn(filters=out_filters, kernel_size=(1,1), stride=(1,1), padding='same', bias=False)

    def call(self, x):
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        return x

# MobileNetV3 Model
class MobileNetV3(tf.keras.Model):
    def __init__(self, config=None, in_filters=3, name='mobilenetv3'):
        super(MobileNetV3, self).__init__(name=name)
        assert config is not None
        self.config = config

        # Stem
        self.stem = tf.keras.Sequential([
            Conv2dBn(**self.config['stem_params']['conv'], name=self.name+"_stem_conv2dbn"),
            HSwish(name=self.name+'_stem_hswish')
        ], name=self.name+'_stem')

        # Convolutional Layers
        self.conv_layers = []
        for i, params in enumerate(self.config["layers_params"]):
            self.conv_layers.append(DepthwiseSeparableConv(**params, name=self.name +f'_depthwise{i}'))

        # Output Convolution
        self.out_conv = tf.keras.Sequential([
            Conv2dBn(self.config['output_filters'], kernel_size=(1,1), padding='same', bias=False, name=self.name+'_out_conv2dbn'),
            HSwish(name=self.name+'_out_hswish')
        ], name=self.name+"_out_conv")

        # Global Average Pooling if apply_gap is True
        self.gap = layers.GlobalAveragePooling2D(name=self.name+'_gap') if self.config["apply_gap"] else None

    def call(self, x):
        x = self.stem(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.out_conv(x)
        if self.gap:
            x = self.gap(x)
        return x

class ResNet18(models.Model):
    def __init__(self, config=None, in_filters=3, name='resnet18'):
        super(ResNet18, self).__init__(name=name)
        assert config is not None
        self.config = config

        self.stem = tf.keras.Sequential([
            Conv2dBnRelu(**self.config['stem_params']['conv'], name=self.name +'_stem_conv'),
            layers.MaxPool2D(**self.config['stem_params']['maxpool'], name=self.name+'_stem_maxpool')
        ], name=self.name+'_stem')

        self.layer_list = []
        for i, layer_params in enumerate(self.config['layers_params']):
            self.layer_list.append(ResConvBlockWithBn(**layer_params, name=self.name+f'_layer_{i}'))

        self.out_conv = Conv2dBnRelu(filters=self.config['output_filters'], kernel_size=(1,1), stride=(1,1), padding='same', dilation_rate=(1,1), bias=False, name=self.name+'_out_conv')

        self.gap = GlobalAvgPool2d(name=self.name+'_gap') if self.config["apply_gap"] else None

    def call(self, x):
        x = self.stem(x)
        for layer in self.layer_list:
            x = layer(x)
        x = self.out_conv(x)
        if self.gap:
            return self.gap(x)
        return x

if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    import yaml
    with open("./backborn_params.yaml", 'r') as f:
        backborn_params = yaml.safe_load(f)

    input_tensor = tf.random.normal([1, 160, 672, 3])  # (batch_size, height, width, filters)

    # resnet18
    resnet18 = ResNet18(backborn_params['resnet18'], in_filters=3)
    output_tensor = resnet18(input_tensor)
    print(output_tensor.shape)

    resnet18.build(input_shape=(None, 160, 672, 3))
    resnet18.summary()

    # mobilenetv3
    mobilenetv3 = MobileNetV3(backborn_params["mobilenetv3"], in_filters=3)
    output_tensor = mobilenetv3(input_tensor)
    print(output_tensor.shape)

    # モデルの概要を表示
    mobilenetv3.build(input_shape=(None, 160, 672, 3))
    mobilenetv3.summary()
