#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models
from net_utils import ResConvBlockWithBn, Conv2dBnRelu, GlobalAvgPool2d

class ResNet18(models.Model):
    def __init__(self, config=None, in_channels=3, name='resnet18'):
        super(ResNet18, self).__init__(name=name)
        assert config is not None
        self.config = config

        self.stem = tf.keras.Sequential(name=self.name+'_stem')
        self.stem.add(Conv2dBnRelu(**self.config['stem_params']['conv'], name=self.name +'_stem_conv'))
        self.stem.add(layers.MaxPool2D(**self.config['stem_params']['maxpool'], name=self.name+'_stem_maxpool'))

        self.layer_list = []
        for i, layer_params in enumerate(self.config['layers_params']):
            self.layer_list.append(ResConvBlockWithBn(**layer_params, name=self.name+f'_layer_{i}'))

        self.out_conv = Conv2dBnRelu(filters=self.config['output_filters'], kernel_size=(1,1), stride=(1,1), padding='same', dilation_rate=(1,1), bias=False, name=self.name+'_out_conv')

        if self.config["apply_gap"]:
            self.gap = GlobalAvgPool2d(name=self.name+'_gap')


    def call(self, x):
        x = self.stem(x)
        for layer in self.layer_list:
            x = layer(x)
        x = self.out_conv(x)
        if self.config["apply_gap"]:
            return self.gap(x)
        return x

if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    import yaml
    with open("./backborn_params.yaml", 'r') as f:
        backborn_params = yaml.safe_load(f)

    model = ResNet18(backborn_params['resnet18'], in_channels=3)
    input_tensor = tf.random.normal([1, 160, 672, 3])  # (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)

    # モデルの概要を表示
    model.build(input_shape=(None, 160, 672, 3))
    model.summary()
