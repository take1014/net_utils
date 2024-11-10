#!/usr/bin/env/python3
import tensorflow as tf
from tensorflow.keras import layers

class LinearRelu(layers.Layer):
    def __init__(self, units, bias=True, name=None):
        super(LinearRelu, self).__init__(name=name)
        self.linear = layers.Dense(units=units, use_bias=bias, name=f"{name}_linear")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x = self.linear(inputs)
        return self.relu(x)

class LinearBnRelu(layers.Layer):
    def __init__(self, units, bias=True, name=None):
        super(LinearBnRelu, self).__init__(name=name)
        self.linear = layers.Dense(units=units, use_bias=bias, name=f"{name}_linear")
        self.bn = layers.BatchNormalization(name=f"{name}_bn")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x = self.linear(inputs)
        x = self.bn(x)
        return self.relu(x)

class LinearBn(layers.Layer):
    def __init__(self, units, bias=True, name=None):
        super(LinearBn, self).__init__(name=name)
        self.linear = layers.Dense(units=units, use_bias=bias, name=f"{name}_linear")
        self.bn = layers.BatchNormalization(name=f"{name}_bn")

    def call(self, inputs):
        x = self.linear(inputs)
        return self.bn(x)

class ResLinearBlock(layers.Layer):
    def __init__(self, in_units, out_units, bias=True, name=None):
        super(ResLinearBlock, self).__init__(name=name)
        self.linear_relu = LinearRelu(units=out_units, bias=bias, name=f"{name}_linear_relu")
        self.linear1 = layers.Dense(units=out_units, use_bias=bias, name=f"{name}_linear1")
        self.linear2 = None
        if in_units != out_units:
            self.linear2 = layers.Dense(units=out_units, use_bias=bias, name=f"{name}_linear2")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x2 = self.linear_relu(inputs)
        x2 = self.linear1(x2)
        x1 = inputs if self.linear2 is None else self.linear2(inputs)
        return self.relu(layers.Add(name=f"{self.name}_add")([x1, x2]))

class ResLinearBlockWithBn(layers.Layer):
    def __init__(self, in_units, out_units, bias=True, name=None):
        super(ResLinearBlockWithBn, self).__init__(name=name)
        self.linear_bn_relu = LinearBnRelu(units=out_units, bias=bias, name=f"{name}_linear_bn_relu")
        self.linear_bn1 = LinearBn(units=out_units, bias=bias, name=f"{name}_linear_bn1")
        self.linear_bn2 = None
        if in_units != out_units:
            self.linear_bn2 = LinearBn(units=out_units, bias=bias, name=f"{name}_linear_bn2")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x2 = self.linear_bn_relu(inputs)
        x2 = self.linear_bn1(x2)
        x1 = inputs if self.linear_bn2 is None else self.linear_bn2(inputs)
        return self.relu(layers.Add(name=f"{self.name}_add")([x1, x2]))

class Conv2dBn(layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), stride=(1,1), padding='valid', dilation_rate=(1,1), bias=True, name=None):
        super(Conv2dBn, self).__init__(name=name)
        self.conv2d = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation_rate, use_bias=bias, name=f"{name}_conv2d")
        self.bn = layers.BatchNormalization(name=f"{name}_bn")

    def call(self, inputs):
        x = self.conv2d(inputs)
        return self.bn(x)

class Conv2dBnRelu(layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), stride=(1,1), padding='valid', dilation_rate=(1,1), bias=True, name=None):
        super(Conv2dBnRelu, self).__init__(name=name)
        self.conv2d_bn = Conv2dBn(filters=filters, kernel_size=kernel_size, stride=stride, padding=padding, dilation_rate=dilation_rate, bias=bias, name=f"{name}_conv2d_bn")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x = self.conv2d_bn(inputs)
        return self.relu(x)

class Conv2dRelu(layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), stride=(1,1), padding='valid', dilation_rate=(1,1), bias=True, name=None):
        super(Conv2dRelu, self).__init__(name=name)
        self.conv2d = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation_rate, use_bias=bias, name=f"{name}_conv2d")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x = self.conv2d(inputs)
        return self.relu(x)

class ResConvBlock(layers.Layer):
    def __init__(self, in_filters, out_filters, kernel_size=(3,3), stride=(1,1), padding='same', dilation_rate=(1,1), bias=True, name=None):
        super(ResConvBlock, self).__init__(name=name)
        self.conv2d_relu = Conv2dRelu(filters=out_filters, kernel_size=kernel_size, stride=stride, padding=padding, dilation_rate=dilation_rate, bias=bias, name=f"{name}_conv2d_relu")
        self.conv2d1 = layers.Conv2D(filters=out_filters, kernel_size=kernel_size, strides=(1,1), padding=padding, dilation_rate=dilation_rate, use_bias=bias, name=f"{name}_conv2d1")
        self.conv2d2 = None
        if stride != (1,1) or in_filters != out_filters:
            self.conv2d2 = layers.Conv2D(filters=out_filters, kernel_size=(1,1), strides=stride, padding=padding, dilation_rate=dilation_rate, use_bias=bias, name=f"{name}_conv2d2")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x2 = self.conv2d_relu(inputs)
        x2 = self.conv2d1(x2)
        x1 = inputs if self.conv2d2 is None else self.conv2d2(inputs)
        return self.relu(layers.Add(name=f"{self.name}_add")([x1, x2]))

class ResConvBlockWithBn(layers.Layer):
    def __init__(self, in_filters, out_filters, kernel_size=(3,3), stride=(1,1), padding='same', dilation_rate=(1,1), bias=True, name=None):
        super(ResConvBlockWithBn, self).__init__(name=name)
        self.conv2d_bn_relu = Conv2dBnRelu(filters=out_filters, kernel_size=kernel_size, stride=stride, padding=padding, dilation_rate=dilation_rate, bias=bias, name=f"{name}_conv2d_bn_relu")
        self.conv2d_bn1 = Conv2dBn(filters=out_filters, kernel_size=kernel_size, stride=(1,1), padding=padding, dilation_rate=dilation_rate, bias=bias, name=f"{name}_conv2d_bn1")
        self.conv2d_bn2 = None
        if stride != (1,1) or in_filters != out_filters:
            self.conv2d_bn2 = Conv2dBn(filters=out_filters, kernel_size=(1,1), stride=stride, bias=bias, name=f"{name}_conv2d_bn2")
        self.relu = layers.ReLU(name=f"{name}_relu")

    def call(self, inputs):
        x2 = self.conv2d_bn_relu(inputs)
        x2 = self.conv2d_bn1(x2)
        x1 = inputs if self.conv2d_bn2 is None else self.conv2d_bn2(inputs)
        return self.relu(layers.Add(name=f"{self.name}_add")([x1, x2]))

class GlobalAvgPool2d(layers.Layer):
    def __init__(self, name=None):
        super(GlobalAvgPool2d, self).__init__(name=name)
        self.pooling = layers.GlobalAveragePooling2D(name=f"{name}_gap")

    def call(self, inputs):
        return self.pooling(inputs)
