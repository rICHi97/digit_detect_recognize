# -*- coding: utf-8 -*-
"""
Created on 2022-03-28 01:06:40

@author: Li Zhi
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations, layers  #pylint: disable=E0401

Input = keras.Input
Model = keras.Model
relu = activations.relu
Add = layers.Add
BatchNormalization = layers.BatchNormalization
Concatenate = layers.Concatenate
Conv2D = layers.Conv2D
Layer = layers.Layer
MaxPooling2D = layers.MaxPooling2D


class Scale(Layer):
    """
    y = alpha * x + beta
    """
    def __init__(self, ch, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.ch = ch
        alpha_init = tf.constant_initializer(1.0)
        self.alpha = tf.Variable(
            initial_value=alpha_init(shape=(1, self.ch), dtype=tf.float32),
            trainable=True,
        )
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=(1, self.ch), dtype=tf.float32)
        )

    def call(self, inputs):
        return self.alpha * inputs + self.beta

    def get_config(self):
        config = super(Scale, self).get_config()
        config.update({'ch': self.ch})
        return config


# TODO：scale层中的变量是否加入loss
# TODO：卷积模块由3个卷积连接而成，中间的取反，剩下的两个有必要scale吗？
class PVAnet():

    def __init__(self, inputs):
        self.inputs = inputs
        self.layers = {}
        self.network = self.create_network()

    def CReLU_blcok_start(self, inputs):
        """
        起始层没有1x1卷积
        conv-bn-neg-concat-scale-relu
        Parameters
        ----------
        Returns
        ----------
        """
        conv = Conv2D(16, 7, 2, 'same')(inputs)
        conv = BatchNormalization()(conv)
        conv_neg = tf.math.multiply(conv, -1.0) # 元素取负，通道数*2
        conv = Concatenate()([conv, conv_neg]) # ch = 32
        scale = Scale(32)
        conv = scale(conv)
        conv = relu(conv)

        return conv

    def CReLU_blcok(self, inputs, first_conv_stride, block_prefix):
        """
        {conv-bn-scale-relu}-{conv-bn-neg-concat-scale-relu}-{conv-bn-scale}
        由3个卷积连接而成
        卷积大小：1,3,1
        Parameters
        ----------
        stride：第一个卷积的步长，其余两个卷积步长为1
        block_prefix：'conv2_1'/'conv2_2'/'conv2_3'/'conv3_1'/'conv3_2'/'conv3_3'/'conv3_4'
            block2阶段，3个连接卷积通道数，24-24-64；
            block3阶段，3个连接卷积通道数，48-48-128；
            其中第二个为卷积取反连接，实际输出通道数为2*

        Returns
        ----------
        """
        if 'conv2' in block_prefix:
            ch1, ch2, ch3 = 24, 24, 64
        elif 'conv3' in block_prefix:
            ch1, ch2, ch3 = 48, 48, 128
        # block_prefix_1
        conv_1 = Conv2D(ch1, 1, first_conv_stride, padding='same')(inputs)
        conv_1 = BatchNormalization()(conv_1)
        scale = Scale(ch1)
        conv_1 = scale(conv_1)
        conv_1 = relu(conv_1)
        # conv_1.name = f'{block_prefix}_1'
        # block_prefix_2
        conv_2 = Conv2D(ch2, 3, 1, padding='same')(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2_neg = tf.math.multiply(conv_2, -1.0)
        conv_2 = Concatenate()([conv_2, conv_2_neg])
        scale = Scale(2 * ch2)
        conv_2 = scale(conv_2)
        conv_2 = relu(conv_2)
        # conv_2.name = f'{block_prefix}_2'
        # block_prefix_3
        conv_3 = Conv2D(ch3, 1, 1, padding='same')(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        scale = Scale(ch3)
        conv_3 = scale(conv_3)
        # conv_3.name = f'{block_prefix}_3'
        self.layers[f'{block_prefix}_3'] = conv_3

        return conv_3

    # TODO：函数封装
    def Inception_block(self, inputs, block_prefix):
        """
        1*1
        1*1->3*3
        1*1->3*3->3*3
        pool->1*1(仅对起始inception模块)
        Parameters
        ----------
        Returns
        ----------
        """
        bn_scale_relu = lambda ch, conv: (
            relu(
                Scale(ch)(
                    BatchNormalization()(conv)
                )
            )
        )

        concat_layers = []
        eq1_conv1_ch, conv1_stride, pool_conv_ch = 64, 1, None

        if 'conv4' in block_prefix:
            eq3_conv1_ch, eq3_conv2_ch = 64, 128
            eq5_conv1_ch, eq5_conv2_ch, eq5_conv3_ch = 24, 48, 48
            out_conv_ch = 256
            if block_prefix == 'conv4_1':
                eq3_conv1_ch, conv1_stride, pool_conv_ch = 48, 2, 128

        elif 'conv5' in block_prefix:
            eq3_conv1_ch, eq3_conv2_ch = 96, 192
            eq5_conv1_ch, eq5_conv2_ch, eq5_conv3_ch = 32, 64, 64
            out_conv_ch = 384
            if block_prefix == 'conv5_1':
                conv1_stride, pool_conv_ch = 2, 128

        # 等效1 x 1
        eq1_conv1 = Conv2D(eq1_conv1_ch, 1, conv1_stride, padding='same')(inputs)
        eq1_conv1 = bn_scale_relu(eq1_conv1_ch, eq1_conv1)
        concat_layers.append(eq1_conv1)

        # 等效3 x 3
        eq3_conv1 = Conv2D(eq3_conv1_ch, 1, conv1_stride, padding='same')(inputs)
        eq3_conv1 = bn_scale_relu(eq3_conv1_ch, eq3_conv1)
        eq3_conv2 = Conv2D(eq3_conv2_ch, 3, 1, padding='same')(eq3_conv1)
        eq3_conv2 = bn_scale_relu(eq3_conv2_ch, eq3_conv2)
        concat_layers.append(eq3_conv2)

        # 等效5 x 5
        eq5_conv1 = Conv2D(eq5_conv1_ch, 1, conv1_stride, padding='same')(inputs)
        eq5_conv1 = bn_scale_relu(eq5_conv1_ch, eq5_conv1)
        eq5_conv2 = Conv2D(eq5_conv2_ch, 3, 1, padding='same')(eq5_conv1)
        eq5_conv2 = bn_scale_relu(eq5_conv2_ch, eq5_conv2)
        eq5_conv3 = Conv2D(eq5_conv3_ch, 3, 1, padding='same')(eq5_conv2)
        eq5_conv3 = bn_scale_relu(eq5_conv3_ch, eq5_conv3)
        concat_layers.append(eq5_conv3)

        if pool_conv_ch is not None:
            pool = MaxPooling2D(pool_size=3, strides=2)(inputs)
            pool_conv = Conv2D(pool_conv_ch, 1, 1, padding='same')(pool)
            pool_conv_ch = bn_scale_relu(pool_conv_ch, pool_conv)
            concat_layers.append(pool_conv_ch)

        # out层无需relu
        concat = Concatenate()(concat_layers)
        out_conv = Conv2D(out_conv_ch, 1, 1)(concat)
        out_conv = BatchNormalization()(out_conv)
        out_conv = Scale(out_conv_ch)(out_conv)
        self.layers[f'{block_prefix}_out'] = out_conv

        return out_conv

    def residual(self, source_layer, target_layer):
        """
        source_layer = conv_project(source_layer) if project else soure_layer
        Parameters
        ----------
        Returns
        ----------
        """
        res = Add()([source_layer, target_layer])

        return res

    # TODO：函数封装
    def create_network(self):
        """
        conv3_4之前为C.ReLu模块
        conv4_1到conv5_4为Inception模块
        两个相连模块之间残差连接
        跨大模块加入一个project中间卷积用于统一通道数
        Parameters
        ----------

        Returns
        ----------
        """

        conv1_1 = self.CReLU_blcok_start(self.inputs)
        pool1_1 = MaxPooling2D(pool_size=3, strides=2)(conv1_1)
        _ = self.CReLU_blcok(pool1_1, 1, 'conv2_1') # blcok_conv_2_1

        pool1_1_project = Conv2D(64, 1, 1, padding='same')(pool1_1)
        pool1_1_project = BatchNormalization()(pool1_1_project)
        scale = Scale(64)
        pool1_1_project = scale(pool1_1_project)
        conv2_1_3 = self.layers['conv2_1_3']
        conv2_1 = self.residual(pool1_1_project, conv2_1_3)

        # conv2_1连接conv2_2
        _ = self.CReLU_blcok(conv2_1, 1, 'conv2_2')
        conv2_2_3 = self.layers['conv2_2_3']
        conv2_2 = self.residual(conv2_1, conv2_2_3)

        # conv2_2连接conv2_3
        _ = self.CReLU_blcok(conv2_2, 1, 'conv2_3')
        conv2_3_3 = self.layers['conv2_3_3']
        conv2_3 = self.residual(conv2_2, conv2_3_3)

        # conv2_3连接conv3_1
        _ = self.CReLU_blcok(conv2_3, 2, 'conv3_1')
        conv2_3_project = Conv2D(128, 1, 2, padding='same')(conv2_3_3)
        conv2_3_project = BatchNormalization()(conv2_3_project)
        scale = Scale(128)
        conv2_3_project = scale(conv2_3_project)
        conv3_1_3 = self.layers['conv3_1_3']
        conv3_1 = self.residual(conv2_3_project, conv3_1_3)

        # conv3_1连接conv3_2
        _ = self.CReLU_blcok(conv3_1, 1, 'conv3_2')
        conv3_2_3 = self.layers['conv3_2_3']
        conv3_2 = self.residual(conv3_1, conv3_2_3)

        # conv3_2连接conv3_3
        _ = self.CReLU_blcok(conv3_2, 1, 'conv3_3')
        conv3_3_3 = self.layers['conv3_3_3']
        conv3_3 = self.residual(conv3_2, conv3_3_3)

        # conv3_3连接conv3_4
        _ = self.CReLU_blcok(conv3_3, 1, 'conv3_4')
        conv3_4_3 = self.layers['conv3_4_3']
        conv3_4 = self.residual(conv3_3, conv3_4_3)

        # conv3_4连接conv4_1
        _ = self.Inception_block(conv3_4, 'conv4_1')
        conv4_1_out = self.layers['conv4_1_out']
        conv3_4_project = Conv2D(256, 1, 2, padding='same')(conv3_4)
        conv3_4_project = BatchNormalization()(conv3_4_project)
        scale = Scale(256)
        conv3_4_project = scale(conv3_4_project)
        conv4_1 = self.residual(conv3_4_project, conv4_1_out)

        # conv4_1连接conv4_2
        _ = self.Inception_block(conv4_1, 'conv4_2')
        conv4_2_out = self.layers['conv4_2_out']
        conv4_2 = self.residual(conv4_1, conv4_2_out)

        # conv4_2连接conv4_3
        _ = self.Inception_block(conv4_2, 'conv4_3')
        conv4_3_out = self.layers['conv4_3_out']
        conv4_3 = self.residual(conv4_2, conv4_3_out)

        # conv4_3连接conv4_4
        _ = self.Inception_block(conv4_3, 'conv4_4')
        conv4_4_out = self.layers['conv4_4_out']
        conv4_4 = self.residual(conv4_3, conv4_4_out) # 输出对应east中的f2

        # conv4_4连接conv5_1
        _ = self.Inception_block(conv4_4, 'conv5_1')
        conv5_1_out = self.layers['conv5_1_out']
        conv4_4_project = Conv2D(384, 1, 2, padding='same')(conv4_4)
        conv4_4_project = BatchNormalization()(conv4_4_project)
        scale = Scale(384)
        conv4_4_project = scale(conv4_4_project)
        conv5_1 = self.residual(conv4_4_project, conv5_1_out)

        # conv5_1连接conv5_2
        _ = self.Inception_block(conv5_1, 'conv5_2')
        conv5_2_out = self.layers['conv5_2_out']
        conv5_2 = self.residual(conv5_1, conv5_2_out)

        # conv5_2连接conv5_3
        _ = self.Inception_block(conv5_2, 'conv5_3')
        conv5_3_out = self.layers['conv5_3_out']
        conv5_3 = self.residual(conv5_2, conv5_3_out)

        # conv5_3连接conv5_4
        _ = self.Inception_block(conv5_3, 'conv5_4')
        conv5_4_out = self.layers['conv5_4_out']
        conv5_4 = self.residual(conv5_3, conv5_4_out) # 输出对应east中的f1

        self.features = {
            'f4': conv2_3, # east, f4
            'f3': conv3_3, # east, f3
            'f2': conv4_4, # east, f2
            'f1': conv5_4, # east, f1
        }

        return Model(inputs=self.inputs, outputs=conv5_4)
