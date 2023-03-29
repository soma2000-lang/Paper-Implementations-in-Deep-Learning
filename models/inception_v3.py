from abc import ABCMeta
import tensorflow as tf
from tensorflow import keras
import numpy as np

from inception_blocks import Conv2d_Bn, inception_block, \
    inception_A, reduction_A, inception_B, reduction_B, inception_C

import tensorflow as tf
import numpy as np


class InceptionV3(tf.keras.Model):
    def __init__(self,shape):
        super(InceptionV3,self).__init__()
        #self.inputs = tf.compat.v1.keras.layers.Input(shape=shape)
        self.block_1 = tf.keras.Sequential([inception_block()])
        self.block_A = tf.keras.Sequential([
            inception_A(),
            inception_A(),
            inception_A(),

            reduction_A()
        ])
        self.block_B = tf.keras.Sequential([
            inception_B(),
            inception_B(),
            inception_B(),
            inception_B(),

            reduction_B()
        ])
        self.block_C = tf.keras.Sequential([
            inception_C(),
            inception_C()
        ])
        self.global_avg = tf.keras.layers.GlobalAvgPool2D()
        #self.fc1 = tf.keras.layers.Dense(2048,activation='relu')
        self.fc2 = tf.keras.layers.Dense(1000,activation='softmax')

    def call(self,shape,training=None,**kwargs):
        #x = self.inputs
        x = self.block_1(shape)
        x = self.block_A(x)
        x = self.block_B(x)
        x = self.block_C(x)
        x = self.global_avg(x,training=training)
        #x = self.fc1(x,training=training)
        x = self.fc2(x,training=training)
        return x

model = InceptionV3((299,299,3))
model.build((1,299,299,3))
model.summary()