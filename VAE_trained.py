import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import math
import tensorflow_addons as tfa
import time
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input ,Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization, LeakyReLU, Dense

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=32)

class AttentionBlock(layers.Layer):
    def __init__(self, units, groups=16, **kwargs):
        super(AttentionBlock,self).__init__()

        self.units = units
        self.groups = groups
        self.norm = tfa.layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init)
        self.key = layers.Dense(units, kernel_initializer=kernel_init)
        self.value = layers.Dense(units, kernel_initializer=kernel_init)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * tf.cast(self.units, tf.float32) ** (-0.5)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])
        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        out = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        return inputs + out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'groups': self.groups,
            'norm': self.norm,
            'query': self.query,
            'key': self.key,
            'value': self.value,
        })

def ResidualBlock(filters, groups=16):
    def apply(inputs):
        x = inputs
        input_width = x.shape[3]

        if input_width == filters:
            residual = x
        else:
            residual = layers.Conv2D(filters, kernel_size=1, kernel_initializer=kernel_init)(x)

        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = keras.activations.swish(x)                                                    
        x = layers.Conv2D( filters, kernel_size=3, padding="same", kernel_initializer=kernel_init)(x)

        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = keras.activations.swish(x)
        x = layers.Conv2D(filters, kernel_size=3, padding="same", kernel_initializer=kernel_init)(x)
        
        x = layers.Add()([x, residual])
        return x

    return apply

def build_model(img_size, img_channels, filters, norm_groups=16):
  
  def encoder():  
    image_input = layers.Input(shape=(img_size, img_size, img_channels), name="image_input")

    x = layers.Conv2D(filters[0], kernel_size=(3, 3), padding="same", kernel_initializer=kernel_init)(image_input)

    #DownBlock
    for i in range(len(filters)):

        x = ResidualBlock(filters[i], groups=norm_groups)(x)

        if i != len(filters) - 1:
             x = layers.Conv2D(filters[i], 3, strides=2, padding="same", kernel_initializer=kernel_init)(x) 

    # MiddleBlock
    x = ResidualBlock(filters[-1], groups=norm_groups)(x)
    x = AttentionBlock(filters[-1], groups=norm_groups)(x)
    x = ResidualBlock(filters[-1], groups=norm_groups)(x)

    x = tfa.layers.GroupNormalization(groups=norm_groups)(x)
    x = keras.activations.swish(x)
    out = layers.Conv2D(img_channels*2, kernel_size=3, padding="same", kernel_initializer=kernel_init)(x) 
    
    return Model(inputs = image_input, outputs = out, name="encoder")

  def decoder():

    # UpBlock
    image_input = layers.Input(shape=(img_size//2**(len(filters)-1), img_size//2**(len(filters)-1), img_channels), name="image_input")

    x = layers.Conv2D(filters[-1], kernel_size=(3, 3), padding="same", kernel_initializer=kernel_init)(image_input)

    for i in reversed(range(len(filters))):

        x = ResidualBlock(filters[i], groups=norm_groups)(x)
   
        if i != 0:
            x = layers.Conv2DTranspose(filters[i], 3, strides=2, padding='same', kernel_initializer=kernel_init)(x)

    x = tfa.layers.GroupNormalization(groups=norm_groups)(x)
    x = keras.activations.swish(x)
    x = layers.Conv2D(img_channels, (3, 3), padding="same", kernel_initializer=kernel_init, activation = tf.keras.activations.tanh)(x)
    return Model(image_input, x, name="decoder")
  
  return encoder(), decoder()

def meanvar(x, encoder, training=True):
    mean, logvar = tf.split(encoder(x, training=True), num_or_size_splits=2, axis=-1)
    return mean, logvar
    
def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean


