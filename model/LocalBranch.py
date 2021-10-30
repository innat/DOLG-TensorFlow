# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:33:39 2021
@author: innat
"""
import config 
import tensorflow as tf 
from tensorflow.keras import (layers, Sequential, activations, initializers)

# Multi-Atrous Branch
class MultiAtrous(tf.keras.Model):
    def __init__(self, dilation_rates=[6, 12, 18], upsampling=1, 
                 kernel_size=3, padding="same",  **kwargs):
        super(MultiAtrous, self).__init__(name='MultiAtrous', **kwargs)
        self.dilation_rates = dilation_rates
        self.kernel_size = kernel_size 
        self.upsampling = upsampling
        self.padding = padding
        # Dilated Convolutions                     
        self.dilated_convs = [
                                layers.Conv2D(
                                    filters       = int(1024 / 4), 
                                    kernel_size   = self.kernel_size,  
                                    padding       = self.padding, 
                                    dilation_rate = rate
                                ) for rate in self.dilation_rates
                             ]
        
        # Global Average Pooling Branch 
        self.gap_branch = Sequential(
            [
                layers.GlobalAveragePooling2D(keepdims=True),
                layers.Conv2D(int(1024 / 2), kernel_size=1),
                layers.Activation('relu'),
                layers.UpSampling2D(size=self.upsampling, interpolation="bilinear")
            ] , name='gap_branch'
        )
        
    def call(self, inputs, training=None, **kwargs):
        local_feature = []

        for dilated_conv in self.dilated_convs:
            x = dilated_conv(inputs) 
            x = self.gap_branch(x)
            local_feature.append(x)
            
        return tf.concat(local_feature, axis=-1)

    def get_config(self):
        config = {
            'dilation_rates': self.dilation_rates,
            'kernel_size'   : self.kernel_size,
            'padding'       : self.padding,
            'upsampling'    : self.upsampling
        }
        base_config = super(MultiAtrous, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
# DOLG: Local-Branch
class DOLGLocalBranch(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DOLGLocalBranch, self).__init__(name='LocalBranch', **kwargs)
        self.multi_atrous = MultiAtrous(padding='same', upsampling=int(config.IMG_SIZE/32))
        self.conv1 = layers.Conv2D(1024, kernel_size=1)
        self.conv2 = layers.Conv2D(1024, kernel_size=1, use_bias=False)
        self.conv3 = layers.Conv2D(1024, kernel_size=1)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        # Local Branach + Normalization / Conv-Bn Module 
        local_feat = self.multi_atrous(inputs)
        local_feat = self.conv1(local_feat)
        local_feat = tf.nn.relu(local_feat)
        
        # Self-Attention
        local_feat = self.conv2(local_feat)
        local_feat = self.bn(local_feat)

        # l-2 norms
        norm_local_feat = tf.math.l2_normalize(local_feat)

        # softplus activations
        attn_map = tf.nn.relu(local_feat)
        attn_map = self.conv3(attn_map)
        attn_map = activations.softplus(attn_map) 

        # Output of the Local-Branch 
        return  norm_local_feat * attn_map 