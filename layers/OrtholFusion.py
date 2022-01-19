# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:39:41 2021
@author: innat
"""
import tensorflow as tf 
from tensorflow.keras import layers 

class OrthogonalFusion(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(name='OrthogonalFusion', **kwargs)
    def call(self, inputs):
        local_feat, global_feat = inputs
        height = local_feat.shape[1]
        width  = local_feat.shape[2]
        depth  = local_feat.shape[3]
    
        local_feat = tf.reshape(local_feat, [-1, height*width, depth])
        local_feat = tf.transpose(local_feat, perm=[0, 2, 1])
        
        projection = tf.matmul(
            tf.expand_dims(global_feat, axis=1), 
            local_feat
        )
        projection = tf.matmul(
            tf.expand_dims(global_feat, axis=2),
            projection
        )
        projection = tf.reshape(projection, [-1, height, width, depth])
        
        global_feat_norm = tf.norm(global_feat, ord=2, axis=1)  
        projection = projection / tf.reshape(global_feat_norm*global_feat_norm, shape=[-1, 1, 1, 1])
        local_feat = tf.transpose(local_feat, perm=[0, 1, 2])
        local_feat = tf.reshape(local_feat, [-1, height, width, depth])
    
        orthogonal_comp = local_feat - projection
        global_feat = tf.expand_dims(tf.expand_dims(global_feat, axis=1), axis=1)
        global_feat = tf.broadcast_to(global_feat, tf.shape(local_feat))
        output = tf.concat([global_feat, orthogonal_comp], axis=-1)
        return output