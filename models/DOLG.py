# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:43:07 2021

@author: innat
"""
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import applications, layers
from layers.GeM import GeneralizedMeanPooling2D
from layers.LocalBranch import DOLGLocalBranch
from layers.OrtholFusion import OrthogonalFusion

class DOLGNet(keras.Model):
    def __init__(self, backbone=None, num_classes=1, activation=None, **kwargs):
        super(DOLGNet, self).__init__(name='DOLGNet', **kwargs)
        # Number of classes 
        self.num_classes = num_classes
        self.activation  = activation
        
        # Base blcoks 
        self.base = backbone
        self.base_input_shape  = self.base.input_shape[0][1]

        # Top building blocks 
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch      = DOLGLocalBranch(IMG_SIZE=self.base_input_shape)
        
        # Tail blcok 1 
        self.glob_branch_pool = keras.Sequential(
            [
                GeneralizedMeanPooling2D(),
                layers.Dense(1024, activation=None)
            ], 
            name='GlobalBranchPooling'
        )
        
        # Head block
        if self.num_classes == 1:
            self.classifier = keras.Sequential(
                [
                    layers.GlobalAveragePooling2D(name='HeadGAP'),
                    layers.Dense(self.num_classes, activation = self.activation)
                ], 
                name='Classifiers'
            )
        elif self.num_classes == 2:
            self.classifier = keras.Sequential(
                [
                    layers.GlobalAveragePooling2D(name='HeadGAP'),
                    layers.Dense(self.num_classes, activation = self.activation)
                ], 
                name='Classifiers'
            )
        else:
            self.classifier = keras.Sequential(
                [
                    layers.GlobalAveragePooling2D(name='HeadGAP'),
                    layers.Dense(self.num_classes, activation = self.activation)
            ], 
                name='Classifiers'
            )

    # forwarding the computation 
    def call(self, inputs, training=None, **kwargs):
        # Get tensor from target layers 
        to_local, to_global = self.base(inputs)

        # Pass the received tensor to Top building blocks 
        local_feat      = self.local_branch(to_local)
        global_feat     = self.glob_branch_pool(to_global)
        orthogonal_feat = self.orthogonal_fusion([local_feat, global_feat]) 
        return self.classifier(orthogonal_feat)

    def build_graph(self):
        x = keras.Input(shape=(self.base_input_shape, self.base_input_shape, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))
    
