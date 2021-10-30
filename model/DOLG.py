# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:43:07 2021

@author: innat
"""
import tensorflow as tf 
from tensorflow.keras import applications, layers, Model, Input

from config import IMG_SIZE
from model.GeM import GeneralizedMeanPooling2D
from model.LocalBranch import DOLGLocalBranch
from model.OrtholFusion import OrthogonalFusion

class DOLGNet(tf.keras.Model):
    def __init__(self, Classifier, **kwargs):
        super(DOLGNet, self).__init__(name='DOLGNet', **kwargs)
        # Number of classes 
        self.Classifier = Classifier
        
        # Top building blocks 
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch      = DOLGLocalBranch()
        
        # Tail blcok 1 
        self.glob_branch_pool = Sequential([
            GeneralizedMeanPooling2D(),
            layers.Dense(1024, activation=None)
        ], name='GlobalBranchPooling')
        
        # Base blcoks 
        base = applications.EfficientNetB5(
            include_top=False,
            weights='imagenet',
            input_tensor=Input((IMG_SIZE, IMG_SIZE, 3))
        )
        self.new_base = Model(
            [base.inputs], 
            [
                base.get_layer('block5g_add').output,  # fol local branch 
                base.get_layer('block7c_add').output   # for global branch 
             ], 
            name='EfficientNet'
        )
        
        # Head block
        if Classifier == 1:
            self.classifier = Sequential([
                layers.GlobalAveragePooling2D(name='HeadGAP'),
                layers.Dense(1, activation = 'sigmoid')
            ], name='Classifiers')

        else:
            self.classifier = Sequential([
                layers.GlobalAveragePooling2D(name='HeadGAP'),
                layers.Dense(Classifier, activation = 'softmax')
            ], name='Classifiers')
    
            
    # forwarding the computation 
    def call(self, inputs, training=None, **kwargs):
        # Get tensor from target layers 
        to_local, to_global = self.new_base(inputs)

        # Pass the received tensor to Top building blocks 
        local_feat      = self.local_branch(to_local)
        global_feat     = self.glob_branch_pool(to_global)
        orthogonal_feat = self.orthogonal_fusion([local_feat, global_feat]) 
        
        if training:
            return self.classifier(orthogonal_feat)
        else:
            return self.classifier(orthogonal_feat), orthogonal_feat

    def build_graph(self):
        x = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        return Model(inputs=[x], outputs=self.call(x))
    