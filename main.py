# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:28:49 2021

@author: innat
"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf; print(tf.__version__)
from model.DOLG import DOLGNet 
from config import IMG_SIZE, CHANNELS, LR, Classifier

# Build Model ----------------------------------------------
def get_model(plot_model, print_summary, with_compile):
    # EfficietNet-DOLGNet
    model = DOLGNet()

    if plot_model:
        print(model(tf.ones((1, IMG_SIZE, IMG_SIZE, CHANNELS)))[0].shape)
        display(tf.keras.utils.plot_model(model.build_graph(), 
                                          show_shapes=True, 
                                          show_layer_names=True,
                                          expand_nested=False))
    if print_summary:
        print(model(tf.ones((1, IMG_SIZE, IMG_SIZE, CHANNELS)))[0].shape)
        print(model.summary())
        
    if with_compile:
        print(model(tf.ones((1, IMG_SIZE, IMG_SIZE, CHANNELS)))[0].shape)
        if Classifier == 1:
            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = LR), 
                loss = tf.keras.losses.BinaryCrossentropy(), 
                metrics = ['accuracy'])  
        else:
            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = LR), 
                loss = tf.keras.losses.CategoricalCrossentropy(), 
                metrics = ['accuracy'])  
        
    return model 


# Get Compiled Model 
model = get_model(plot_model = True,  print_summary = False, with_compile  = True)


# MNIST Dataset ------------------------------
def mnist_process(x, y):
    x = tf.expand_dims(tf.cast(x, dtype=tf.float32), axis=-1)  
    x = tf.repeat(x, repeats=3, axis=-1)
    x = tf.divide(x, 255)       
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])  
    y = tf.one_hot(y , depth=Classifier)  
    return x, y

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(mnist_process)
train_ds = train_ds.shuffle(buffer_size=1024).batch(16)

for image, label in train_ds.take(5):
    print(image.shape, label.shape)


# Training --------------------------------
from tensorflow.keras import optimizers, metrics, losses
model.compile(optimizer=optimizers.Adam(learning_rate=LR),
              loss=losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(train_ds)


