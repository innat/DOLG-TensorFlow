# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:30:00 2021

@author: innat
"""
global_seed = 101
BATCH_SIZE  = 10
IMG_SIZE    = 768
CHANNELS    = 3
LR = 0.003
mixed_precision = False


def accelerate_gpu(mp=False):
    import tensorflow as tf 
    # Params 
    AUTO = tf.data.AUTOTUNE
    print(tf.executing_eagerly())
    print('A: ', tf.test.is_built_with_cuda)
    print('B: ', tf.test.gpu_device_name())
    
    GPUS = tf.config.list_physical_devices('GPU')
    if GPUS:
        try:
            for GPU in GPUS:
                tf.config.experimental.set_memory_growth(GPU, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(GPUS), "Physical GPUs,", len(logical_gpus), "Logical GPUs") 
        except RuntimeError as  RE:
            print(RE)
    if mp:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('Mixed precision enabled')

import random, numpy as np, os
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    
    