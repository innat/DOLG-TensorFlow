# DOLG-TensorFlow

This is an unofficial implementation of **Deep Orthogonal Fusion of Local and Global Features (DOLG)** in `TensorFlow 2 (Keras)`. [Paper](https://arxiv.org/pdf/2108.02927.pdf). In this repository, the following architechture has been implemented. Details [here](https://mp.weixin.qq.com/s/7B3hZUpLtTt8NcGt0c-77w).

![image](https://user-images.githubusercontent.com/17668390/138777383-b1d475d7-c842-4577-8554-30cf2013cadc.png)


## Prerequisites

```
TensorFlow 2
TensorFlow Addons 
```

## Execute

Reset the config file. `config.py`

```python
BATCH_SIZE  = 10
IMG_SIZE    = 768
CHANNELS    = 3
LR = 0.003
mixed_precision = False
```

Build the **DOLG** model with **EfficientNetB5**. 

```python
from config import IMG_SIZE, CHANNELS
from model.DOLG import DOLGNet 

print(model(tf.ones((1, IMG_SIZE, IMG_SIZE, CHANNELS)))[0].shape)
display(tf.keras.utils.plot_model(model.build_graph(), 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=False))
```



## Code Example:

[Here](https://github.com/innat/DOLG-TensorFlow/tree/main/Code%20Example) are some end-to-end code examples. 

- [DenseNet DOLGNet Malaria](https://github.com/innat/DOLG-TensorFlow/blob/main/Code%20Example/DenseNet%20DOLGNet%20Malaria.ipynb) | [Colab](https://colab.research.google.com/drive/1VI7qZQZX_sWZZM8eKN98gCbiY3Ju1NpY?usp=sharing)
- [EfficientNet DOLGNet Oxford Flowers 102](https://github.com/innat/DOLG-TensorFlow/blob/main/Code%20Example/EfficientNet%20DOLGNet%20Oxford%20Flowers%20102.ipynb) | [Colab](https://colab.research.google.com/drive/1WvxR6gh0SzqcYUnSNnVQRw9UiFzgFMgm?usp=sharing)
- [ResNet DOLGNet Cmaterdb](https://github.com/innat/DOLG-TensorFlow/blob/main/Code%20Example/ResNet%20DOLGNet%20Cmaterdb.ipynb) | [Colab](https://colab.research.google.com/drive/1uEV9GsEZnTyWoilVww8d_Jmn3cAcefZr?usp=sharing)


## Citations
```python
@misc{yang2021dolg,
      title={DOLG: Single-Stage Image Retrieval with Deep Orthogonal 
                                         Fusion of Local and Global Features}, 
      author={Min Yang and Dongliang He and Miao Fan and Baorong Shi and 
                                        Xuetong Xue and Fu Li and Errui Ding and Jizhou Huang},
      year={2021},
      eprint={2108.02927},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```




