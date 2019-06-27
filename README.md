# Weight_Standardization
Weight Standardization Analysis with Keras

Weight standardization is a new regularization technique that can outperform Batch Normalisation on micro batches, while reaching the same performance on large batches.

**The original research paper :**
  Siyuan Qiao Huiyu Wang Chenxi Liu Wei Shen Alan Yuille : “Weight Standardization” (2019)
  https://arxiv.org/pdf/1903.10520v1.pdf

## Experiments

We tested the performance of Group Normalization (GN) + Weight Standardization (WS) against Batch Normalization (BN) with a multiclass classifier on the MNIST-FASHION dataset.

We are using Python 3.6 and the Keras library.

The code for the original classifier is this official Google Example :
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb

To implement GN we used the code available at this repo :
https://github.com/titu1994/Keras-Group-Normalization

To implement WS we used a custom keras kernel regulizer funcitons :

```python
def ws_reg(kernel):
  kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
  kernel = kernel - kernel_mean
  kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
  kernel = kernel / (kernel_std + 1e-5)
  return kernel
```

## Results

<img src=https://imgur.com/H3NZnWi.png />

Final accuracy per size and model :

Batch size | 2 | 64 | 128
:-:|:--:|:--:|:--:
BN | 18.78 | 92.49 | 92.93
GN+WS | 91.86 | 92.82 | 92.9

## Files

- "/src" : source code, the python notebook is meant to be load in colab, with your drive mounted locallt?
The python code will run for a default batch size of 128 and 10 apochs.

- "weight_standardization_analysis.pdf" : detailed report of the project.

- "/data" : model architectures, model weights, model histories, and plots for several dfferent parameters.


This project was made by Clement Rouault and Thomas Ehling
