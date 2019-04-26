# -*- coding: utf-8 -*-



"""

This classifier is a modification of the original classifier available at :

    https://colab.research.google.com/drive/1fImuGSGDsYrvj7PNEPIC4LkqtibPAEBH

"""

import pickle

PATH_DRIVE = "../data/"

from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints

from keras.utils.generic_utils import get_custom_objects


class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                                                                       'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                                                                       'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})


def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))


def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    # kernel_std = tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_std')
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

type(train_images)

validation_images = train_images[50000:, ]
train_images = train_images[:50000, ]

validation_labels = train_labels[50000:, ]
train_labels = train_labels[:50000, ]

train_images = train_images.reshape(50000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
validation_images = validation_images.reshape(10000, 28, 28, 1)

train_images = train_images / 255.0

validation_images = validation_images / 255.0

test_images = test_images / 255.0

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

chanDim = -1

model_bn = Sequential()
model_bn.add(Conv2D(32, (3, 3), padding="same",
                    input_shape=(28, 28, 1)))
model_bn.add(Activation("relu"))
model_bn.add(BatchNormalization(axis=chanDim))
model_bn.add(Conv2D(32, (3, 3), padding="same"))
model_bn.add(Activation("relu"))
model_bn.add(BatchNormalization(axis=chanDim))
model_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_bn.add(Dropout(0.25))

# second CONV => RELU => CONV => RELU => POOL layer set
model_bn.add(Conv2D(64, (3, 3), padding="same"))
model_bn.add(Activation("relu"))
model_bn.add(BatchNormalization(axis=chanDim))
model_bn.add(Conv2D(64, (3, 3), padding="same"))
model_bn.add(Activation("relu"))
model_bn.add(BatchNormalization(axis=chanDim))
model_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_bn.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model_bn.add(Flatten())
model_bn.add(Dense(512))
model_bn.add(Activation("relu"))
model_bn.add(BatchNormalization())
model_bn.add(Dropout(0.5))

# softmax classifier
model_bn.add(Dense(10))
model_bn.add(Activation("softmax"))

print(model_bn.summary())

# from keras.utils.vis_utils import plot_model
# plot_model(model_bn, to_file=PATH_DRIVE+'model_bn_plot.png', show_shapes=True, show_layer_names=True)


model_gn = Sequential()
model_gn.add(Conv2D(32, (3, 3), padding="same",
                    input_shape=(28, 28, 1), kernel_regularizer=ws_reg))
model_gn.add(Activation("relu"))
model_gn.add(GroupNormalization(axis=chanDim))
model_gn.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=ws_reg))
model_gn.add(Activation("relu"))
model_gn.add(GroupNormalization(axis=chanDim))
model_gn.add(MaxPooling2D(pool_size=(2, 2)))
model_gn.add(Dropout(0.25))

# second CONV => RELU => CONV => RELU => POOL layer set
model_gn.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=ws_reg))
model_gn.add(Activation("relu"))
model_gn.add(GroupNormalization(axis=chanDim))
model_gn.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=ws_reg))
model_gn.add(Activation("relu"))
model_gn.add(GroupNormalization(axis=chanDim))
model_gn.add(MaxPooling2D(pool_size=(2, 2)))
model_gn.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model_gn.add(Flatten())
model_gn.add(Dense(512))
model_gn.add(Activation("relu"))
model_gn.add(GroupNormalization())
model_gn.add(Dropout(0.5))

# softmax classifier
model_gn.add(Dense(10))
model_gn.add(Activation("softmax"))

print(model_gn.summary())


def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    # kernel_std = tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_std')
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)
    return kernel


def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))


model_bn.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

model_gn.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

"""## Train the model

Training the neural network model requires the following steps:

1. Feed the training data to the model—in this example, the `train_images` and `train_labels` arrays.
2. The model learns to associate images and labels.
3. We ask the model to make predictions about a test set—in this example, the `test_images` array. We verify that the predictions match the labels from the `test_labels` array.

To start training,  call the `model.fit` method—the model is "fit" to the training data:
"""

nb_epochs = 10
batch_s = 128

history = model_bn.fit(train_images, train_labels, validation_data=(validation_images, validation_labels),
                       epochs=nb_epochs, batch_size=batch_s)

with open(PATH_DRIVE + 'trainHistoryDict_bn_e{}_bs{}'.format(nb_epochs, batch_s), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# history = pickle.load( open(PATH_DRIVE+'trainHistoryDict_bn_e{}_bs{}'.format(nb_epochs, batch_s), "rb" ) )

model_bn.save_weights(PATH_DRIVE + 'my_model_weights_bn_ee{}_bs{}.h5'.format(nb_epochs, batch_s))

# model_bn.load_weights(PATH_DRIVE + 'my_model_weights_bn_ee{}_bs{}.h5'.format(nb_epochs, batch_s))

history_test = model_gn.fit(train_images, train_labels, validation_data=(validation_images, validation_labels),
                            epochs=nb_epochs, batch_size=batch_s)

# import pickle
with open(PATH_DRIVE + 'trainHistoryDict_gn_ws_e{}_bs{}'.format(nb_epochs, batch_s), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# history_test = pickle.load( open(PATH_DRIVE+'trainHistoryDict_gn_ws_e{}_bs{}'.format(nb_epochs, batch_s), "rb" ) )

model_gn.save_weights(PATH_DRIVE + 'my_model_weights_gn_ws_e{}_bs{}.h5'.format(nb_epochs, batch_s))

# model_gn.load_weights(PATH_DRIVE + 'my_model_weights_gn_ws_e{}_bs{}.h5'.format(nb_epochs, batch_s))

# history_bn = model_bn.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), epochs=nb_epochs, batch_size=32)

"""As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.

# Visualize
"""


def plot(history, history_bn, nb_epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    acc_values = history_dict['acc']
    val_loss_values = history_dict['val_loss']
    val_acc_values = history_dict['val_acc']

    history_dict_bn = history_bn.history
    loss_values_bn = history_dict_bn['loss']
    acc_values_bn = history_dict_bn['acc']
    val_loss_values_bn = history_dict_bn['val_loss']
    val_acc_values_bn = history_dict_bn['val_acc']

    first_epoch = 0
    epochs = range(first_epoch + 1, nb_epochs + 1)

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(epochs, loss_values[first_epoch:], 'b', label='Training loss GN+WS')
    ax.plot(epochs, val_loss_values[first_epoch:], 'r', label='Validation loss GN+WS')
    ax.plot(epochs, loss_values_bn[first_epoch:], 'c', label='Training loss BN')
    ax.plot(epochs, val_loss_values_bn[first_epoch:], 'g', label='Validation loss BN')
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    plt.show()

    plt.clf()

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(epochs, acc_values[first_epoch:], 'b', label='Training Accuracy GN+WS')
    ax.plot(epochs, acc_values_bn[first_epoch:], 'c', label='Training Accuracy BN')
    ax.plot(epochs, val_acc_values[first_epoch:], 'r', label='Validation Accuracy GN+WS')
    ax.plot(epochs, val_acc_values_bn[first_epoch:], 'g', label='Validation Accuracy BN')
    plt.title("Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    plt.show()

    plt.show()


plot(history_test, history, nb_epochs)

test_loss, test_acc = model_bn.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

test_loss, test_acc = model_gn.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
