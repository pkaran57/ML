import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_core.python.keras.layers import Dense

# disable GPU support
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# network and training parameters
NUM_CLASSES = 10   # number of outputs = number of digits
INPUT_SHAPE = 784
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
"""
There are three ways of creating a model in tf.keras: Sequential API , Functional API, and Model subclassing. 
A Sequential() model is a linear pipeline (a stack) of neural network layers.
"""
model = tf.keras.models.Sequential()

"""
This code fragment defines a single layer called "dense_layer" with 10 artificial neurons that expects 784 input variables (also known as features).
Each neuron can be initialized with specific weights via the kernel_initializer parameter.
"""
model.add(Dense(units=NUM_CLASSES,
                input_shape=(INPUT_SHAPE,),
                kernel_initializer=tf.keras.initializers.zeros, # https://www.tensorflow.org/api_docs/python/tf/keras/initializers
                name='dense_layer',
                activation=tf.keras.activations.softmax))       # https://www.tensorflow.org/api_docs/python/tf/keras/activations

model.compile()

# Network and training parameters.
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
# Loading MNIST dataset.
# verify
# You can verify that the split between train and test is 60,000, and 10,000 respectively.
# Labels have one-hot representation.is automatically applied
mnist = tfds.image
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_train is 60000 rows of 28x28 values; we  --> reshape it to
# 60000 x 784.
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalize inputs to be within in [0, 1].
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# One-hot representation of the labels.
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)