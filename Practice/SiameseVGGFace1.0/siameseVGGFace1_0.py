import keras
import tensorflow as tf

print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)


import os
import random
import numpy as np
from matplotlib import pyplot as plt
import gc

from keras.models import  Model
from keras.layers import Layer, Flatten, Dense, Input, Lambda
import keras.backend as K
from sklearn.datasets import fetch_lfw_pairs

train_data = fetch_lfw_pairs(subset='train', funneled=False, resize=1.0, color=True, slice_=(slice(0, 250), slice(0, 250)))
#test_data = fetch_lfw_pairs(subset= 'test', funneled=False, resize=1.0, color=True, slice_=(slice(0, 250), slice(0, 250)))

# Assuming train_data is the variable containing the LFW pairs dataset
images_train = train_data.pairs
labels_train = train_data.target

# Assuming images and labels are NumPy arrays
images = tf.convert_to_tensor(images_train)
labels = tf.convert_to_tensor(labels_train)

del train_data
del images_train
del labels_train
gc.collect()

# Assuming 'images' is your tensor with dimensions (2200, 2, 250, 250, 3)
images_shape = images.shape

# Reshape the tensor to (2200*2, 250, 250, 3) to treat each image as a separate entity
images_reshaped = tf.reshape(images, (-1, images_shape[2], images_shape[3], images_shape[4]))

# Separate the images into two tensors
first_images = images_reshaped[::2]  # Select every other image starting from the first
second_images = images_reshaped[1::2]  # Select every other image starting from the second

# Now, 'first_images' and 'second_images' are tensors with dimensions (2200, 250, 250, 3)

# Assuming 'first_images', 'second_images', and 'labels' are your tensors
first_images_dataset = tf.data.Dataset.from_tensor_slices(first_images)
second_images_dataset = tf.data.Dataset.from_tensor_slices(second_images)
labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

# Zip the datasets
data = tf.data.Dataset.zip((first_images_dataset, second_images_dataset, labels_dataset))

data = data.batch(16)
data = data.prefetch(8)

def make_embedding():
  #Getting the trained model
  vgg_model = siamese_model = tf.keras.models.load_model('vgg_model.h5')

  #Freezing its layers
  for layer in vgg_model.layers:
    layer.trainable = False

  input_layer = vgg_model.get_layer(None, 0).input
  last_layer = vgg_model.get_layer('pool5').output
  x = Flatten(name='flatten')(last_layer)
  x = Dense(512, activation='relu', name='fc6')(x)
  x = Dense(128, activation='relu', name='fc7')(x)

  return Model(inputs=[input_layer], outputs=[x], name='embedding')

embedding = make_embedding()