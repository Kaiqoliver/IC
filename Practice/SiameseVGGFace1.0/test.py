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
print(embedding.summary())
print('Teste2')

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
