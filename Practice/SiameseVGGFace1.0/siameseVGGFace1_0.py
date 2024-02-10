import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)


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

class L1Dist(Layer):
  def __init__(self, **kwargs):
    super().__init__()

def call(self, input_embedding, validation_embedding):
  return tf.math.abs(input_embedding - validation_embedding)

def make_siamese_model():
  # Handle inputs
  input_image = Input(name='input_img', shape=(250,250,3))
  validation_image = Input(name='validation_img', shape=(250,250,3))

  # Combine siamese distance components
  siamese_layer = L1Dist()
  siamese_layer._name = 'distance'
  distances = siamese_layer(embedding(input_image), embedding(validation_image))

  # Classification Layer
  classifier = Dense(1, activation='sigmoid')(distances)

  return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
print(siamese_model.summary())

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(batch):
  with tf.GradientTape() as tape:
    #Get anchor and positive/negative image
    X = batch[:2]
    #Get label
    y = batch[2]

    # Forward pass
    yhat = siamese_model(X, training=True)
    #Calculate loss
    loss = binary_cross_loss(y, yhat)

  # Calculate gradients
  grad = tape.gradient(loss, siamese_model.trainable_variables)

  # Calculate updated weights and apply to siamese model
  opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

  gc.collect()

  return loss

from tensorflow.keras.metrics import Precision, Recall, Accuracy

def convert(y_hat):
  for i in range(len(y_hat)):
    if (y_hat[i] > 0.5):
      y_hat[i] = 1.0
    else:
      y_hat[i] = 0
  return y_hat

def train(data, EPOCHS):
  # Loop through epochs
  for epoch in range(1, EPOCHS+1):
    print('\n Epoch {}/{}'.format(epoch, EPOCHS))
    progbar = tf.keras.utils.Progbar(len(data))

    a = Accuracy()

    # Loop through each batch
    for idx, batch in enumerate(data):
      # Run train step here
      loss = train_step(batch)
      yhat = siamese_model.predict(batch[:2], verbose=0)
      a.update_state(batch[2], convert(yhat))
      gc.collect()
      progbar.update(idx+1)

    print(loss.numpy(), a.result().numpy())
    gc.collect()
    #Save checkpoints
    #if epoch % 10 == 0:
      #checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 100
train(data, EPOCHS)

siamese_model.save('siamesemodelv4.h5')