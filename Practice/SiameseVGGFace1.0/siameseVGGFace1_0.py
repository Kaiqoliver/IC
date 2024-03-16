import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import os
import random
import numpy as np
#from matplotlib import pyplot as plt
import gc

from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Layer, Flatten, Dense, Input, Lambda
import tensorflow.keras.backend as K
from sklearn.datasets import fetch_lfw_pairs
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

train_data = fetch_lfw_pairs(subset='10_folds', funneled=False, resize=1.0, color=True, slice_=(slice(0, 250), slice(0, 250)))
#test_data = fetch_lfw_pairs(subset= 'test', funneled=False, resize=1.0, color=True, slice_=(slice(0, 250), slice(0, 250)))

images_train = train_data.pairs
labels_train = train_data.target

#images_test = test_data.pairs
#labels_test = test_data.target

images_train, images_test, labels_train, labels_test = train_test_split(images_train, labels_train, test_size=0.1, random_state=42)

del train_data
#del test_data
gc.collect()

def calculate_means(pairs_rgb_dataset):
    means = [0.0, 0.0, 0.0]
    stds = [0.0, 0.0, 0.0]

    # Reshape the dataset for easier computation
    reshaped_dataset = np.reshape(pairs_rgb_dataset, (-1, 250, 250, 3))
    reshaped_dataset = np.concatenate(reshaped_dataset, axis=0)
    reshaped_dataset = np.concatenate(reshaped_dataset, axis=0)

    means[0] = np.mean(reshaped_dataset[:,0])
    means[1] = np.mean(reshaped_dataset[:,1])
    means[2] = np.mean(reshaped_dataset[:,2])

    stds[0] = np.std(reshaped_dataset[:,0])
    stds[1] = np.std(reshaped_dataset[:,1])
    stds[2] = np.std(reshaped_dataset[:,2])

    return means, stds

channel_means, channel_stds = calculate_means(images_train)

print(channel_means, channel_stds)

def normalize_images(images_train, channel_means, channel_stds):
  # Iterate through each pair of images
  for i in range(images_train.shape[0]):
      # Normalize the first image in the pair
      images_train[i, 0] = (images_train[i, 0] - channel_means) / channel_stds
      # Normalize the second image in the pair
      images_train[i, 1] = (images_train[i, 1] - channel_means) / channel_stds
  
  return images_train

# Create a generator for training data
images_train = normalize_images(images_train, channel_means, channel_stds)
images_test = normalize_images(images_test, channel_means, channel_stds)

def train_data_generator(images, labels):
    for i in range(images.shape[0]):
        yield images[i][0], images[i][1]

# Create TensorFlow datasets
data_train = tf.data.Dataset.from_tensor_slices((images_train[:, 0], images_train[:, 1], labels_train))
print(data_train)
data_train = data_train.batch(16)
data_train = data_train.prefetch(8)

del images_train
del labels_train
gc.collect()

data_test = tf.data.Dataset.from_tensor_slices((images_test[:,0], images_test[:,1] , labels_test))
data_test = data_test.batch(16)
data_test = data_test.prefetch(8)

del images_test
del labels_test
gc.collect()

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
    #print(batch)
    # Get anchor and positive/negative image
    X = batch[0]
    # Get label
    y = batch[1]

    # Forward pass
    yhat = siamese_model(X, training=True)
    # Calculate loss
    loss = binary_cross_loss(y, yhat)

  # Calculate gradients
  grad = tape.gradient(loss, siamese_model.trainable_variables)

  # Calculate updated weights and apply to siamese model
  opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

  # Clearing the tape explicitly to avoid the debug info message
  del tape
  gc.collect()

  return loss

from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.callbacks import ProgbarLogger

def convert(y_hat):
  for i in range(len(y_hat)):
    if (y_hat[i] > 0.5):
      y_hat[i] = 1.0
    else:
      y_hat[i] = 0
  return y_hat

def count_elements(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))

        # Count the number of elements in the dataset
        dataset_length = count_elements(data)
        progbar = tf.keras.utils.Progbar(dataset_length, verbose=0)

        a = Accuracy()

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Extract anchor and positive images from the batch
            input_img, validation_img, labels = batch
            # Concatenate the images along the batch dimension
            concatenated_imgs = (input_img, validation_img)
            # Run train step here
            loss = train_step((concatenated_imgs, labels))
            yhat = siamese_model.predict(concatenated_imgs, verbose=0)
            a.update_state(labels, convert(yhat))


            # Run train step here
            #loss = train_step(batch)
            #yhat = siamese_model.predict(batch, verbose=0)
            #a.update_state(batch[2], convert(yhat))
            gc.collect()
            progbar.update(idx+1)

        v = Accuracy()

        for idx, batch in enumerate(data_test):
            input_img, validation_img, labels = batch
            concatenated_imgs = (input_img, validation_img)
            yhat = siamese_model.predict(concatenated_imgs, verbose=0)
            v.update_state(labels, convert(yhat))
            gc.collect()


        print(loss.numpy(), a.result().numpy(), v.result().numpy())
        gc.collect()


EPOCHS = 100
train(data_train, EPOCHS)

siamese_model.save('siamesemodelv4.h5')