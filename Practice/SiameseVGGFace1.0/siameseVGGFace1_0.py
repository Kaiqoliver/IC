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


train_data = fetch_lfw_pairs(subset='train', funneled=False, resize=1.0, color=True, slice_=(slice(0, 250), slice(0, 250)))
test_data = fetch_lfw_pairs(subset= 'test', funneled=False, resize=1.0, color=True, slice_=(slice(0, 250), slice(0, 250)))

images_train = train_data.pairs
labels_train = train_data.target

images_test = test_data.pairs
labels_test = test_data.target



# Reshaping the dataset to (4400, 250, 250, 3)
reshaped_train_images = images_train.reshape((-1, 250, 250, 3))
reshaped_test_images = images_test.reshape((-1, 250, 250, 3))

# Calculating mean and variance for each channel within each image
train_mean_per_channel = np.mean(reshaped_train_images, axis=(1, 2), keepdims=True)
train_variance_per_channel = np.var(reshaped_train_images, axis=(1, 2), keepdims=True)

test_mean_per_channel = np.mean(reshaped_test_images, axis=(1, 2), keepdims=True)
test_variance_per_channel = np.var(reshaped_test_images, axis=(1, 2), keepdims=True)
 
# Normalizing each channel of each image separately based on its own mean and variance
normalized_train_images = (reshaped_train_images - train_mean_per_channel) / np.sqrt(train_variance_per_channel)
normalized_test_images = (reshaped_test_images - test_mean_per_channel) / np.sqrt(test_variance_per_channel)

# Reshape the normalized images back to the original shape
normalized_train_images = normalized_train_images.reshape((2200, 2, 250, 250, 3))
normalized_test_images = normalized_test_images.reshape((1000, 2, 250, 250, 3))

images_train = normalized_train_images
images_test = normalized_test_images

# Assuming images and labels are NumPy arrays
images_train_tensor = tf.convert_to_tensor(images_train)
labels_train_tensor = tf.convert_to_tensor(labels_train)


#del train_data
#del test_data
#del images_train
#del labels_train
#gc.collect()

# Assuming 'images' is your tensor with dimensions (2200, 2, 250, 250, 3)
images_train_shape = images_train_tensor.shape
images_test_shape = images_test.shape

# Reshape the tensor to (2200*2, 250, 250, 3) to treat each image as a separate entity
images_train_tensor_reshaped = tf.reshape(images_train_tensor, (-1, images_train_shape[2], images_train_shape[3], images_train_shape[4]))
images_test_reshaped = np.reshape(images_test, (-1, images_test_shape[2], images_test_shape[3], images_test_shape[4]))

# Separate the images into two tensors
first_train_images = images_train_tensor_reshaped[::2]  # Select every other image starting from the first
second_train_images = images_train_tensor_reshaped[1::2]  # Select every other image starting from the second

first_test_images = images_test_reshaped[::2]  # Select every other image starting from the first
second_test_images = images_test_reshaped[1::2]  # Select every other image starting from the second

# Now, 'first_images' and 'second_images' are tensors with dimensions (2200, 250, 250, 3)

# Assuming 'first_images', 'second_images', and 'labels' are your tensors
first_train_images_dataset = tf.data.Dataset.from_tensor_slices(first_train_images)
second_train_images_dataset = tf.data.Dataset.from_tensor_slices(second_train_images)
labels_train_dataset = tf.data.Dataset.from_tensor_slices(labels_train_tensor)

# Zip the datasets
data_train = tf.data.Dataset.zip((first_train_images_dataset, second_train_images_dataset, labels_train_dataset))


data_train = data_train.batch(16)
data = data_train.prefetch(8)

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
    # Get anchor and positive/negative image
    X = batch[:2]
    # Get label
    y = batch[2]

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

def train(data, EPOCHS):
  # Loop through epochs
  for epoch in range(1, EPOCHS+1):
    print('\n Epoch {}/{}'.format(epoch, EPOCHS))
    progbar = tf.keras.utils.Progbar(len(data), verbose=0)

    a = Accuracy()

    # Loop through each batch
    for idx, batch in enumerate(data):
      # Run train step here
      loss = train_step(batch)
      yhat = siamese_model.predict(batch[:2], verbose=0)
      a.update_state(batch[2], convert(yhat))
      gc.collect()
      progbar.update(idx+1)

    v = Accuracy()
    yhatv = siamese_model.predict([first_test_images, second_test_images], batch_size=16, verbose=0)
    v.update_state(labels_test, convert(yhatv))

    print(loss.numpy(), a.result().numpy(), v.result().numpy())
    gc.collect()

def train2(data, EPOCHS):
    
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Define a callback for progress bar
    progbar_callback = ProgbarLogger()

    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))

        a = Accuracy()

        # Train the model using fit method and include the ProgbarLogger callback
        history = siamese_model.fit(data, epochs=1, verbose=0, callbacks=[progbar_callback])

        # Extract loss and accuracy from the history
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]

        print(loss, accuracy)
        gc.collect()

        # Save checkpoints
        # if epoch % 10 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 100
train(data, EPOCHS)

siamese_model.save('siamesemodelv4.h5')