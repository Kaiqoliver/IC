# -*- coding: utf-8 -*-
"""MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tLnCg5ClShp2GrhApmKKZdr7uze4t8OI

Let's start by creating the core objects of our model. It will have three layers. The input will have 784 nodes, the first layer will have 256 nodes, the second will have 128, the output layer will have 10 nodes.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

input_shape = 28*28
num_layers = 3
layers_shape = [input_shape , 200, 80, 10]

"""Data prep:"""

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to have mean 0 and standard deviation 1
train_images = (train_images / 255.0 - 0.5) / 0.5
test_images = (test_images / 255.0 - 0.5) / 0.5

# Reshape images
train_images = train_images.reshape(-1, 28 * 28, 1)
test_images = test_images.reshape(-1, 28 * 28, 1)

# Add a bias term (ones column) to the input data
train_images = np.concatenate([np.ones((train_images.shape[0], 1, 1)), train_images], axis=1)
test_images = np.concatenate([np.ones((test_images.shape[0], 1, 1)), test_images], axis=1)

# One-hot encode the labels
num_classes = 10  # Number of classes (digits 0-9)
train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes)
test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes)

# Reshape the one-hot encoded labels to (num_samples, num_classes, 1)
train_labels_one_hot = train_labels_one_hot.reshape(-1, num_classes, 1)
test_labels_one_hot = test_labels_one_hot.reshape(-1, num_classes, 1)

"""Initializing parameters:"""

def init_params():
  #np.random.seed(42)
  weight_list = []

  for i in range(1, num_layers + 1):
    weight_list.append(np.random.uniform(-1,1,size=(layers_shape[i-1]+1,layers_shape[i])))

  return weight_list

def check_vector_type(vector):
    shape = vector.shape
    if len(shape) == 2:
        if shape[0] == 1:
            print("Row Vector")
        elif shape[1] == 1:
            print("Column Vector")
        else:
            print("Neither Row nor Column Vector")
    else:
        print("Neither Row nor Column Vector")

"""Forward Propagation function:"""

def ReLU(Z):
  return np.maximum(0,Z)

def Sigmoid(z):
  return 1.0/(1.0 + np.exp(-z))

def Tahn(z):
  return np.tanh(z)

def forward_propagation(weight_list, x):
  signal_list = []
  output_list = []

  signal_list.append((weight_list[0].T.dot(x)))
  output_list.append(np.vstack((np.array([[1]]), ReLU(signal_list[0]))))

  for i in range(1, num_layers-1):
    signal_list.append(weight_list[i].T.dot(output_list[i-1]))
    output_list.append(np.vstack((np.array([[1]]), Sigmoid(signal_list[i]))))

  signal_list.append(weight_list[-1].T.dot(output_list[-1]))
  output_list.append(Sigmoid(signal_list[-1]))
  #output_list.append(np.vstack((np.array([[1]]), Sigmoid(signal_list[-1]))))

  return signal_list, output_list

"""Backward Propagation function:"""

def Sigmoid_linha(z):
  return Sigmoid(z)*(1 - Sigmoid(z))

def ReLU_linha(z):
  return z > 0

def Tahn_linha(z):
  return 1 - z**2
def backpropagation(weight_list, signal_list, output_list, y):
  sensitivity_list = []
  for i in range(num_layers):
    sensitivity_list.append(np.zeros(1))
  sensitivity_list[-1] = 2 * (output_list[-1] - y) * Sigmoid_linha(signal_list[-1])
  #print(sensitivity_list[-1].shape)
  for i in range(num_layers-2, -1, -1):
    sensitivity_list[i] = Sigmoid_linha(signal_list[i]) * (weight_list[i+1].dot(sensitivity_list[i+1])[1:])

  return sensitivity_list

"""Algorithm to compute the error and the gradients"""

def gradients(weight_list, X, Y):
  #N = X.shape[0]
  N = len(X)
  error = 0.0
  G_list = []
  for i in range(num_layers):
    G_list.append(0.0*weight_list[i])

  for i in range(N):
    #signal_list, output_list = forward_propagation(weight_list, X[i].reshape(-1, 1))
    signal_list, output_list = forward_propagation(weight_list, X[i])
    sensitivity_list = backpropagation(weight_list, signal_list, output_list, Y[i])
    error += 1/N * np.sum((output_list[-1] - Y[i])**2)

    #G_xn = X[i].reshape(-1, 1).dot(sensitivity_list[0].reshape(-1, 1).T)
    G_xn = X[i].dot(sensitivity_list[0].T)
    #print(G_list[0])
    G_list[0]+=1/N * G_xn
    #print(G_list[0])

    for j in range(1, len(weight_list)):
      G_i = output_list[j-1].dot(sensitivity_list[j].T)
      #print(G_list[j])
      G_list[j] += 1/N * G_i
      #print(G_list[j])

  return error, G_list

"""Update the weights"""

def update_weights(weight_list, G_list, learning_rate):
  for i in range(num_layers):
    #weight_list[i] += learning_rate * (G_list[i]/np.linalg.norm(G_list[i]))
    weight_list[i] += learning_rate * (G_list[i])

  return weight_list

"""Create the metrics:"""

def get_prediction(y_n):
  return np.argmax(y_n)

def get_accuracy(weight_list, X_test, Y_test):
  cont = 0 ; m = len(X_test)
  for i in range(m):
    signal_list, output_list = forward_propagation(weight_list, X_test[i])
    y_hat = get_prediction(output_list[-1])
    y = get_prediction(Y_test[i])
    if y_hat == y:
      cont += 1
  return 100 * cont/m

"""Organizing the learning process:"""

from tqdm import tqdm

def fit(X, Y, X_test, Y_test, learning_rate, batch, epochs_batch):
  weight_list = init_params()
  error_list = []
  accuracy_list = []
  for i in range(epochs_batch):
    error = 0.0
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_rand = X[indices]
    Y_rand = Y[indices]
    for j in tqdm(range(0, X.shape[0], batch)):
    #for j in range(0, X.shape[0], batch):
      error_aux, G_list = gradients(weight_list, X_rand[j:j+batch], Y_rand[j:j+batch])
      weight_list = update_weights(weight_list, G_list, learning_rate)
      error += error_aux
    error_list.append(error)
    accuracy_test = get_accuracy(weight_list, X_test, Y_test)
    accuracy_train = get_accuracy(weight_list, X, Y)
    accuracy_list.append(accuracy_test)
    #if i % 10 == 0 or i == epochs_batch:
    print(f"Epoch {i}: accuracy test={accuracy_test}%; accuracy train={accuracy_train}%; error={error}")

  return weight_list, error_list, accuracy_list

"""Finally:"""

weight_list, error_list, accuracy_list = fit(train_images, train_labels_one_hot, test_images, test_labels_one_hot, 0.01, 64, 10)

W1 = [[0.1, 0.2],
      [0.3, 0.4]]
W2 = [[0.2],
      [1],
      [-3]]
W3 = [[1],
      [2]]
weight_list = [np.array(W1), np.array(W2), np.array(W3)]
x = np.array([[1],[2]])
y = np.array([[1]])
X = [x]
Y = [y]
signal_list, output_list = forward_propagation(weight_list,x)
print(signal_list)
print(output_list)

sense_list = backpropagation(weight_list, signal_list, output_list, y)
print(sense_list)

error, G_list = gradients(weight_list, X, Y)
print(G_list)