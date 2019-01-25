# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from mlxtend.data import loadlocal_mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
X, y = loadlocal_mnist(
        images_path='C:\\Hiep-working\\NeuralNetwork\\Fashion_MNIST\\data\\train-images-idx3-ubyte', 
        labels_path='C:\\Hiep-working\\NeuralNetwork\\Fashion_MNIST\\data\\train-labels-idx1-ubyte')

print('Dimensions X: %s x %s' % (X.shape[0], X.shape[1]))
print('Dimensions y: %s' % (y.shape[0]))
#print('\n1st row', X[0])
