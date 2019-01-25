# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from mlxtend.data import loadlocal_mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Load data using library
fashion_mnist = keras.datasets.fashion_mnist
(train_images2, train_labels2), (test_images2, test_labels2) = fashion_mnist.load_data()

print('Dimensions train_images2: %s x %s' % (train_images2.shape[0], train_images2.shape[1]))
print('Dimensions train_labels2: %s' % (train_labels2.shape[0]))

train_images, train_labels = loadlocal_mnist(
        images_path='C:\\Hiep-working\\NeuralNetwork\\Fashion_MNIST\\data\\train-images-idx3-ubyte', 
        labels_path='C:\\Hiep-working\\NeuralNetwork\\Fashion_MNIST\\data\\train-labels-idx1-ubyte')

test_images, test_labels = loadlocal_mnist(
        images_path='C:\\Hiep-working\\NeuralNetwork\\Fashion_MNIST\\data\\t10k-images-idx3-ubyte', 
        labels_path='C:\\Hiep-working\\NeuralNetwork\\Fashion_MNIST\\data\\t10k-labels-idx1-ubyte')

print('Dimensions train_images: %s x %s' % (train_images.shape[0], train_images.shape[1]))
print('Dimensions train_labels: %s' % (train_labels.shape[0]))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print("train_images.shape=", train_images.shape)      
# print("len(train_labels)=", len(train_labels))
# print(train_labels)
# print(test_images.shape)
# print(len(test_labels))

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)     
# plt.show()    

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Training...")
model.fit(train_images, train_labels, epochs=5)              
print("Training...Done")
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)


predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.show()
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)