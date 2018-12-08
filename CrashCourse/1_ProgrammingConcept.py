# from __future__ import print_function
import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as pipnp
# import pandas as pd  

# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  z = tf.constant(4, name="z_const")
  my_sum = tf.add(tf.add(x, y, name="x_y_sum"), z, "x_y_z_sum")


  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    print(my_sum.eval())