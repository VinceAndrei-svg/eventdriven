import numpy as np
import tensorflow as tf

print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

a = np.array([1.0, 2.0, 3.0])
b = tf.constant([1.0, 2.0, 3.0])

print("NumPy array:", a)
print("TensorFlow constant:", b)
