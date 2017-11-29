"""
Some code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

Transfer learning with Inception v3

Top layer has an input of 2048-dimensional vector for each image.
We train a softmax layer on top of this representation. Assuming
softmax contains N labels, we will learn N + 2048 * N parameters
for weights and biases.

Our folder structure is
~/dog_photos/train/
~/dog_photos/test/
~/dog_photos/valid/

... <breed>/photo1.jpg

"""

import tensorflow as tf

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1


