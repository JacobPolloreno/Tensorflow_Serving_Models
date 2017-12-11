import numpy as np
import tensorflow as tf

from typing import Tuple


class CNN(object):

    def __init__(self,
                 x,
                 learning_rate,
                 keep_prob: float=.3,
                 n_classes: int=10) -> None:

        self.x = x
        # self.x = tf.placeholder(tf.float32,
        #                         [None, image_shape[0],
        #                          image_shape[1], image_shape[2]], name='x')
        self.y = tf.placeholder(tf.float32, [None, n_classes], name='y')
        # self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.learning_rate = tf.Variable(learning_rate,
                                         trainable=False,
                                         name="learning_rate")
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(keep_prob, (),
                                                     name="keep_prob")

        self.logits, self.cost, self.accuracy,\
            self.pred_class = self.model_loss()
        self.opt = self.model_opt()

    def _conv2d_maxpool(self, x_tensor,
                        conv_num_outputs: int,
                        conv_ksize: Tuple,
                        conv_strides: Tuple,
                        pool_ksize: Tuple,
                        pool_strides: Tuple):
        """Apply convolution then max pooling to x_tensor

        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        :return:
            A tensor that represents convolution and max pooling of x_tensor
        """
        input_depth = x_tensor.get_shape().as_list()[-1]
        weight = tf.Variable(
            tf.truncated_normal(
                [conv_ksize[0], conv_ksize[1],
                 input_depth, conv_num_outputs],
                mean=0.0, stddev=0.1))
        bias = tf.Variable(tf.zeros(conv_num_outputs))

        def conv2d(x, W, b, strides):
            """Apply a 2d convolution to a x tensor

            :param x: Tensorflow x input
            :param W: weights applied to the convolution layer
            :param b: bias applied to convolution layer
            :parm strides: 2-D tuple specifying convolution kernel strides
            : return convolution of x tensor
            """
            x = tf.nn.conv2d(x, W,
                             strides=[1, strides[0], strides[1], 1],
                             padding='SAME')
            x = tf.nn.bias_add(x, b)
            x = tf.nn.relu(x)
            return x

        def maxpool2d(x, ksize, pstrides):
            """Apply max pooling to convoluted tensor using 2D kernel/stride

            :param x: convoluted tensor(output from conv2d())
            :param ksize: 2-D tuple for kernel size for pool
            :param strides: 2-D tuple for strides for pool
            : return max pooling of convoluted x tensor
            """
            return tf.nn.max_pool(conv, ksize=[1, ksize[0], ksize[1], 1],
                                  strides=[1, pstrides[0], pstrides[1], 1],
                                  padding='SAME')

        # Apply convolution and nonlinear activation
        conv = conv2d(x_tensor, weight, bias, conv_strides)

        # Max Pooling Implementation
        conv = maxpool2d(conv, pool_ksize, pool_strides)
        return conv

    def _flatten(self, x_tensor: np.ndarray):
        """Flatten x_tensor to (Batch Size, Flattened Image Size)

        :x_tensor: A tensor of size (Batch Size, ...),
                    where ... are the image dimensions.
        :return: A tensor of size (Batch Size, Flattened Image Size).
        """
        dim = x_tensor.get_shape().as_list()
        volume = dim[1] * dim[2] * dim[3]
        return tf.reshape(x_tensor, [-1, volume])

    def _fully_conn(self, x_tensor: np.ndarray, num_outputs: int):
        """Apply a fully connected layer to x_tensor using weight and bias

        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        weights = tf.Variable(
            tf.truncated_normal(
                [x_tensor.get_shape().as_list()[1],
                 num_outputs], mean=0.0, stddev=0.1))
        bias = tf.Variable(tf.zeros(num_outputs))

        fully_conn = tf.add(tf.matmul(x_tensor, weights), bias)
        fully_conn = tf.nn.relu(fully_conn)
        return fully_conn

    def _output(self, x_tensor: np.ndarray, num_outputs: int):
        """Apply a output layer to x_tensor using weight and bias

        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        weights = tf.Variable(
            tf.truncated_normal(
                [x_tensor.get_shape().as_list()[1],
                 num_outputs], mean=0.0, stddev=0.1))
        bias = tf.Variable(tf.zeros(num_outputs))
        return tf.add(tf.matmul(x_tensor, weights), bias)

    def build_nn(self):
        """Create a convolutional neural network model

        : return: Tensor that represents logits
        """
        # Apply Convolution and Max Pool layers
        # Layer 1 - 32x32x3 to x32
        conv1 = self._conv2d_maxpool(self.x,
                                     conv_num_outputs=16,
                                     conv_ksize=(3, 3),
                                     conv_strides=(1, 1),
                                     pool_ksize=(2, 2),
                                     pool_strides=(2, 2))
        # Layer 2 - x32 to  x64
        conv2 = self._conv2d_maxpool(conv1,
                                     conv_num_outputs=32,
                                     conv_ksize=(2, 2),
                                     conv_strides=(1, 1),
                                     pool_ksize=(2, 2),
                                     pool_strides=(2, 2))
        conv3 = self._conv2d_maxpool(conv2,
                                     conv_num_outputs=64,
                                     conv_ksize=(2, 2),
                                     conv_strides=(1, 1),
                                     pool_ksize=(2, 2),
                                     pool_strides=(2, 2))

        # Apply a Flatten Layer
        flat = self._flatten(conv3)

        # Apply Fully Connected Layers
        # Fully connected layer
        fc1 = self._fully_conn(flat, num_outputs=2048)
        fc1 = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = self._fully_conn(fc1, num_outputs=1024)
        fc2 = tf.nn.dropout(fc2, self.keep_prob)

        # Apply an Output Layer
        out = self._output(fc1, num_outputs=10)
        out = tf.identity(out, name='logits')

        return out

    def model_loss(self):
        """Get the loss

        """

        logits = self.build_nn()
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.y),
            name='cost')

        pred_class = tf.cast(tf.argmax(logits, 1), tf.int64, name="pred_class")
        correct_pred = tf.equal(pred_class,
                                tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
                                  name='accuracy')

        return logits, cost, accuracy, pred_class

    def model_opt(self):
        """Get optimization operations

        """
        optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                           name='opt').minimize(self.cost)

        return optimizer
