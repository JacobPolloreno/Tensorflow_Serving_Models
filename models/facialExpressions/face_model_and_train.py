import numpy as np
import pandas as pd
import tensorflow as tf

from collections import namedtuple
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

# FER 2013 data
#   consist of 48 x 48 grayscale images of faces
_DATASET_DIR = 'data/'
_DATASET_PATH = _DATASET_DIR + 'fer2013.csv'
_DATASET_DIRS = ['training', 'testing', 'validation']
_IMAGE_DIM = (48, 48)
_LABELS = ['angry', 'disgust', 'fear', 'happy',
           'sad', 'suprise', 'neutral']

_TESTING_IDX_START = 28709
_VALIDATION_IDX_START = 32298

Conv = namedtuple(
    'Conv', ['kernel_size', 'strides', 'filters', 'padding', 'name'])
Pool = namedtuple(
    'Pool', ['pool_size', 'strides', 'name'])

_CONV_DEFS = [
    Conv(kernel_size=[3, 3], strides=1, filters=32,
         padding='same', name="conv1"),
    Pool(pool_size=[2, 2], strides=2, name="pool1"),
    Conv(kernel_size=[3, 3], strides=1, filters=64,
         padding='same', name="conv2"),
    Pool(pool_size=[2, 2], strides=2, name="pool2"),
    Conv(kernel_size=[3, 3], strides=1, filters=128,
         padding='same', name="conv3"),
    Pool(pool_size=[2, 2], strides=2, name="pool3"),
]


def cnn_model_fn(features,
                 labels,
                 mode,
                 params):
    """Model function for CNN

    """

    conv_defs = params['conv_defs']
    num_classes = params['num_classes']

    # Define Hyperparams
    kp = tf.constant(params['keep_prob'],
                     dtype=tf.float32,
                     name="keep_prob")
    lr = tf.constant(params['learning_rate'],
                     dtype=tf.float32,
                     name="learning_rate")

    conv_defs_iterator = iter(conv_defs)

    # Input Layer
    # Reshape to 4-D tensor: [batch_size, width, height, channels]
    # Face images are 48x48 pixels and one color channel
    input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])

    # Convolution Layer #1
    # Input Tensor Shape: [batch_size, 48, 48, 1]
    # Output Tensor Shape: [batch_size, 48, 48, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        activation=tf.nn.relu,
        **next(conv_defs_iterator)._asdict())

    # batch_norm1 = tf.layers.batch_normalization(
    #     conv1,
    #     training=mode == tf.estimator.ModeKeys.TRAIN,
    #     name="batch_norm1")

    # conv1 = tf.nn.relu(batch_norm1)

    # Pooling Layer #1
    # 2x2 filter with stride of 2
    # Input Tensor Shape: [batch_size, 48, 48, 32]
    # Output Tensor Shape: [batch_size, 24, 24, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        **next(conv_defs_iterator)._asdict())

    # Convolution Layer #2
    # 64 filters using a 5x5 filter
    # Input Tensor Shape: [batch_size, 24, 24, 32]
    # Output Tensor Shape: [batch_size, 24, 24, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        activation=tf.nn.relu,
        **next(conv_defs_iterator)._asdict())

    # batch_norm2 = tf.layers.batch_normalization(
    #     conv2,
    #     training=mode == tf.estimator.ModeKeys.TRAIN,
    #     name="batch_norm2")

    # conv2 = tf.nn.relu(batch_norm2)

    # Pooling Layer #2
    # Second pooling layer with 2x2 filter and stride 2
    # Input Tensor Shape: [batch_size, 24, 24, 64]
    # Output Tensor Shape: [batch_size, 12, 12, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        **next(conv_defs_iterator)._asdict())

    # Convolution Layer #3
    # 64 filters using a 5x5 filter
    # Input Tensor Shape: [batch_size, 12, 12, 64]
    # Output Tensor Shape: [batch_size, 12, 12, 128]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        activation=tf.nn.relu,
        **next(conv_defs_iterator)._asdict())

    # Pooling Layer #2
    # Second pooling layer with 2x2 filter and stride 2
    # Input Tensor Shape: [batch_size, 12, 12, 128]
    # Output Tensor Shape: [batch_size, 6, 6, 128]
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        **next(conv_defs_iterator)._asdict())

    # Flatten tensor into batch of vectors
    # Input Tensor Shape: [batch_size, 6, 6, 128]
    # Output Tensor Shape: [batch_size, 6 * 6 * 128]
    flatten_layer = tf.reshape(
        tensor=pool3,
        shape=[-1, 6 * 6 * 128])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 6 * 6 * 128]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=flatten_layer,
        units=2048,
        activation=tf.nn.relu,
        name="dense")

    # Add dropout
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=kp,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        name="dropout")

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, num_classes]
    logits = tf.layers.dense(
        inputs=dropout,
        units=num_classes,
        name="logits")

    predictions = {
        # Generate predictions
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Configure the Predict op (PREDICT MODE)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # Calculate Loss
    onehot_labels = tf.one_hot(
        indices=tf.cast(labels, tf.int32),
        depth=num_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=logits)

    # Configure the Training Op (TRAIN MODE)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    # Configure the Evaluate Op (EVAL MODE)
    eval_metric_ops = {
        # Add evaluation metrics (EVAL MODE)
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    data = pd.read_csv(_DATASET_PATH, usecols=['emotion', 'pixels'])
    features = np.zeros(shape=(len(data.pixels), 2304),
                        dtype=np.float32)

    for idx, row in enumerate(data.pixels):
        image_array = np.fromstring(
            str(row),
            dtype=np.uint8,
            sep=' ')
        features[idx] = image_array
    labels = np.asarray(data.emotion, dtype=np.int32)
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        features, labels, test_size=0.20, random_state=55)

    # Data Preprocessing
    training_mean = train_data.mean()
    train_data -= training_mean
    eval_data -= training_mean

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='./convnet_model/',
        params={'num_classes': len(_LABELS),
                'conv_defs': _CONV_DEFS,
                'learning_rate': 0.001,
                'keep_prob': 0.4})

    # Set up logging for prediction
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log,
    #     every_n_iter=2000)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=20000)
        # hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(
        input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
