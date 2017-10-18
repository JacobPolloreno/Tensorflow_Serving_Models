import os
import pickle
import tensorflow as tf

from cifar_helper import preprocess_and_save_data,\
        load_preprocess_training_batch
from cifar_get_data import get_dataset
from cifar_model import CNN
from typing import Tuple


CHECKPOINTS_DIR = 'checkpoints/'


def create_checkpoints_dir():
    """Creates the checkpoints directory if it does not exist

    """
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)


def train(net,
          num_epochs: int=100,
          batch_size: int=5000,
          n_batches: int=5,
          keep_prob: float=0.3,
          classes: int=10,
          image_dim: Tuple=(32, 32, 3)):

    # Get validation data
    valid_features, valid_labels = pickle.load(
        open('data/preprocess_validation.p', mode='rb'))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            for batch_i in range(1, n_batches + 1):
                for batch_features,\
                        batch_labels in load_preprocess_training_batch(
                        batch_i, batch_size):
                    feed = {
                        net.x: batch_features,
                        net.y: batch_labels,
                        net.keep_prob: keep_prob}
                        # net.learning_rate: learning_rate}

                # Single pass
                sess.run(net.opt, feed)

                # Get Metrics
                feed.update({net.keep_prob: 1.})
                loss = sess.run(net.cost, feed)
                feed.update({net.x: valid_features, net.y: valid_labels})
                valid_acc = sess.run(net.accuracy, feed)

                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(
                    epoch_i + 1, batch_i), end='')
                print(
                    'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                        loss, valid_acc))

        saver.save(sess, './checkpoints/cifar.ckpt')
        print('Model Trained and Saved')

    return loss, valid_acc


def main():
    create_checkpoints_dir()

    # Get data
    dataset_path = 'cifar-10-batches-py'

    if not os.path.isdir(dataset_path):
        print("Getting data...")
        get_dataset(dataset_path)

    if not os.path.isdir('data'):
        print("Preprocessing and saving data...")
        preprocess_and_save_data(dataset_path)

    # Initialize the model
    tf.reset_default_graph()
    image_shape = (32, 32, 3)
    learning_rate = 1.e-3
    _input = tf.placeholder(tf.float32, (None, *image_shape), name='x')
    net = CNN(_input, learning_rate)

    _, _ = train(net)


if __name__ == "__main__":
    main()
