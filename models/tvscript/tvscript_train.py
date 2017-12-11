import os
import tensorflow as tf
import tvscript_helper as helper

from tvscript_model import RNN

'''
Trains the RNN Model
'''

CHECKPOINTS_DIR = 'checkpoints/'


def create_checkpoints_dir():
    '''
    Creates the checkpoints directory if it does not exist
    '''
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)


def train(net,
          int_text,
          num_epochs: int=50,
          batch_size: int=32,
          seq_length: int=50,
          show_every_n_batches: int=64):

    saver = tf.train.Saver()

    batches = helper.get_batches(int_text, batch_size, seq_length)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            state = sess.run(net.initial_state,
                             {net.input_data: batches[0][0]})

            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    net.input_data: x,
                    net.targets: y,
                    net.initial_state: state}

                # Single pass
                train_loss, state, _ = sess.run([net.cost,
                                                 net.final_state,
                                                 net.train_op],
                                                feed)

                if (epoch_i * len(batches) +
                        batch_i) % show_every_n_batches == 0:
                    print(
                        'Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i, batch_i, len(batches), train_loss))

        saver.save(sess, './checkpoints/tvscript.ckpt')
        print('Model Trained and Saved')

        # Save seq_length and checkpoints dir for generating a new TV
        # script
        helper.save_params((seq_length, CHECKPOINTS_DIR))

    return train_loss


def main():
    create_checkpoints_dir()

    # Get data
    dataset_path = 'dataset'

    if not os.path.isdir('data'):
        print("Preprocessing and saving data...")
        helper.preprocess_and_save_data(dataset_path)

    # Load preprocessed data
    int_text, vocab_to_int, int_to_vocab,\
        token_dict = helper.load_preprocess()

    # Set Hyperparamters
    rnn_size = 350
    embed_dim = 200
    seq_length = 50
    learning_rate = 1.e-3
    vocab_size = len(int_to_vocab)

    # Initialize the model
    tf.reset_default_graph()
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    net = RNN(input_data, learning_rate,
              vocab_size, rnn_size,
              embed_dim, seq_length)

    train(net, int_text, seq_length, show_every_n_batches=64)


if __name__ == "__main__":
    main()
