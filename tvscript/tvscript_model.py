import tensorflow as tf

from tensorflow.contrib import seq2seq


class RNN(object):
    def __init__(self,
                 input_data,
                 learning_rate,
                 vocab_size: int,
                 rnn_size: int=350,
                 embed_dim: int=200,
                 seq_length: int=50) -> None:
        """Init RNN

        :param input_data: TF placeholder for text input.
        :param vocab_size: Number of words in vocabulary.
        :param embed_dim: Number of embedding dimensions
        :param rnn_size: Size of RNNs
        """
        self.input_data = input_data
        self.targets = tf.placeholder(tf.int32,
                                      [None, None],
                                      name="targets")
        self.learning_rate = tf.Variable(learning_rate,
                                         trainable=False,
                                         name="learning_rate")

        self.embed_dim = embed_dim
        self.rnn_size = rnn_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        self.input_data_shape = tf.shape(input_data)
        self.cell, self.initial_state = self._get_init_cell(
            self.input_data_shape[0])

        self.logits, self.final_state,\
            self.cost, self.probs = self.model_loss()

        self.opt, self.train_op = self.model_opt()

    def _get_init_cell(self, batch_size):
        """Create an RNN Cell and initialize it.

        :return: Tuple (cell, initialize state)
        """
        use_layers = False

        if use_layers:
            layer = []
            # Experiment with multiple layers
            for _ in range(3):
                lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
                layer.append(lstm)

            cell = tf.contrib.rnn.MultiRNNCell(layer)
        else:
            lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            cell = tf.contrib.rnn.MultiRNNCell([lstm])
        initial_state = cell.zero_state(batch_size, tf.float32)
        initial_state = tf.identity(initial_state, name='initial_state')
        return cell, initial_state

    def _get_embed(self):
        """Create embedding for <input_data>.

        :return: Embedded input.
        """
        embedding = tf.Variable(
            tf.random_uniform((self.vocab_size, self.embed_dim), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, self.input_data)
        return embed

    def _build_rnn(self, cell, inputs):
        """Create a RNN using a RNN Cell

        :param cell: RNN Cell
        :param inputs: Input text data
        :return: Tuple (Outputs, Final State)
        """
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        final_state = tf.identity(state, name='final_state')
        return outputs, final_state

    def build_nn(self):
        """Build part of the neural network

        :return: Tuple (Logits, FinalState)
        """
        inputs = self._get_embed()
        outputs, final_state = self._build_rnn(self.cell,
                                               inputs)
        logits = tf.contrib.layers.fully_connected(
            outputs,
            self.vocab_size,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(),
            biases_initializer=tf.zeros_initializer(),
            trainable=True)

        return logits, final_state

    def model_loss(self):
        logits, final_state = self.build_nn()

        probs = tf.nn.softmax(logits, name="probs")

        cost = seq2seq.sequence_loss(
            logits,
            self.targets,
            tf.ones([self.input_data_shape[0],
                     self.input_data_shape[1]]),
            name="cost")

        return logits, final_state, cost, probs

    def model_opt(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad,
                            var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

        return optimizer, train_op
