import numpy as np
import os
import pickle

from collections import Counter


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)

    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict),
                open('data/preprocess.p', 'wb'))

    # For tensorflow serving
    with open('data/vocab_hash.txt', 'w') as f:
        for k, v in vocab_to_int.items():
            f.write("{} {}\n".format(k, v))


def load_preprocess():
    """
    Load the Preprocessed Training data
    and return them in batches of <batch_size> or less
    """
    return pickle.load(open('data/preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('data/params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('data/params.p', mode='rb'))


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)

    vocab_to_int = {word: i for i, word in enumerate(vocab, 0)}
    int_to_vocab = {i: word for i, word in enumerate(vocab, 0)}
    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation
        and the value is the token
    """
    punc_to_token = dict([
        ('--', '||dash||'), ('.', '||period||'), (',', '||comma||'),
        ('"', '||quotation_mark||'), (';', '||semicolon||'),
        ('!', '||exclamation_mark||'), ('?', '||question_mark||'),
        ('(', '||left_parentheses||'), (')', '||right_parentheses||'),
        ('\n', '||return||')
    ])
    return punc_to_token


def get_batches(int_text, n_steps, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # Get batch size and number of batches we can make
    batch_size = n_steps * seq_length
    n_batches = len(int_text) // batch_size

    # Keep only enough input data for full batches
    features = np.array(int_text[:n_batches * batch_size])
    targets = np.array(int_text[1:n_batches * batch_size + 1])
    targets[-1] = features[0]

    batch_x = np.split(features.reshape(n_steps, -1), n_batches, 1)
    batch_y = np.split(targets.reshape(n_steps, -1), n_batches, 1)

    batches = np.array(list(zip(batch_x, batch_y)))

    return batches


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys
        and words as the values
    :return: String of the predicted word
    """
    t = np.cumsum(probabilities)
    rand_s = np.sum(probabilities) * np.random.rand(1)
    pred_word = int_to_vocab[int(np.searchsorted(t, rand_s))]

    return pred_word


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = loaded_graph.get_tensor_by_name("input:0")
    InitialStateTensor = loaded_graph.get_tensor_by_name("initial_state:0")
    FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")
    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor
