import numpy as np
import tensorflow as tf
import tvscript_helper as helper


def main(_):
    _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    seq_length, load_dir = helper.load_params()

    # Generate TV Script
    gen_length = 300
    prime_word = 'homer_simpson'

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load the saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get tensors for loaded model
        input_text, inital_state, final_state, probs = helper.get_tensors(
            loaded_graph)

        # Setneces generation setup
        gen_sentences = [prime_word + ':']
        prev_state = sess.run(inital_state, {input_text: np.array([[1]])})

        # Generate sentences
        for _ in range(gen_length):
            # Dyanmic input
            dyn_input = [[vocab_to_int[word] for word in
                          gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Predictions
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, inital_state: prev_state})

            pred_word = helper.pick_word(
                probabilities[0, dyn_seq_length - 1], int_to_vocab)

            gen_sentences.append(pred_word)

        # Remove Tokens
        tv_script = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            # ending = ' ' if key in ['\n', '(', '"'] else ''
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ', '(')

        print(tv_script)


if __name__ == '__main__':
    tf.app.run()
