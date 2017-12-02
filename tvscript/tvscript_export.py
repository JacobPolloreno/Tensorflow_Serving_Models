import os
import tensorflow as tf
import shutil

from tvscript_model import RNN
from tensorflow.contrib.lookup import HashTable, TextFileInitializer
from tensorflow.python.lib.io import file_io

'''
Loads the RNN model, injects additional layers for
the input transformation and export the models
into protobuf format
'''

# Command line arguments
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', './tvscript_export',
                           """Directory where to export the model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
FLAGS = tf.app.flags.FLAGS


def _write_assets(assets_directory, assets_filename):
    """Writes asset files to be used with SavedModel for half plus two.
    Args:
    assets_directory: The directory to which the assets should be written.
    assets_filename: Name of the file to which the asset contents should be
        written.
    Returns:
    The path to which the assets file was written.
    """
    if not file_io.file_exists(assets_directory):
        file_io.recursive_create_dir(assets_directory)

    path = os.path.join(
        tf.compat.as_bytes(assets_directory),
        tf.compat.as_bytes(assets_filename))
    # file_io.write_string_to_file(path, "asset-file-contents")
    return path


def main(_):

    with tf.Graph().as_default():
        # Create assets file that can be saved and restored
        assets_dir = './assets/'
        assets_filename = 'vocab_hash.txt'
        assets_filepath = _write_assets(assets_dir,
                                        assets_filename)

        # Setup assets collection
        assets_filepath = tf.constant(assets_filepath)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS,
                             assets_filepath)
        filename_tensor = tf.Variable(
            assets_filename,
            name="filename_tensor",
            trainable=False,
            collections=[])
        assign_filename_op = filename_tensor.assign(assets_filename)

        # Inject the placeholder into the graph
        serialized_tf_example = tf.placeholder(tf.string,
                                               name="tf_example")
        feature_configs = {
            'text': tf.FixedLenFeature(
                shape=[],
                dtype=tf.string)
        }
        tf_example = tf.parse_example(
            serialized_tf_example,
            feature_configs)
        text = tf_example['text']

        table = HashTable(
            TextFileInitializer('assets/vocab_hash.txt', tf.string,
                                0, tf.int32, 1), -1)

        def get_tokens(text_tensor):
            words = tf.string_split([text_tensor], " ")
            keys = tf.sparse_tensor_to_dense(words, default_value='')
            tokens = table.lookup(keys)
            return tokens

        int_text = tf.map_fn(get_tokens, text, dtype=tf.int32)
        int_text = tf.squeeze(int_text, [0])

        # Create the model
        learning_rate = 1.e-3
        net = RNN(int_text, learning_rate, vocab_size=6779)

        # Reverse lookup
        reverse_table = HashTable(
            TextFileInitializer('assets/vocab_hash.txt', tf.int64,
                                1, tf.string, 0), 'UNK')
        values, indices = tf.nn.top_k(net.probs, 1)
        pred_classes = reverse_table.lookup(tf.to_int64(indices))

        # Create the saver to restore from checkpoints
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Create export directory
            export_path = os.path.join(
                tf.compat.as_bytes(FLAGS.output_dir),
                tf.compat.as_bytes(str(FLAGS.model_version)))
            if os.path.exists(export_path):
                shutil.rmtree(export_path)

            # Create Model Builder
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # Create Tensors Info
            predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(
                text)
            # predict_tensor_probs_info = tf.saved_model.utils.build_tensor_info(
            #     net.probs)
            # TODO: Change output from net.probs to labels
            predict_tensor_label_info = tf.saved_model.utils.build_tensor_info(
                pred_classes)

            # Build prediction signature
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'text': predict_tensor_inputs_info},
                    outputs={'labels': predict_tensor_label_info},
                    # outputs={'probs': predict_tensor_probs_info,
                    #          'labels': predict_tensor_label_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

            # Save the model
            legacy_init_op = tf.group(tf.tables_initializer(),
                                      assign_filename_op,
                                      name="legacy_init_op")
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_labels': prediction_signature
                },
                assets_collection=tf.get_collection(
                    tf.GraphKeys.ASSET_FILEPATHS),
                # main_op=tf.group(assign_filename_op),
                legacy_init_op=legacy_init_op)

            builder.save()

    print("Successfully exported RNN model version'{}' into '{}'".format(
        FLAGS.model_version, FLAGS.output_dir))


if __name__ == '__main__':
    tf.app.run()
