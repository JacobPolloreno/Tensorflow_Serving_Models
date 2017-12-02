import os
import shutil
import tensorflow as tf

from cifar_model import CNN


# Command-line arguments
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                           """Directory containing trained checkpoints""")
tf.app.flags.DEFINE_string('output_dir', './cifar_export',
                           """Directory to export the model to""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
FLAGS = tf.app.flags.FLAGS


def preprocess_image(image_buffer):
    """Preprocess bytes into a 3D Tensor

    :param image_buffer: Buffer that contains JPEG image
    :return 4D image tensor (1, width, height, channels) with pixels
        scaled to [-1, 1].
    """
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.resize_images(image, [32, 32])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image


def main(_):

    with tf.Graph().as_default():

        # Inject placeholder into the graph
        serialized_tf_example = tf.placeholder(tf.string,
                                               name='input_image')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
        images = tf.squeeze(images, [0])

        # Create the model
        learning_rate = 1.e-3
        net = CNN(images, learning_rate, keep_prob=1.)

        # Create labels tensor and lookup
        labels = ['airplane', 'automobile', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        labels_tensor = tf.constant(labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            labels_tensor)

        # Get prob values with indices
        values, indices = tf.nn.top_k(net.logits, len(labels))
        prediction_classes = table.lookup(tf.to_int64(indices))

        # FIX issue with Cloud ML and Tensorflow 1.4
        # Outer dimension needs to be [?, ...]
        # https://github.com/tensorflow/models/issues/1811
        # logits = tf.reshape(net.logits, [-1, len(labels)])

        # Create saver to restore from checkpoints
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Restore the model from last checkpoint
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

            # Create tensors info
            predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(
                jpegs)
            predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(
                values)
            predict_tensor_classes_info = tf.saved_model.utils.build_tensor_info(
                prediction_classes)

            # Build prediction signature
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': predict_tensor_inputs_info},
                    outputs={
                        'scores': predict_tensor_scores_info,
                        'classes': predict_tensor_classes_info
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

            # Save the model
            legacy_init_op = tf.group(tf.tables_initializer(),
                                      name="legacy_init_op")
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_labels': prediction_signature
                },
                legacy_init_op=legacy_init_op)

            builder.save()

    print("Successfully exported CNN model version'{}' into '{}'".format(
        FLAGS.model_version, FLAGS.output_dir))


if __name__ == '__main__':
    tf.app.run()
