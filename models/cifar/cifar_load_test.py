import tensorflow as tf
import pickle


def main(_):

    # Get validation data
    valid_features, valid_labels = pickle.load(
        open('data/preprocess_validation.p', mode='rb'))

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Restore the saved model
        saver = tf.train.import_meta_graph('./checkpoints/cifar.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

        # Get tensors by name
        x_tensor = loaded_graph.get_tensor_by_name("x:0")
        y_tensor = loaded_graph.get_tensor_by_name("y:0")
        keep_prob_tensor = loaded_graph.get_tensor_by_name("keep_prob:0")
        pred_class_tensor = loaded_graph.get_tensor_by_name("pred_class:0")
        accuracy_tensor = loaded_graph.get_tensor_by_name("accuracy:0")

        # Make predictions
        accuracy, pred_class = sess.run(
            [accuracy_tensor, pred_class_tensor],
            feed_dict={
                x_tensor: valid_features,
                y_tensor: valid_labels,
                keep_prob_tensor: 1.})

        # Print
        print("Prediction accuracy: {:.2f}%".format(accuracy * 100))
        print("Predicted classes: {}".format(pred_class))


if __name__ == '__main__':
    tf.app.run()
