import sys
import tensorflow as tf

from grpc.beta import implementations
from pathlib import Path
from tvscript_helper import token_lookup

sys.path.append(str(Path('.').absolute().parent))

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server',
                           'localhost:9000',
                           """PredictionService host:port""")
tf.app.flags.DEFINE_string('text',
                           'homer_simpson:',
                           """Prime text to use to generate script""")
tf.app.flags.DEFINE_integer('seq_length',
                            100,
                            """Number of words to generate""")
FLAGS = tf.app.flags.FLAGS


def __prepare_script_results__(text):
    token_dict = token_lookup()

    for key, token in token_dict.items():
        text = text.replace(' ' + token.lower(), key)
    text = text.replace('\n ', '\n')
    text = text.replace('( ', '(')

    return text


def main(_):
    host, port = FLAGS.server.split(":")
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request
    request = predict_pb2.PredictRequest()

    # Call RNN model to make predictions on text
    request.model_spec.name = 'tvscript'
    request.model_spec.signature_name = 'predict_labels'

    prime_text = FLAGS.text

    for i in range(FLAGS.seq_length):
        request.inputs['text'].CopyFrom(
            tf.contrib.util.make_tensor_proto(prime_text, shape=[1]))
        result = stub.Predict(request, 20.0)
        next_word = str(result.outputs['labels'].string_val.pop(), 'utf-8')
        prime_text = ' '.join([prime_text, next_word])

    print(__prepare_script_results__(prime_text))


if __name__ == '__main__':
    tf.app.run()
