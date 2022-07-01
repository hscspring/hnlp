import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class Model(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.num_hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.label_size = config.num_labels
        self.embedding_size = config.embed_size
        self.dropout_rate = config.dropout

        self.embedding = tfk.layers.Embedding(
            self.vocab_size, self.embedding_size
        )
        self.biLSTM = tfk.layers.Bidirectional(
            tfk.layers.LSTM(self.num_hidden, return_sequences=True)
        )
        self.dense = tfk.layers.Dense(self.label_size, activation="softmax")

        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(self.label_size, self.label_size))
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)

    @tf.function
    def call(self, x, y=None, training=None):
        text_lens = K.sum(tf.cast(K.not_equal(x, 0), dtype=tf.int32), axis=-1)

        z = self.embedding(x)
        z = self.dropout(z, training)
        z = self.biLSTM(z)
        logits = self.dense(z)

        if y is None:
            return logits, text_lens
        log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(
            logits,
            y,
            text_lens,
            transition_params=self.transition_params
        )
        return logits, text_lens, log_likelihood


def loss_fn(output, y_true):
    _, _, log_likelihood = output
    return - tf.reduce_mean(log_likelihood)


def metric_step(model, output, batch_labels):
    y_preds = []
    y_trues = []
    for logit, text_len, labels in zip(output[0], output[1], batch_labels):
        viterbi_path, _ = tfa.text.viterbi_decode(
            logit[:text_len], model.transition_params)
        text_len -= 2
        ps = viterbi_path[1:-1]
        ts = labels[:text_len].numpy().tolist()[:text_len]
        assert len(ps) == len(
            ts), f"text_len {text_len}, length predict {len(ps)}, length true {len(ts)}"
        y_preds.append(ps)
        y_trues.append(ts)
    return y_preds, y_trues
