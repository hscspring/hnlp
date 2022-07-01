import tensorflow as tf
from torch import initial_seed
import tensorflow.keras as tfk
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class TokenClassificationSoftmaxModel(tfk.Model):
    def __init__(self, config):
        super().__init__()
        self.num_hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.label_size = config.num_labels
        self.embedding_size = config.embed_size
        self.dropout_rate = config.dropout

        self.embedding = tfk.layers.Embedding(
            self.vocab_size,
            self.embedding_size,
            mask_zero=True
        )
        self.bi_lstm = tfk.layers.Bidirectional(
            tfk.layers.LSTM(self.num_hidden, return_sequences=True)
        )
        self.dense = tfk.layers.Dense(
            units=self.label_size,
            activation="softmax",
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)

    @tf.function
    def call(self, x, y=None, training=None):
        z = self.embedding(x)
        z = self.dropout(z, training)
        z = self.bi_lstm(z)
        # z = self.dropout(z, training)
        z = self.dense(z)
        # output = K.l2_normalize(z, axis=1)
        return z


class TokenClassificationCrfModel(tf.keras.Model):
    """
    From: https://github.com/saiwaiyanyu/bi-lstm-crf-ner-tf2.0
    """

    def __init__(self, config):
        super().__init__()
        self.num_hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.label_size = config.num_labels
        self.embedding_size = config.embed_size
        self.dropout_rate = config.dropout

        self.embedding = tfk.layers.Embedding(
            self.vocab_size,
            self.embedding_size,
            mask_zero=True
        )
        self.bi_lstm = tfk.layers.Bidirectional(
            tfk.layers.LSTM(self.num_hidden, return_sequences=True)
        )
        self.dense = tfk.layers.Dense(
            self.label_size,
            activation="softmax"
        )

        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(self.label_size, self.label_size)),
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)

    def call(self, x, y=None, training=None):
        text_lens = K.sum(tf.cast(K.not_equal(x, 0), dtype=tf.int32), axis=-1)

        z = self.embedding(x)
        z = self.dropout(z, training)
        z = self.bi_lstm(z)
        logits = self.dense(z)

        if y is None:
            return (logits, )
        log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(
            logits,
            y,
            text_lens,
            transition_params=self.transition_params
        )
        return logits, log_likelihood
