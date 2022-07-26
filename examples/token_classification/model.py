import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from transformers import TFAutoModel, AutoConfig


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


class LstmSoftmaxModel(tfk.Model):
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
        self.lstm = tfk.layers.LSTM(self.num_hidden, return_sequences=True)

        self.dense = tfk.layers.Dense(
            units=self.label_size,
            activation="softmax",
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)

    @tf.function
    def call(self, x, y=None, training=None):
        z = self.embedding(x)
        z = self.dropout(z, training)
        z = self.lstm(z)
        z = self.dense(z)
        return z


class LstmCrfModel(tf.keras.Model):
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
            mask_zero=False
        )
        self.lstm = tfk.layers.LSTM(self.num_hidden, return_sequences=True)

        self.dense = tfk.layers.Dense(
            self.label_size,
            activation="softmax"
        )

        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(self.label_size, self.label_size)),
            name="crf",
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)

    def call(self, x, y=None, training=None):
        text_lens = K.sum(tf.cast(K.not_equal(x, 0), dtype=tf.int32), axis=-1)

        z = self.embedding(x)
        z = self.dropout(z, training)
        z = self.lstm(z)
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

    def get_learning_rate(self):
        dct = {
            "crf": 2.0,
        }
        return dct


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
            mask_zero=False
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
            name="crf",
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

    def get_learning_rate(self):
        dct = {
            "crf": 2.0,
        }
        return dct


class BertSoftmaxModel(tfk.Model):
    def __init__(self, config):
        super().__init__()
        self.label_size = config.num_labels
        self.dropout_rate = config.dropout

        bert_root = config.bert_root
        bert_config = AutoConfig.from_pretrained(bert_root / "config.json")
        self.bert = TFAutoModel.from_config(bert_config)
        self.dense = tfk.layers.Dense(
            units=self.label_size,
            activation="softmax"
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)

    @tf.function
    def call(self, x, y=None, training=None):
        z = self.bert(x)
        hs = z.last_hidden_state
        z = self.dropout(hs, training)
        output = self.dense(z)
        return output


class BertCrfModel(tf.keras.Model):
    """
    From: https://github.com/saiwaiyanyu/bi-lstm-crf-ner-tf2.0
    """

    def __init__(self, config):
        super().__init__()

        self.label_size = config.num_labels
        self.dropout_rate = config.dropout

        bert_root = config.bert_root
        bert_config = AutoConfig.from_pretrained(bert_root / "config.json")
        self.bert = TFAutoModel.from_config(bert_config)
        self.dense = tfk.layers.Dense(
            units=self.label_size,
            activation="softmax"
        )
        self.dropout = tfk.layers.Dropout(self.dropout_rate)

        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(self.label_size, self.label_size)),
            name="crf",
        )

    def call(self, x, y=None, training=None):
        text_lens = K.sum(tf.cast(K.not_equal(x, 0), dtype=tf.int32), axis=-1)

        z = self.bert(x)
        hs = z.last_hidden_state
        z = self.dropout(hs, training)
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

    def get_learning_rate(self):
        dct = {
            "crf1": 2.0,
        }
        return dct
