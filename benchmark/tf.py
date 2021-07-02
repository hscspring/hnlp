from typing import Dict
from addict import Dict as ADict
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as tfk


def build_model(config: Dict):

    inputs = tfk.layers.Input(shape=[None])
    mask = tfk.layers.Lambda(
        lambda inputs: tfk.backend.not_equal(inputs, 0))(inputs)
    z = tfk.layers.Embedding(
        config.vocab_size + config.num_oov_buckets, config.embed_size
    )(inputs)
    z = tfk.layers.LSTM(config.hidden_size,
                        return_sequences=True)(z, mask=mask)
    z = tfa.MultiHeadAttention()
