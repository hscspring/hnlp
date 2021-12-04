from typing import Dict, Any
from addict import Dict as ADict
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K

from hnlp.module_tf import cnn, gru


class Model:

    name: str

    def __post_init__(self):
        pass


def build_model(module, config, inputs):
    inputs = tfk.layers.Input(shape=(config.max_seq_len, ))
    mask = tfk.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
    embed = tfk.layers.Embedding(config.vocab_size, config.embed_size)(inputs)
    z = tfk.layers.Dropout(config.dropout)(embed)
    z = module(config, z)
    out = K.l2_normalize(z, axis=1)
    model = tfk.Model(inputs=[inp], outputs=[out])
    return model