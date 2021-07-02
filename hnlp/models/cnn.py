from typing import Dict
import tensorflow.keras as tfk
from hnlp.utils import check_parameter


@check_parameter
def cnn(config: Dict):
    mdc = config.model
    vocab_size = mdc.vocab_size + mdc.num_oov_buckets
    inputs = tfk.layers.Input(shape=[None])
    mask = tfk.layers.Lambda(
        lambda inputs: tfk.backend.not_equal(inputs, 0))(inputs)
    z = tfk.layers.Embedding(vocab_size, mdc.embed_size)(inputs)
    z = tfk.layers.GRU(mdc.hidden_size, return_sequences=True)(z, mask=mask)
    z = tfk.layers.GRU(mdc.hidden_size)(z, mask=mask)
    outputs = tfk.layers.Dense(mdc.num_labels, activation="softmax")(z)
    model = tfk.Model(inputs=[inputs], outputs=[outputs])
    return model
