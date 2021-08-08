from typing import Dict, Any
from addict import Dict as ADict
import tensorflow as tf
import tensorflow.keras as tfk

from hnlp.layer.pretrained_tf import PretrainedBert, PretrainedWord2vec


def pretrained(config: Dict[str, Any], inputs, mask=None, training=False):
    """

    Parameters
    -----------
    config: Config dict
    inputs: (None, seq_len)
    mask: (None, seq_len)
    """
    config = ADict(config)
    if config.use_pretrained_bert:
        layer = PretrainedBert(config.pretrained_bert_path, config.fix_pretrained)(
            inputs, mask, training
        )
        return layer
    else:
        if config.use_pretrained_word2vec:
            embed = PretrainedWord2vec(
                config.pretrained_word2vec_path, config.fix_pretrained
            )(inputs)
        else:
            embed = tfk.layers.Embedding(
                config.vocab_size + config.num_oov_buckets, config.embed_size
            )(inputs)
    return embed


def text_cnn(config: Dict[str, Any], embed):
    """

    Parameters
    -----------
    config: Config dict
    embed: (None, seq_len, embed_size)

    Note
    ---------
    Dropout has been added for output concat
    """
    config = ADict(config)
    embed = embed[:, :, :, None]
    conv2 = tfk.layers.Conv2D(
        config.filters, kernel_size=2, strides=(1, config.embed_size), padding="valid"
    )(embed)
    pool2 = tfk.layers.MaxPool2D(
        pool_size=(config.max_seq_len - 2 + 1, 1), strides=(1, 1), padding="valid"
    )(conv2)

    conv3 = tfk.layers.Conv2D(
        config.filters, kernel_size=3, strides=(1, config.embed_size), padding="valid"
    )(embed)
    pool3 = tfk.layers.MaxPool2D(
        pool_size=(config.max_seq_len - 3 + 1, 1), strides=(1, 1), padding="valid"
    )(conv3)

    conv4 = tfk.layers.Conv2D(
        config.filters, kernel_size=4, strides=(1, config.embed_size), padding="valid"
    )(embed)
    pool4 = tfk.layers.MaxPool2D(
        pool_size=(config.max_seq_len - 4 + 1, 1), strides=(1, 1), padding="valid"
    )(conv4)

    concat = tfk.layers.Concatenate(axis=-1)([pool2, pool3, pool4])
    z = tf.squeeze(concat, [1, 2])
    z = tfk.layers.Dropout(config.dropout)(z)
    return z


def text_gru(config: Dict[str, Any], embed, mask):
    """

    Parameters
    ----------
    config: Config dict
    embed: (None, seq_len, embed_size)
    mask: (None, seq_len)

    """
    config = ADict(config)
    z = tfk.layers.GRU(config.hidden_size, return_sequences=True)(embed, mask=mask)
    z = tfk.layers.GRU(config.hidden_size)(z)
    return z
