from typing import Dict, Any
from addict import Dict as ADict
import tensorflow as tf
import tensorflow.keras as tfk

from hnlp.layer.pretrained_tf import PretrainedBert, PretrainedWord2vec


def pretrained(
    config: Dict[str, Any], inputs, mask=None, token_type_ids=None, training=False
):
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
    elif config.use_pretrained_word2vec:
        embed = PretrainedWord2vec(
            config.pretrained_word2vec_path, config.fix_pretrained
        )(inputs)
        return embed
    else:
        raise NotImplementedError


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
    embed = tf.expand_dims(embed, axis=-1)
    convs = []
    for size in map(int, config.filter_sizes.split(",")):
        conv = tfk.layers.Conv2D(
            config.num_filters, kernel_size=size, strides=(1, config.embed_size), padding="valid"
        )(embed)
        pool = tfk.layers.MaxPool2D(
            pool_size=(config.max_seq_len - size + 1, 1), strides=(1, 1), padding="valid"
        )(conv)
        convs.append(pool)
    concat = tfk.layers.Concatenate(axis=-1)(convs)
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
