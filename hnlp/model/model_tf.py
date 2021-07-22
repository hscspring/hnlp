from addict import Dict as ADict
import tensorflow as tf
import tensorflow.keras as tfk

from hnlp.layer.pretrained_tf import PretrainedBert, PretrainedWord2vec


def pretrained(config, inputs, mask=None):
    config = ADict(config)
    if config.use_pretrained_bert:
        layer = PretrainedBert(config.pretrained_bert_path, config.fix_pretrained)(
            inputs, mask
        )
        if config.use_cls:
            embed = layer[1]
        else:
            embed = layer[0]
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


def text_cnn(config, embed):
    """

    Parameters
    -----------
    config:
    embed: None, seq_len, embed_size
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


def gru(config, embed, mask):
    config = ADict(config)
    z = tfk.layers.Dropout(config.dropout)(embed)
    z = tfk.layers.GRU(config.hidden_size, return_sequences=True)(z, mask=mask)
    z = tfk.layers.GRU(config.hidden_size)(z)
    return z
