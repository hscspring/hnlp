"""
Pretrained layer

"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from transformers import TFAutoModel, AutoConfig


class PretrainedWord2vec(tfk.layers.Layer):

    def __init__(self, pretrained_path: str, fix_pretrained: bool):
        super(PretrainedWord2vec, self).__init__()
        trainable = not fix_pretrained
        pretrained = np.load(pretrained_path)
        embed_array = pretrained.astype("float32")
        embed_tensor = tf.constant(embed_array)
        vocab_size = embed_tensor.shape[0]
        embed_size = embed_tensor.shape[1]
        self.embed = tfk.layers.Embedding(
            vocab_size, embed_size,
            embeddings_initializer=tf.keras.initializers.Constant(embed_tensor),
            trainable=trainable
        )

    def call(self, inputs):
        return self.embed(inputs)


class PretrainedBert(tfk.layers.Layer):

    """
    Note
    -------
    trainable to control the whole net or some layer
    training to control the net is running with the infer mode
    """

    def __init__(self, pretrained_path: str, fix_pretrained: bool):
        super(PretrainedBert, self).__init__()
        config_path = os.path.join(str(pretrained_path), "bert_config.json")
        config = AutoConfig.from_pretrained(config_path)
        trainable = not fix_pretrained
        self.bert = TFAutoModel.from_config(config)
        self.bert.trainable = trainable

    def call(self, inputs, mask, training):
        return self.bert(**{"input_ids": inputs, "attention_mask": mask}, training=training)
