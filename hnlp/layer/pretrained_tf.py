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
        self.vocab_size = embed_array.shape[0]
        self.embed_size = embed_array.shape[1]
        self.embed = tfk.layers.Embedding(
            self.vocab_size,
            self.embed_size,
            embeddings_initializer=tfk.initializers.Constant(embed_array),
            trainable=trainable,
        )

    def get_config(self):
        config = super(PretrainedWord2vec, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_size": self.embed_size
        })
        return config

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
        self.config = AutoConfig.from_pretrained(config_path)
        trainable = not fix_pretrained
        self.bert = TFAutoModel.from_config(self.config)
        self.bert.trainable = trainable

    def get_config(self):
        config = super(PretrainedBert, self).get_config()
        config.update(self.config)
        return config

    def call(self, inputs, mask, token_type_ids, training):
        return self.bert(
            **{
                "input_ids": inputs,
                "attention_mask": mask,
                "token_type_ids": token_type_ids,
            },
            training=training
        )
