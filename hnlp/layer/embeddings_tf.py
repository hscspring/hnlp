import tensorflow.keras as tfk
import tensorflow as tf
from pnlp import MagicDict


class Embeddings(tfk.layers.Layer):
    """
    Modified from transformers

    """

    def __init__(self, config: MagicDict, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        dc = self.default_config

        self.vocab_size = config.vocab_size or dc.vocab_size
        self.embed_size = config.embed_size or dc.embed_size
        self.max_seq_len = config.max_seq_len or dc.max_seq_len
        self.token_type_size = config.token_type_size or dc.token_type_size
        self.initializer_range = config.initializer_range or dc.initializer_range

        self.sum = tfk.layers.Add()
        self.ln = tfk.layers.LayerNormalization(
            epsilon=config.layer_norm_eps or dc.layer_norm_eps,
        )
        self.dropout = tfk.layers.Dropout(rate=config.embed_dropout or dc.embed_dropout)

    def get_config(self):
        return {**self.default_config, **self.config}

    @property
    def default_config(self):
        return MagicDict(
            {
                "vocab_size": 8021,
                "embed_size": 256,
                "max_seq_len": 512,
                "token_type_size": 1,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "embed_dropout": 0.1,
            }
        )

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embedings"):
            self.word_embeddings = self.add_weight(
                shape=[self.vocab_size, self.embed_size],
                initializer=tfk.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                shape=[self.max_seq_len, self.embed_size],
                initializer=tfk.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                shape=[self.token_type_size, self.embed_size],
                initializer=tfk.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        input_shape = input_ids.shape.as_list()
        word_embeds = tf.gather(params=self.word_embeddings, indices=input_ids)
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=0, limit=input_shape[1]), axis=0
            )
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        position_embeds = tf.gather(
            params=self.position_embeddings, indices=position_ids
        )
        position_embeds = tf.tile(
            input=position_embeds, multiples=(input_shape[0], 1, 1)
        )
        token_type_embeds = tf.gather(
            params=self.token_type_embeddings, indices=token_type_ids
        )

        embeds = self.sum(inputs=[word_embeds, position_embeds, token_type_embeds])
        embeds = self.ln(inputs=embeds)
        embeds = self.dropout(inputs=embeds, training=training)

        return embeds
