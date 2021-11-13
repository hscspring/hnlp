from pnlp import MagicDict

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as tfk


class InteractiveSelfAttention(tfk.layers.Layer):
    """
    Context Attention.

    Note
    --------
    There are actually three types of Attention in <Effective Approaches to Attention-based Neural Machine Translation> (Luong Attention):
    - dot
    - general
    - concat
    Here we do not use concat as there is only one input tensor.
    Additionally, we do not add `sum, activation and dropout` to tht output tensor.


    References
    ------------------
    - Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision, 2018/04, http://arxiv.org/abs/1805.12307
    - Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification, 2016/08, https://aclanthology.org/P16-2034
    """

    def __init__(self, config: MagicDict = MagicDict(), **kwargs):
        super().__init__(**kwargs)

        self.config = config
        dc = self.default_config

        self.initializer_range = config.initializer_range or dc.initializer_range
        self.attention_type = config.attention_type or dc.attention_type

    def get_config(self):
        return {**self.default_config, **self.config}

    @property
    def default_config(self):
        return MagicDict({
            "initializer_range": 0.02,
            "attention_type": "dot",  # should be one of dot, general
        })

    def build(self, input_shape: tf.TensorShape):
        size = input_shape[-1]
        # self.W = tfk.layers.Dense(size)
        # self.U = tfk.layers.Desne(1)
        if self.attention_type == "general":
            self.W = self.add_weight(
                name="W",
                shape=[size, size],
                initializer=tfk.initializers.TruncatedNormal(
                    stddev=self.initializer_range),
            )
        self.U = self.add_weight(
            name="U",
            shape=[size],
            initializer=tfk.initializers.TruncatedNormal(
                stddev=self.initializer_range),
        )
        self.b = self.add_weight(
            name="b",
            shape=[size],
            initializer=tfk.initializers.constant(0),
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, mask: tf.Tensor = None):
        if self.attention_type == "general":
            a = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        else:
            a = x
        a = tf.tensordot(a, self.U, axes=1)
        alpha = tf.nn.softmax(a)
        v = x * K.expand_dims(alpha, -1)
        return v
