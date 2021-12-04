from dataclasses import dataclass
import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow.keras.backend as K


from hnlp.node import Node


@dataclass 
class Task(Node):

    name: str
    optimizer: str
    loss_fn: str
    metric: str

    def __post_init__(self):
        self.identity = "task"
        self.node = ""



def cosine_task(model, loss_fn, inp1, inp2, y_true, training: bool):
    z1 = model(inp1, training=training)
    z2 = model(inp2, training=training)
    sim = -tfk.losses.cosine_similarity(z1, z2)
    logits = tf.stack([1- sim, sim], axis=1)
    probs = K.softmax(logits)
    loss = loss_fn(y_true, probs)
    return loss, probs


def cls_task():
    pass