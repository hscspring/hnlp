import tensorflow as tf


tf.random.set_seed(42)


def gen_input(
    inp_num: int = 1, label_num: int = 2, batch_size: int = 3, seq_len: int = 20
):
    res = []
    for i in range(inp_num):
        inp = tf.random.uniform(
            shape=(batch_size, seq_len), minval=0, maxval=1000, dtype=tf.int32
        )
        res.append(inp)
    labels = tf.cast(
        tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=label_num, dtype=tf.int32
        ),
        dtype=tf.float16,
    )
    res.append(labels)
    return tuple(res)


def gen_hidden(batch_size: int = 3, seq_len: int = 20, hidden_size: int = 256):
    return tf.random.uniform(
        shape=(batch_size, seq_len, hidden_size), minval=0, maxval=1, dtype=tf.float32
    )
