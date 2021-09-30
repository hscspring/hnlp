from tensorflow.keras.losses import kullback_leibler_divergence as kld
import tensorflow.keras.backend as K


def rdrop_loss(y_true, y_pred):
    loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return K.mean(loss)
