from tensorflow.keras.losses import kullback_leibler_divergence as kld
import tensorflow.keras.backend as K


class Loss:

    @staticmethod
    def rdrop_loss(y_pred, y_true):
        loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
        return K.mean(loss)

    @staticmethod
    def crossentropy_loss(output, y_true):
        loss = K.sparse_categorical_crossentropy(
            y_true, output, from_logits=False)
        return K.mean(loss)

    @staticmethod
    def mean_loss_crf(output, y_true):
        loss = output[-1]
        return - K.mean(loss)
