from hnlp.config import ARCH

if ARCH == "tf":
    from hnlp.model.model_tf import pretrained, cnn, gru
else:
    raise NotImplementedError
