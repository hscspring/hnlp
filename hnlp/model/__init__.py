from hnlp.config import ARCH
if ARCH == "tf":
    from hnlp.model.model_tf import pretrained, text_cnn, gru
else:
    raise NotImplemented
