from hnlp.config import ARCH

if ARCH == "tf":
    from hnlp.module.module_tf import pretrained, cnn, gru
else:
    raise NotImplementedError
