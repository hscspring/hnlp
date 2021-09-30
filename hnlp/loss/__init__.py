from hnlp.config import ARCH


if ARCH == "tf":
    from hnlp.loss.loss_tf import rdrop_loss
else:
    raise NotImplementedError
