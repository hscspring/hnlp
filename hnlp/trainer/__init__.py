from hnlp.config import ARCH


if ARCH == "tf":
    from hnlp.trainer.loss_tf import rdrop_loss
else:
    raise NotImplementedError
