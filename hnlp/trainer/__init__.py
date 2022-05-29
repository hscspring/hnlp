from hnlp.config import ARCH


if ARCH == "tf":
    from hnlp.trainer.loss_tf import rdrop_loss
    from hnlp.trainer.trainer_tf import Trainer
else:
    raise NotImplementedError
