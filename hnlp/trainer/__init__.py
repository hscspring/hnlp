from hnlp.config import ARCH


if ARCH == "tf":
    from hnlp.trainer.loss_tf import Loss
    from hnlp.trainer.metric_tf import MetricStep
    from hnlp.trainer.trainer_tf import Trainer
else:
    raise NotImplementedError
