from dataclasses import dataclass
from typing import Dict, Union, Optional
from collections.abc import Iterable
from hnlp.register import Register
from hnlp.node import Node
from hnlp.config import check_name

from hnlp.pretrained.fasttext_adapter import FasttextPretrainedModel
from hnlp.pretrained.transformers_adapter import BertPretrainedModel


@dataclass
class Pretrained(Node):

    """
    Middleware of Pretrained models.


    Parameters
    -----------
    name: pretrained model name
    model_path: pretrained model location
    model_config: pretrained model config
    training_type: should be "scratch", "continuous", "predict"
    """

    name: Optional[str]
    model_path: Optional[str] = None
    model_config: Optional[Dict[str, Union[str, int, float]]] = None
    training_type: Optional[str] = None

    def __post_init__(self):
        super().__init__()
        self.identity = "pretrained_model"
        check_name(self.identity, self.name)
        cls_name = "_".join([self.name, self.identity])
        cls = Register.get(cls_name)
        if not cls:
            info = "hnlp: invalid pretrained name: {}".format(self.name)
            raise NotImplementedError(info)
        self.node = cls(
            self.name, self.model_path, self.model_config, self.training_type
        )

    def fit(self, iter_corpus: Iterable):
        return self.node.fit(iter_corpus)
