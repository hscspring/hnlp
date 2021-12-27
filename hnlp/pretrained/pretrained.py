From typing import Dict, Union, Optional
from collections.abc import Iterable
from hnlp.node import Node


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

    def __init__(self,
                 name: Optional[str],
                 model_path: Optional[str] = None,
                 model_config: Optional[Dict[str, Union[str, int,
                                                        float]]] = None,
                 training_type: Optional[str] = None):

        self.name = name
        self.model_path = model_path
        self.model_config = model_config
        self.training_type = training_type
        self.identity = "pretrained_model"
        self.node = super().get_cls(self.identity,
                                    self.name)(self.name, self.model_path,
                                               self.model_config,
                                               self.training_type)

    def fit(self, iter_corpus: Iterable):
        return self.node.fit(iter_corpus)
