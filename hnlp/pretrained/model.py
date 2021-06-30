from dataclasses import dataclass
from typing import Dict, Union, Optional
from hnlp.node import Node


from hnlp.utils import check_dir
from hnlp.pretrained.transformers_adapter import TransformersModel
from hnlp.pretrained.fasttext_adapter import FastTextModel


@dataclass
class Pretrained(Node):

    """
    Middleware of Pretrained models.


    Parameters
    -----------
    type: pretrained model type, could be "fasttext" or "transformers"
    name: pretrained model name
    model_path: pretrained model location
    model_config: pretrained model config
    training_type: should be "scratch", "continuous", "predict"
    """

    type: str
    name: Optional[str]
    model_path: Optional[str]
    model_config: Optional[Dict[str, Union[str, int, float]]]
    training_type: Optional[str]

    def __post_init__(self):
        super().__init__()
        self.identity = "pretrained"

        check_dir(self.model_path)

        if self.type == "fasttext":
            cls = FastTextModel
        elif self.type == "transformers":
            cls = TransformersModel
        else:
            info = "hnlp: {} is not a valid type now".format(self.type)
            raise NotImplementedError(info)

        self.node = cls(
            self.name, self.model_path, self.model_config, self.training_type
        )
