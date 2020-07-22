from dataclasses import dataclass
import os
from transformers import BertModel, BertConfig


from hnlp.node import Node
from hnlp.register import Register
from hnlp.base import PretrainedBaseModel, device
from hnlp.utils import check_file, check_dir, build_pretrained_config_from_json


@dataclass
class Pretrained(Node):

    name: str
    model_path: str
    is_training: bool = False
    output_hidden_states: bool = False
    output_attentions: bool = False

    def __post_init__(self):
        super().__init__()
        self.identity = "pretrained"
        self.batch = True
        check_dir(self.model_path)
        PretrainedModel = Register.get(self.name)
        if not PretrainedModel:
            raise NotImplementedError
        self.node = PretrainedModel(
            self.model_path,
            self.is_training,
            self.output_hidden_states,
            self.output_attentions)


@Register.register
@dataclass
class Bert(PretrainedBaseModel):

    model_path: str
    is_training: bool
    output_hidden_states: bool
    output_attentions: bool

    def __post_init__(self):
        config_path = os.path.join(self.model_path, "config.json")
        check_file(config_path)
        config = BertConfig.from_json_file(config_path)
        config.output_hidden_states = self.output_hidden_states
        config.output_attentions = self.output_attentions
        if self.is_training:
            self.model = BertModel.from_pretrained(
                self.model_path, config=config).to(device)
        else:
            self.model = BertModel(config).to(device)
