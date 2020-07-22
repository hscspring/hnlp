import torch
import torch.nn as nn
from torch import Tensor

from hnlp.base import ModelInputType
from hnlp.pretrained.model import Bert
from hnlp.register import Register


@Register.register
class BertFcClassifier(nn.Module):

    """
    Bert Based FC Classifier.
    Using the first token [CLS] to classify.
    """

    def __init__(self,
                 pretrained_path: str,
                 is_training: bool,
                 output_hidden_states: bool = False,
                 output_attentions: bool = False):
        super().__init__()
        self.is_training = is_training
        self.pretrain = Bert(pretrained_path, is_training,
                             output_hidden_states, output_attentions)
        self.bert = self.pretrain.model
        self.num_labels = self.bert.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size,
                                    self.num_labels)
        self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_(
            mean=0.0, std=self.bert.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def __forward(self, inp: Tensor):
        inp = self.dropout(inp)
        logits = self.classifier(inp)
        return logits

    def forward(self, inp: ModelInputType):
        outputs = self.pretrain(inp)
        pooled_output = outputs[1]
        if not self.is_training:
            with torch.no_grad():
                logits = self.__forward(pooled_output)
        else:
            logits = self.__forward(pooled_output)
        return logits
