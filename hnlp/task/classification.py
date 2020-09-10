import torch
import torch.nn as nn
from torch import Tensor

from hnlp.base import task_config
from hnlp.register import Register

@Register.register
class FcClassifier(nn.Module):

    """
    FC Classifier.
    Using the first token [CLS] to classify.
    """

    def __init__(self, is_training: bool):
        super().__init__()
        self.is_training = is_training
        self.config = task_config.classifier.fc
        print(self.config)
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size,
                                    self.num_labels)
        self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_(
            mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def __forward(self, inp: Tensor):
        inp = self.dropout(inp)
        logits = self.classifier(inp)
        return logits

    def forward(self, pretrained_output):
        pooled_output = pretrained_output[1]
        if not self.is_training:
            with torch.no_grad():
                logits = self.__forward(pooled_output)
        else:
            logits = self.__forward(pooled_output)
        return logits
