from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class FC(nn.Module):

    hidden_size: int
    hidden_dropout_prob: float
    num_labels: int

    def __post_init__(self):
        super().__init__()
        self.fc = nn.Linear(self.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, inputs: Tensor):
        output = self.fc(inputs)
        output = self.dropout(output)
        return output
