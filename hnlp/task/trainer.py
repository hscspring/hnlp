from dataclasses import dataclass
from argparse import Namespace

import torch
import torch.nn as nn

from hnlp.config import device
from hnlp.utils import get_attr


@dataclass
class Trainer:

    args: Namespace
    model: nn.Module

    def __post_init__(self):
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=0.9)
        self.n_epochs = get_attr(self.args, "n_epochs", 1)
