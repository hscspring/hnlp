from dataclasses import dataclass
from typing import List

import torch

from hnlp.node import Node


@dataclass
class PretrainedProcessor(Node):

    name: str

    def __post_init__(self):
        super().__init__()
        self.identity = "pretrained_processor"
        self.batch = True
        if self.name == "bert":
            self.node = BertProcessor()
        else:
            raise NotImplementedError


class BertProcessor:

    def __call__(self, batch: List[List[int]]):
        """
        Referenced from transformers.
        """
        input_ids = torch.tensor(batch)
        device = input_ids.device
        input_shape = input_ids.size()
        attention_mask = torch.ones(input_shape, device=device)
        token_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=device)
        position_ids = torch.arange(
            input_shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids
        }
