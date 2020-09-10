from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch.utils.data.dataloader import DataLoader

from hnlp.node import Node


@dataclass
class PretrainedProcessor(Node):

    name: str = "pretrained"

    def __post_init__(self):
        super().__init__()
        self.identity = "processor"
        if self.name == "pretrained":
            self.node = PretrainedProcessor()
        else:
            raise NotImplementedError


class PretrainedProcessor:

    def process_batch(self, batch: List[List[int]]):
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

    def __call__(self, inputs: List[List[int]] or DataLoader):
        if isinstance(inputs, DataLoader):
            for batch in inputs:
                yield self.process_batch(batch)
        else:
            yield self.process_batch(inputs)
