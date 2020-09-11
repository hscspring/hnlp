from dataclasses import dataclass
import importlib
from functools import wraps, partial
from typing import List, Tuple, Dict, TypeVar
import os

from torch import Tensor
import torch
import torch.nn as nn

from hnlp.utils import check_file

from pathlib import Path

from addict import Dict as ADict
import pnlp


class TaskConfig(ADict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(TaskConfig, self).__init__(*args, **kwargs)
        # self.__dict__ = self

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


root = Path(os.path.abspath(__file__)).parent
task_config = TaskConfig(pnlp.read_json(root / "task/config.json"))

transformers = importlib.import_module("transformers")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModelInputType = TypeVar("ModelInputType",
                         Tensor,
                         List[List[int]],
                         Tuple[Tensor],
                         Dict[str, Tensor])
ModelLabelType = TypeVar("ModelLabelType",
                         List[str],
                         List[int],
                         Tensor)


def convert_input(inputs: ModelInputType):
    # Tensor or List is treated as input_ids
    if isinstance(inputs, list) == True:
        inp = {
            "input_ids": torch.tensor(inputs).to(device)
        }
    elif isinstance(inputs, Tensor):
        inp = {
            "input_ids": inputs.to(device)
        }
    # Tensors in Tuple should be sorted as follows:
    # input_ids, attention_mask, token_type_ids, position_ids
    elif isinstance(inputs, Tuple):
        keys = ["input_ids", "attention_mask",
                "token_type_ids", "position_ids"]
        inputs = [v.to(device) for v in inputs]
        inp = dict(zip(keys, inputs))
    else:
        assert type(inputs) == dict
        inp = {k: v.to(device) for k, v in inputs.items()}
    return inp


def convert_label(labels: ModelLabelType):
    if isinstance(labels, Tensor):
        return labels
    labels = list(map(int, labels))
    return torch.tensor(labels)


def convert_model_input(func=None, *, target: str = "pretrained"):

    if func is None:
        return partial(convert_model_input, target=target)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        inputs = kwargs.get("inputs")
        labels = kwargs.get("labels")
        print(args)
        print(kwargs)
        if inputs == None and len(args) == 0:
            raise ValueError("hnlp: Invalid inputs.")
        if inputs == None and len(args) > 0:
            inputs = args[0]
        if not labels and len(args) > 1:
            labels = args[1]

        if target == "pretrained":
            inp = convert_input(inputs)
        else:
            inp = inputs

        if isinstance(labels, Tensor) or labels:
            lab = convert_label(labels)
            return func(self, inp, lab)
        else:
            return func(self, inp)
    return wrapper
