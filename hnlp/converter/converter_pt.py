from functools import wraps, partial
from typing import Tuple

from torch import Tensor
import torch
from hnlp.config import ModelInputType, device, ModelLabelType


def convert_input(inputs: ModelInputType):
    # Tensor or List is treated as input_ids
    if isinstance(inputs, list) == True:
        inp = {"input_ids": torch.tensor(inputs).to(device)}
    elif isinstance(inputs, Tensor):
        inp = {"input_ids": inputs.to(device)}
    # Tensors in Tuple should be sorted as follows:
    # input_ids, attention_mask, token_type_ids, position_ids
    elif isinstance(inputs, Tuple):
        keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]
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
        if inputs is None and len(args) == 0:
            raise ValueError("hnlp: Invalid inputs.")
        if inputs is None and len(args) > 0:
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
