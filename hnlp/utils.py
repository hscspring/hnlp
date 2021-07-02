from collections import namedtuple
from functools import wraps, partial
from addict import Dict as ADict
from pathlib import Path
from typing import Any, Tuple
import re
import pnlp

from torch import Tensor
import torch
from hnlp.config import ModelInputType, device, ModelLabelType


name_split_reg = re.compile(r"[-_]")


def convert_input(inputs: ModelInputType):
    # Tensor or List is treated as input_ids
    if isinstance(inputs, list) == True:
        inp = {"input_ids": torch.tensor(inputs).to(device)}
    elif isinstance(inputs, Tensor):
        inp = {"input_ids": inputs.to(device)}
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


def build_class_name(name: str):
    return "".join(map(str.capitalize, name_split_reg.split(name)))


def check_dir(path: str):
    p = Path(path)
    if not path or not p.is_dir():
        raise ValueError(f"hnlp: {path} should be a path.")


def check_file(path: str):
    p = Path(path)
    if not path or not p.is_file():
        raise ValueError(f"hnlp: {path} should be a file.")


def build_config_from_json(json_path: str):
    js = pnlp.read_json(json_path)
    # like argparse.Namespace
    Config = namedtuple("Config", js.keys())
    return Config(**js)


def build_pretrained_config_from_json(pretrained_config, json_path: str):
    js = pnlp.read_json(json_path)
    return pretrained_config(**js)


def get_attr(typ: type, attr: str, default: Any):
    if not hasattr(typ, attr):
        return default
    return getattr(typ, attr)


def check_parameter(func):
    @wraps(func)
    def wrapper(config):
        config = ADict(config)
        return func(config)

    return wrapper
