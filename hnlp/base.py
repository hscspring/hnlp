from dataclasses import dataclass
from functools import wraps, partial
from typing import List, Tuple, Dict, TypeVar


from torch import Tensor
import torch
import torch.nn as nn

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
        if not inputs and len(args) > 0:
            inputs = args[0]
        if not isinstance(inputs, Tensor) and not inputs:
            raise ValueError("hnlp: Invalid inputs.")
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


@dataclass
class PretrainedBaseModel:

    # Here we use the `convert_model_input` mainly for those
    # who only want a pretrained model.
    @convert_model_input
    def __call__(self, inputs: ModelInputType):
        """
        inputs element shape: [batch_size, seq_len]
        - if output_hidden_states and output_attentions is False. Outputs is a two elements tuple:
            - outputs[0] is the `last_hidden_state`
            - outputs[1] is the `pooled_output`
        - if both are True, Outputs is a four elements tuple:
            - outputs[0] and outputs[1] like before
            - outputs[2] is all_hidden_states, 
              length = layer_num + 1(embedding output), 
              all_hidden_states[-1] == outputs[0], 
              shape is length * ([1, seq_len, hidden_size])
            - outputs[3] is all_attentions, 
              length = layer_num, 
              shape is length * ([1, attention_heads_num, seq_len, seq_len])
        """
        if self.is_training:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
        return outputs
