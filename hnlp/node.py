from typing import List, TypeVar, Generic
from functools import partial
from torch import Tensor

from hnlp.base import ModelInputType, ModelLabelType

"""
This is the base node for generating pipes.

Actually the deeplearning process is just like a pipeline,
data is not that important for the framework.

Here we referenced from saveral excellent repos:

- https://pytorch.org/text/_modules/torchtext/data/pipeline.html#Pipeline
- https://github.com/robdmc/consecution

"""

T = TypeVar('T', str, tuple, List[str], List[tuple], ModelInputType)


class Node:

    identity = None
    batch_input = False

    def __init__(self):
        self.nodes = [self]

    def __call__(self, inputs: T):
        return self.call(inputs)

    def call(self, inputs: T):
        if self.batch_input:
            return self.node(inputs)
        
        if isinstance(inputs, str) == True:
            return self.__call_str(inputs)
        elif isinstance(inputs, tuple) == True:
            return self.__call_tuple(inputs)
        elif isinstance(inputs, list) == True:
            return self.__call_list(inputs)
        else:
            # inputs(dict) processed by pretrained processor 
            return self.node(inputs)

    def __call_str(self, inputs: str):
        return self.node(inputs)

    def __call_tuple(self, inputs: tuple):
        tmp = [self.node(inputs[0])]
        for ele in inputs[1:]:
            tmp.append(ele)
        return tuple(tmp)

    def __call_list(self, inputs: list):
        result = []
        # most node only accept str input.
        for inp in inputs:
            if isinstance(inp, tuple) == True:
                new = self.__call_tuple(inp)
            else:
                new = self.node(inp)
            result.append(new)
        return result

    # for pipeline
    # All calls are happened on the Middleware Layer (Corpus, Tokenizer, etc.)
    # We DONOT call the actual object.

    def run(self, inputs: T):
        for node in self.nodes:
            # this is actually the above `__call__` function
            inputs = node(inputs)
        return inputs

    def __rshift__(self, other):
        self.nodes.append(other)
        return self

    # for Model Node `fit`
    def fit(self, inputs: ModelInputType, labels: ModelLabelType):
        for node in self.nodes:
            if hasattr(node, "fit"):
                inputs = node.fit(inputs, labels)
            else:
                inputs = node(inputs)
        return inputs

    # for Model Node `predict`
    def predict(self, inputs: ModelInputType):
        for node in self.nodes:
            if hasattr(node, "predict"):
                inputs = node.predict(inputs)
            else:
                inputs = node(inputs)
        return inputs


"""
This is a functional pipeline tools
which support N(fun1) >> N(fun2) ...

We referenced this excellent design from :

- https://github.com/kachayev/fn.py 

"""


class N:

    def __init__(self, f=lambda arg: arg, *args, **kwargs):
        self.f = partial(f, *args, **kwargs) if any([args, kwargs]) else f

    def __ensure_callable(self, f):
        return self.__class__(*f) if isinstance(f, tuple) else f

    @classmethod
    def __compose(cls, g, f):
        return cls(lambda *args, **kwargs: g(f(*args, **kwargs)))

    def __rshift__(self, g):
        return self.__class__.__compose(self.__ensure_callable(g), self.f)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)
