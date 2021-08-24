from typing import List, TypeVar
from functools import partial
from hnlp.register import Register

"""
This is the base node for generating pipes.

Actually the deeplearning process is just like a pipeline,
data is not that important for the framework.
Especially for the data processing.

Here we referenced from saveral excellent repos:

- https://pytorch.org/text/_modules/torchtext/data/pipeline.html#Pipeline
- https://github.com/robdmc/consecution

"""


T = TypeVar("T", str, tuple, List[str], List[tuple])


class Node:

    identity = None

    def __init__(self):
        self.nodes = [self]

    def get_cls(self, identity: str, name: str):
        cls_name = "_".join([name, identity])
        cls = Register.get(cls_name)
        if not cls:
            raise NotImplementedError
        return cls

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs: T):

        if isinstance(inputs, str):
            return self.__call_str(inputs)
        elif isinstance(inputs, tuple):
            return self.__call_tuple(inputs)
        elif isinstance(inputs, list):
            return self.__call_list(inputs)
        else:
            return self.node(inputs)

    def __call_str(self, inputs: str):
        return self.node(inputs)

    def __call_tuple(self, inputs: tuple):
        res = []
        for ele in inputs:
            if isinstance(ele, str):
                out = self.node(ele)
            else:
                out = ele
            res.append(out)
        return tuple(res)

    def __call_list(self, inputs: list):
        result = []
        # most node only accept str input.
        for inp in inputs:
            if isinstance(inp, tuple):
                out = self.__call_tuple(inp)
            else:
                out = self.node(inp)
            result.append(out)
        return result

    # for pipeline
    # All calls are happened on the Middleware Layer (Corpus, Tokenizer, etc.)
    # We DONOT call the actual object.

    def run(self, inputs):
        for node in self.nodes:
            # this is actually the above `__call__` function
            inputs = node(inputs)
        return inputs

    def __rshift__(self, other):
        self.nodes.append(other)
        return self


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
