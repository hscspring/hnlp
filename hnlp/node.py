from typing import List
from functools import partial

"""
This is the base node for generate pipes.

Actually the deeplearning process is just like a pipeline,
data is not that important for the framework.

Here we referenced from saveral excellent repos:

- https://pytorch.org/text/_modules/torchtext/data/pipeline.html#Pipeline
- https://github.com/robdmc/consecution

"""


class Node:

    identity = None
    join = False

    def __init__(self):
        self.nodes = [self]

    def __call__(self, inputs: str or tuple or List[str or tuple]):
        return self.call(inputs)

    def call(self, inputs: str or tuple or List[str or tuple]):
        if self.join:
            return self.node(inputs)

        if isinstance(inputs, str) == True:
            return self.__call_str(inputs)

        if isinstance(inputs, tuple) == True:
            return self.__call_tuple(inputs)

        return self.__call_list(inputs)

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

    def run(self, inputs: str or tuple or List[str or tuple]):
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
    
    def __init__(self, f = lambda arg: arg, *args, **kwargs):
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
