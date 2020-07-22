from dataclasses import dataclass
from functools import wraps

from hnlp.utils import build_class_name


@dataclass
class Register:

    _dict = {}

    @classmethod
    def register(cls, _class):
        Register._dict[_class.__name__] = _class

        @wraps(_class)
        def wrapper(*args, **kwargs):
            return _class(*args, **kwargs)
        return wrapper

    @classmethod
    def get(cls, name: str):
        cls_name = build_class_name(name)
        return Register._dict.get(name) or Register._dict.get(cls_name)
