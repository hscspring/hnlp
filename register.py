from hnlp.utils import build_class_name


class Register:

    _dict = {}

    @classmethod
    def register(cls, _class):
        cls._dict[_class.__name__] = _class
        return _class

    @classmethod
    def get(cls, name: str):
        cls_name = build_class_name(name)
        return Register._dict.get(name) or Register._dict.get(cls_name)


class Base:

    registry = {}

    def __init_subclass__(cls, **kwargs):
        cls.registry[cls.__name__] = cls
        super().__init_subclass__(**kwargs)
