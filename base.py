from collections import namedtuple
from addict import Dict as ADict
import pnlp
import json


class Config(ADict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        # self.__dict__ = self

    def from_json_file(self, json_file: str):
        self.__dict = pnlp.read_json(json_file)
        Config = namedtuple("Config", self.__dict.keys())
        return Config(**self.__dict)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        return self.__dict or dict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
