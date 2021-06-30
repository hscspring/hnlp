from dataclasses import dataclass, field
from pnlp import Text
from typing import List
import re

from hnlp.node import Node
from hnlp.register import Register


@dataclass
class Preprocessor(Node):

    name: str = "common"
    pats: List = field(default_factory=list)

    def __post_init__(self):
        super().__init__()
        self.identity = "preprocessor"
        cls_name = "_".join([self.name, self.identity])
        DataPreProcessor = Register.get(cls_name)
        if not DataPreProcessor:
            raise NotImplementedError
        self.node = DataPreProcessor(self.pats)


@Register.register
@dataclass
class CommonPreprocessor:
    # clean, replace ...
    pats: List = field(default_factory=list)

    def __post_init__(self):
        self.cleaner = None
        self.replacer = None
        if self.pats:
            self.cleaner = Text(self.pats)

    def __call__(self, text: str) -> str:
        if self.cleaner:
            text = self.cleaner.clean(text)
        if self.replacer:
            text = self.replacer.replace(text)
        return text
