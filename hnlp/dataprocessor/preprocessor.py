from dataclasses import dataclass, field
from pnlp import Text
from typing import List
import re

from hnlp.node import Node


@dataclass
class Preprocessor(Node):

    name: str = "common"
    pats: List = field(default_factory=list)

    def __post_init__(self):
        super().__init__()
        self.identity = "proprocessor"
        if self.name == "common":
            self.node = ComonPreprocessor(self.pats)
        else:
            raise NotImplementedError


@dataclass
class ComonPreprocessor:
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
