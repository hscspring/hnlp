from dataclasses import dataclass, field
from pnlp import Text
from typing import List

from hnlp.node import Node
from hnlp.register import Register


@dataclass
class Preprocessor(Node):

    name: str = "clean"
    pats: List = field(default_factory=list)

    def __post_init__(self):
        super().__init__()
        self.identity = "preprocessor"
        self.node = super().get_cls(self.identity, self.name)(self.pats)


@Register.register
@dataclass
class CleanPreprocessor:

    pats: List = field(default_factory=list)

    def __post_init__(self):
        self.cleaner = None
        if self.pats:
            self.cleaner = Text(self.pats)

    def __call__(self, text: str) -> str:
        if self.cleaner:
            text = self.cleaner.clean(text)
        return text
