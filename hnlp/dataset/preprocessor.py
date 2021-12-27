from pnlp import Text
from typing import List, Optional

from hnlp.node import Node
from hnlp.register import Register


class Preprocessor(Node):

    def __init__(self, name: str = "clean", pats: Optional[List[str]] = None):
        super().__init__()
        self.name = name
        self.pats = pats
        self.identity = "preprocessor"
        self.node = super().get_cls(self.identity, self.name)(self.pats)


@Register.register
class CleanPreprocessor:

    def __init__(self, pats: List[str]):
        self.pats = pats
        self.cleaner = None
        if self.pats:
            self.cleaner = Text(self.pats)

    def __call__(self, text: str) -> str:
        if self.cleaner:
            text = self.cleaner.clean(text)
        return text
