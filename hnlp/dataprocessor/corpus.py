from dataclasses import dataclass
from pathlib import Path
from pnlp import Reader


from hnlp.node import Node
from hnlp.register import Register


@dataclass
class Corpus(Node):
    
    name: str = "custom"
    pattern: str = ".*"
    sep: str = "|||"

    def __post_init__(self):
        super().__init__()
        self.identity = "corpus"
        cls_name = "_".join([self.name, self.identity])
        CorpusData = Register.get(cls_name)
        if not CorpusData:
            raise NotImplementedError
        self.node = CorpusData(self.pattern, self.sep)


@Register.register
@dataclass
class CustomCorpus:

    pattern: str = ".*"
    sep: str = "|||"

    def __post_init__(self):
        self.reader = Reader(self.pattern)

    def __call__(self, data_path: str):
        result = []
        for line in self.reader(data_path):
            item = tuple(line.text.split(self.sep))
            if len(item) == 1:
                item = item[0]
            result.append(item)
        return result
