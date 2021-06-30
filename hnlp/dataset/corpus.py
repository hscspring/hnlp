from dataclasses import dataclass
from typing import Optional, Dict
import json
from pnlp import Reader


from hnlp.node import Node
from hnlp.register import Register


@dataclass
class Corpus(Node):

    """

    Parameters
    -----------
    name: corpus name
    path: corpus path, could be a directory or file.
        Each line of data MUST BE JSON, which contains "text" and "label" keys
    pattern: file pattern for file names
    """

    name: str
    path: Optional[str]
    label_map: Optional[Dict[str, int]] = None
    pattern: str = ".json"

    def __post_init__(self):
        super().__init__()
        self.identity = "corpus"
        cls_name = "_".join([self.name, self.identity])
        CorpusData = Register.get(cls_name)
        if not CorpusData:
            raise NotImplementedError
        self.node = CorpusData(self.path, self.label_map, self.pattern)

    def __len__(self):
        return len(self.node)

    def __iter__(self):
        for item in self.node:
            yield item

    def __call__(self):
        return self.node()


@Register.register
@dataclass
class CustomCorpus:

    path: str
    label_map: Dict[str, int]
    pattern: str

    def __post_init__(self):
        self.reader = Reader(self.pattern)

    def __iter__(self):
        for line in self.reader(self.path):
            js = json.loads(line.text.strip())
            label = js["label"]
            if label:
                if isinstance(label, str):
                    label = self.label_map[label]
                elif isinstance(label, list):
                    label = [self.label_map[v] for v in label]
                else:
                    info = "hnlp: invalid label, must be str integer or list of str integer"
                    raise ValueError(info)
                js["label"] = label
            yield js

    def __call__(self):
        return iter(self)

    def __len__(self):
        i = 0
        for line in self.reader(self.path):
            i += 1
        return i
