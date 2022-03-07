"""
Corpus Module
================

The core module of Corpus. Support LabeledCorpus and UnLabeledCorpus.
"""

from typing import Optional, Dict, Tuple
from addict import Dict as ADict
from pyarrow import json as pjson
from pyarrow import concat_tables
import pandas as pd
from sklearn.utils import shuffle
from pnlp import Reader

from hnlp.node import Node
from hnlp.register import Register


class Corpus(Node):
    """
    Corpus middleware.

    Parameters
    -----------
    name: Corpus name
        Could be "labeled" or "unlabeled"
    pattern: File pattern for file names in the given directory
        If you input a file (not a directory), it won't work
        The default value is "*.*"
    keys: For labeled dataset, your file(s) should be json format with several keys
        The default value is ("text", "label")
    shuffle: Whether to shuffle the input data
    label_map: A Dict to convert your string label to Number, if the label is Number, you do not need to set
        The default value is ADict()

    Note
    -----
    Your string label will be treated as an input text. Unless you've converted them to Number by using the label_map.
    """

    def __init__(
            self,
            name: str,
            pattern: str = "*.*",
            keys: Optional[Tuple[str, str]] = ("text", "label"),
            shuffle: bool = True,
            label_map: Dict[str, int] = ADict(),
    ):

        super().__init__()
        self.name = name
        self.pattern = pattern
        self.keys = keys
        self.shuffle = shuffle
        self.label_map = label_map
        self.identity = "corpus"
        self.node = super().get_cls(self.identity,
                                    self.name)(self.pattern, self.keys,
                                               self.shuffle, self.label_map)

    def __len__(self):
        return len(self.node)

    def __iter__(self):
        for item in self.node:
            yield item

    def __getitem__(self, i: int):
        return self.node[i]


@Register.register
class LabeledCorpus:
    """
    LabeledCorpus module

    Only support lines of json file. Each file should contain a "text" key and a "label" key.

    Parameters
    -----------
    path: json corpus file.
    """

    def __init__(
        self,
        pattern: str,
        keys: Optional[Tuple[str, str]],
        shuffle: bool,
        label_map: Dict[str, int],
    ):
        self.pattern = pattern
        self.keys = list(keys)
        self.shuffle = shuffle
        self.label_map = label_map
        self.data = pd.DataFrame()
        self.reader = Reader()

    def read_json(self, path: str) -> pd.DataFrame:
        res = []
        for js_file in self.reader.gen_files(path, self.pattern):
            table = pjson.read_json(js_file)
            res.append(table)
        tab = concat_tables(res)
        df = tab.to_pandas()
        return df

    def extract_and_transform(self, df: pd.DataFrame):
        if self.label_map:
            df["label"] = df["label"].apply(lambda x: self.label_map.get(x))
        data = df[self.keys]
        return data

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        for v in self.data.itertuples(index=False):
            yield tuple(v)

    def __getitem__(self, i: int):
        return self.data.iloc[i]

    def __call__(self, path: str):
        df = self.read_json(path)
        self.data = self.extract_and_transform(df)
        if self.shuffle:
            self.data = shuffle(self.data)
        res = []
        for v in self:
            res.append(v)
        return res


@Register.register
class UnlabeledCorpus:
    """
    UnlabeldCorpus module

    Parameters
    -----------
    pattern: Pattern for file in the directory
    label_map: Label map for input, should ignore
    """

    def __init__(
        self,
        pattern: str,
        keys: Optional[Tuple[str, str]],
        shuffle: bool,
        label_map: dict,
    ):

        self.pattern = pattern
        self.keys = keys
        self.shuffle = shuffle
        self.label_map = label_map
        self.data = []
        self.reader = Reader(self.pattern)
        self._len = 0

    def read_file(self, path: str):
        data = []
        for line in self.reader(path):
            self._len += 1
            data.append(line.text.strip())
        return data

    def __iter__(self):
        for line in self.data:
            yield line

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        return self.data[i]

    def __call__(self, path: str):
        self.data = self.read_file(path)
        if self.shuffle:
            self.data = shuffle(self.data)
        return self.data
