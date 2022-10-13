"""
Corpus Module
================

The core module of Corpus. Support LabeledCorpus and UnLabeledCorpus.
"""

from typing import Optional, Dict, Tuple, Union, List
from addict import Dict as ADict
from pyarrow import json as pjson
from pyarrow import concat_tables
import numpy as np
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
    add_special_label: Whether to add special label for ner task
    special_label: The special label, default is "O"

    Note
    -----
    Your string label will be treated as an input text. Unless you've converted them to Number by using the label_map.
    """

    def __init__(
        self,
        name: str,
        pattern: str = "*.*",
        keys: Optional[Tuple[str, str]] = ("text", "label"),
        shuffle: bool = False,
        label_map: Dict[str, int] = ADict(),
        add_special_label: bool = False,
        special_label: str = "O",
    ):

        super().__init__()
        self.name = name
        self.identity = "corpus"

        if name == "labeled":
            self.node = super().get_cls(
                self.identity,
                self.name)(
                pattern,
                keys,
                shuffle,
                label_map,
                add_special_label,
                special_label,
            )
        else:
            self.node = super().get_cls(
                self.identity,
                self.name)(
                pattern,
                shuffle)

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
        add_special_label: bool,
        special_label: str = "O",
    ):
        self.pattern = pattern
        self.keys = list(keys)
        self.shuffle = shuffle
        self.label_map = label_map
        self.add_special_label = add_special_label
        self.special_label = special_label
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

    def map_label_to_int(
            self, labels: Union[str, np.ndarray]) -> Union[List[int], int]:
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
            res = []
            if self.add_special_label:
                int_spe_label = self.label_map[self.special_label]
                res.append(int_spe_label)
            for v in labels:
                int_label = self.label_map.get(v, v)
                res.append(int_label)
            if self.add_special_label:
                res.append(int_spe_label)
            return res
        else:
            return self.label_map.get(labels)

    def extract_and_transform(self, df: pd.DataFrame):
        if self.label_map:
            df["label"] = df["label"].apply(lambda x: self.map_label_to_int(x))
        data = df[self.keys]
        return data

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        for v in self.data.itertuples(index=False):
            yield tuple(v)

    def __getitem__(self, i: int):
        return self.data.iloc[i]

    def __call__(self, path: str, *args):
        if args:
            sample_num = args[0]
        else:
            sample_num = 0
        df = self.read_json(path)
        self.data = self.extract_and_transform(df)
        if self.shuffle:
            self.data = shuffle(self.data)
        res = []
        i = 0
        for v in self:
            i += 1
            res.append(v)
            if sample_num > 0 and i >= sample_num:
                break
        return res


@ Register.register
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
        shuffle: bool,
    ):

        self.pattern = pattern
        self.shuffle = shuffle
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

    def __call__(self, path: str, *args):
        if args:
            sample_num = args[0]
        else:
            sample_num = 0
        self.data = self.read_file(path)
        if self.shuffle:
            self.data = shuffle(self.data)
        if sample_num > 0:
            return self.data[:sample_num]
        return self.data
