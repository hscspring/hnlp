from typing import List, Tuple, Union, TypeVar

import numpy as np
from hnlp.config import SpeToken


DatasetType = TypeVar(
    "DatasetType",
    List[List[Union[str, int]]],
    List[Tuple[List[Union[str, int]], Union[str, int, float, List[int]]]]
)


class MapStyleDataset:

    def __init__(
        self,
        data: DatasetType,
        min_seq_len: int = 0,
    ):
        self.min_seq_len = min_seq_len
        self.data = self.filter_sequences(data)

    def __iter__(self):
        for v in self.data:
            yield v

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def filter_sequences(self, data: DatasetType):

        def x_len(ele):
            if type(ele) == tuple:
                return len(ele[0])
            else:
                return len(ele)

        result = list(filter(lambda x: self.min_seq_len <= x_len(x), data))
        return result

    def get_random_batch(self, batch_size: int):
        indexes = np.random.choice(self.length,
                                   batch_size,
                                   replace=False,
                                   p=None)
        chosen = [self.data[i] for i in indexes]
        return MapStyleDataset.batch_sequences(
            chosen, self.max_seq_len, self.dynamic_length)

    @staticmethod
    def batch_sequences(
        batch: DatasetType,
        max_seq_len: int,
        dynamic_length: bool
    ):
        if type(batch[0]) == tuple:
            new_batch = []
            ele_len = len(batch[0])
            for i in range(ele_len):
                tks = [v[i] for v in batch]
                padded = MapStyleDataset.padding_tokens(
                    tks, max_seq_len, dynamic_length)
                new_batch.append(padded)
            return tuple(new_batch)
        else:
            batch_tokens = batch
            padded_tokens = MapStyleDataset.padding_tokens(
                batch_tokens, max_seq_len, dynamic_length)
            return padded_tokens

    @staticmethod
    def padding_tokens(
        batch_tokens: List[List[Union[str, int]]],
        max_seq_len: int,
        dynamic_length: bool,
        pad_value: int = 0,
    ):
        max_len = max([len(item) for item in batch_tokens])
        if dynamic_length:
            max_len = min(max_len, max_seq_len)
        else:
            max_len = max_seq_len
        padded_tokens = []
        for ele in batch_tokens:
            if len(ele) > max_len:
                ele = ele[:max_len]
            else:
                ele = ele + [pad_value] * (max_len - len(ele))
            padded_tokens.append(ele)
        return padded_tokens
