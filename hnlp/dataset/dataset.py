from typing import List, Tuple, Union, TypeVar

import numpy as np
from hnlp.config import SpeToken


DatasetType = TypeVar(
    "DatasetType",
    List[List[Union[str, int]]],
    List[Tuple[List[Union[str, int]], Union[str, int, float]]]
)


class MapStyleDataset:

    def __init__(
        self,
        data: DatasetType,
        min_seq_len: int = 0,
        max_seq_len: int = 512,
        dynamic_length: bool = True,
        pad_token: str = SpeToken.pad,
        pad_token_id: int = 0
    ):
        self.data = self.filter_sequences(data)
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.dynamic_length = dynamic_length
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.length = len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return self.length

    def get_batch(self, batch_size: int):
        indexes = np.random.choice(self.length,
                                   batch_size,
                                   replace=False,
                                   p=None)
        chosen = [self.data[i] for i in indexes]
        return self.batch_sequences(chosen)

    def filter_sequences(self, data: DatasetType):

        def x_len(ele):
            if type(ele) == tuple:
                return len(ele[0])
            else:
                return len(ele)

        result = list(filter(lambda x: self.min_seq_len <= x_len(x), data))
        return result

    def batch_sequences(self, batch: DatasetType):
        if type(batch[0]) == tuple:
            batch_tokens = [ele[0] for ele in batch]
            padded_tokens = self.padding_tokens(batch_tokens)
            if len(batch[0]) > 1:
                labels = [ele[1] for ele in batch]
                return (padded_tokens, labels)
            return padded_tokens
        else:
            batch_tokens = batch
            padded_tokens = self.padding_tokens(batch_tokens)
            return padded_tokens

    def padding_tokens(self, batch_tokens: List[List[Union[str, int]]]):

        def get_pad(ele):
            if type(ele[0]) == int:
                return self.pad_token_id
            else:
                return self.pad_token

        max_len = max([len(item) for item in batch_tokens])
        if self.dynamic_length:
            max_len = min(max_len, self.max_seq_len)
        else:
            max_len = self.max_seq_len
        padded_tokens = []
        for ele in batch_tokens:
            if len(ele) > max_len:
                ele = ele[:max_len]
            else:
                ele = ele + [get_pad(ele)] * (max_len - len(ele))
            padded_tokens.append(ele)
        return padded_tokens
