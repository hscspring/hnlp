from typing import List, Callable

import tensorflow as tf
import numpy as np

from hnlp.node import Node
from hnlp.register import Register
from hnlp.dataset.dataset import DatasetType, MapStyleDataset


class DataManager(Node):

    def __init__(
        self,
        name: str = "sequence",
        batch_size: int = 1,
        min_seq_len: int = 1,
        max_seq_len: int = 512,
        dynamic_length: bool = False,
        drop_last: bool = False,
        collate_fn: Callable = None,
        return_type: str = "tf"
    ):
        super().__init__()
        self.name = name
        self.identity = "data_manager_tf"

        self.node = super().get_cls(self.identity, self.name)(
            batch_size,
            min_seq_len,
            max_seq_len,
            dynamic_length,
            drop_last,
            collate_fn,
            return_type
        )

    # override the Node's call function
    def call(self, inp, *args):
        return self.node(inp, *args)


class BatchLoader:

    def __init__(
        self,
        batch_size: int,
        min_seq_len: int,
        max_seq_len: int,
        dynamic_length: bool,
        drop_last: bool,
        collate_fn: Callable,
        return_type: str,
    ):
        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.dynamic_length = dynamic_length
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.return_type = return_type

    def __call__(self, inputs: List[DatasetType], *args):
        self.dataset = MapStyleDataset(inputs, self.min_seq_len)
        if not self.collate_fn:
            collate_fn = MapStyleDataset.batch_sequences
        else:
            collate_fn = self.collate_fn
        loader = DataLoader(
            self.dataset,
            self.batch_size,
            self.random_sample,
            self.drop_last,
            collate_fn,
            self.max_seq_len,
            self.dynamic_length,
            self.return_type
        )
        return loader


class DataLoader:

    def __init__(
            self,
            dataset: MapStyleDataset,
            batch_size: int,
            random_sample: bool,
            drop_last: bool,
            collate_fn: Callable,
            max_seq_len: int,
            dynamic_length: bool,
            return_type: str,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_sample = random_sample
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.max_seq_len = max_seq_len
        self.dynamic_length = dynamic_length
        self.return_type = return_type

        if self.random_sample:
            rng = np.random.default_rng()
            rng.shuffle(self.dataset.data)

        self.length = len(self.dataset)
        self.batch_num = self.length // self.batch_size

        if self.drop_last == False and self.length % self.batch_size != 0:
            self.batch_num += 1

    def __iter__(self):
        for i in range(self.batch_num):
            batch = self.dataset.data[
                i * self.batch_size: (i + 1) * self.batch_size
            ]
            batch = self.collate_fn(
                batch,
                max_seq_len=self.max_seq_len,
                dynamic_length=self.dynamic_length
            )
            if self.return_type == "tf":
                batch = (tf.constant(v, dtype=tf.int32) for v in batch)
            elif self.return_type == "np":
                batch = (np.array(v, dtype=np.int32) for v in batch)
            yield batch

    def __len__(self):
        return self.length


@Register.register
class RandomDataManagerTf(BatchLoader):

    def __init__(
        self,
        batch_size: int,
        min_seq_len: int,
        max_seq_len: int,
        dynamic_length: bool,
        drop_last: bool,
        collate_fn: Callable,
        return_type: str,
    ):
        super().__init__(
            batch_size,
            min_seq_len,
            max_seq_len,
            dynamic_length,
            drop_last,
            collate_fn,
            return_type)
        self.random_sample = True


@Register.register
class SequenceDataManagerTf(BatchLoader):

    def __init__(
        self,
        batch_size: int,
        min_seq_len: int,
        max_seq_len: int,
        dynamic_length: bool,
        drop_last: bool,
        collate_fn: Callable,
        return_type: str,
    ):
        super().__init__(
            batch_size,
            min_seq_len,
            max_seq_len,
            dynamic_length,
            drop_last,
            collate_fn,
            return_type)
        self.random_sample = False
