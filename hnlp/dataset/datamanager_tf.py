from posixpath import split
from typing import List, Callable

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

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
        return_type: str = "tf",
        split_val: bool = True,
    ):
        super().__init__()
        self.name = name
        self.identity = "data_manager_tf"

        self.split_val = split_val

        self.node = super().get_cls(self.identity, self.name)(
            batch_size,
            min_seq_len,
            max_seq_len,
            dynamic_length,
            drop_last,
            collate_fn,
            return_type,
            self.split_val,
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
        split_val: bool,
    ):
        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.dynamic_length = dynamic_length
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.return_type = return_type
        self.split_val = split_val

    def __call__(self, inputs: List[DatasetType], *args):
        if not self.collate_fn:
            collate_fn = MapStyleDataset.batch_sequences
        else:
            collate_fn = self.collate_fn

        if self.split_val:
            train, val = train_test_split(inputs, test_size=0.2)
            self.train_dataset = MapStyleDataset(train, self.min_seq_len)
            self.val_dataset = MapStyleDataset(val, self.min_seq_len)
            train_loader = DataLoader(
                self.train_dataset,
                self.batch_size,
                self.random_sample,
                self.drop_last,
                collate_fn,
                self.max_seq_len,
                self.dynamic_length,
                self.return_type
            )
            val_loader = DataLoader(
                self.val_dataset,
                self.batch_size,
                False,
                self.drop_last,
                collate_fn,
                self.max_seq_len,
                self.dynamic_length,
                self.return_type
            )
            return train_loader, val_loader
        else:
            train = inputs
            self.train_dataset = MapStyleDataset(train, self.min_seq_len)
            loader = DataLoader(
                self.train_dataset,
                self.batch_size,
                self.random_sample,
                self.drop_last,
                collate_fn,
                self.max_seq_len,
                self.dynamic_length,
                self.return_type
            )
            return loader


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
        split_val: bool,
    ):
        super().__init__(
            batch_size,
            min_seq_len,
            max_seq_len,
            dynamic_length,
            drop_last,
            collate_fn,
            return_type,
            split_val
        )
        self.random_sample = True

    def __call__(self, inputs: List[DatasetType], *args):
        return super().__call__(inputs, *args)


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
        split_val: bool,
    ):
        super().__init__(
            batch_size,
            min_seq_len,
            max_seq_len,
            dynamic_length,
            drop_last,
            collate_fn,
            return_type,
            split_val
        )
        self.random_sample = False

    def __call__(self, inputs: List[DatasetType], *args):
        return super().__call__(inputs, *args)


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
            rng = np.random.default_rng(42)
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
