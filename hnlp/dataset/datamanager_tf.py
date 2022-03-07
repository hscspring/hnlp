from typing import List, Tuple, Any

from hnlp.node import Node
from hnlp.register import Register
from hnlp.dataset.dataset import DatasetType


class DataManager(Node):

    def __init__(
        self,
        name: str = "random",
        batch_size: int = 1,
        min_seq_len: int = 1,
        max_seq_len: int = 512,
        dynamic_length: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.dynamic_length = dynamic_length
        self.drop_last = drop_last
        self.identity = "data_manager_tf"
        self.node = super().get_cls(self.identity, self.name)(
            self.batch_size,
            self.min_seq_len,
            self.max_seq_len,
            self.dynamic_length,
            self.drop_last,
        )

    def call(self, inp):
        return self.node(inp)


class BatchLoader:

    def __init__(
        self,
        batch_size: int,
        min_seq_len: int,
        max_seq_len: int,
        dynamic_length: bool,
        drop_last: bool,
    ):
        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.dynamic_length = dynamic_length
        self.drop_last = drop_last

    def __call__(self, inputs: List[DatasetType], *args):
        self.dataset = MapStyleDataset(inputs, self.min_seq_len,
                                       self.max_seq_len, self.dynamic_length)
        return tf.data.Dataset.from_slices(self.dataset.data)


@Register.register
class RandomDataManagerPt(BatchLoader):

    def __init__(self):
        self.sampler = RandomSampler


@Register.register
class SequenceDataManagerPt(BatchLoader):

    def __init__(self):
        self.sampler = SequentialSampler
