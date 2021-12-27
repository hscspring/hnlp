from typing import List, Tuple, Any

from hnlp.node import Node
from hnlp.register import Register


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
        self.identity = "data_manager"
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

    def __call__(self, inputs: List[List[str or int]] or
                 List[Tuple[List[str or int], Any]], *args):
        self.dataset = MapStyleDataset(inputs, self.min_seq_len,
                                       self.max_seq_len, self.dynamic_length)
        batch_sampler = BatchSampler(
            self.sampler(self.dataset),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        loader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.dataset.batch_sequences,
        )
        return loader


@Register.register
class RandomDataManagerPt(BatchLoader):

    def __init__(self):
        self.sampler = RandomSampler


@Register.register
class SequenceDataManagerPt(BatchLoader):

    def __init__(self):
        self.sampler = SequentialSampler
