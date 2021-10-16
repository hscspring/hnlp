from dataclasses import dataclass
from typing import List, Tuple, Any

from hnlp.node import Node
from hnlp.register import Register


@dataclass
class DataManager(Node):

    name: str = "random"
    batch_size: int = 1
    min_seq_len: int = 1
    max_seq_len: int = 512
    dynamic_length: bool = False
    drop_last: bool = False

    def __post_init__(self):
        super().__init__()
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


@dataclass
class BatchLoader:

    batch_size: int
    min_seq_len: int
    max_seq_len: int
    dynamic_length: bool
    drop_last: bool

    def __call__(
        self,
        inputs: List[List[str or int]] or List[Tuple[List[str or int], Any]],
        *args
    ):
        self.dataset = MapStyleDataset(
            inputs, self.min_seq_len, self.max_seq_len, self.dynamic_length
        )
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
@dataclass
class RandomDataManagerPt(BatchLoader):
    def __post_init__(self):
        self.sampler = RandomSampler


@Register.register
@dataclass
class SequenceDataManagerPt(BatchLoader):
    def __post_init__(self):
        self.sampler = SequentialSampler
