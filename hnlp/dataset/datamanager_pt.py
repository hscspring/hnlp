from typing import List, Tuple, Any

from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data import BatchSampler, DataLoader

from hnlp.node import Node
from hnlp.register import Register
from hnlp.dataset import MapStyleDataset
"""
There are many different usages in the pytorch doc (https://pytorch.org/docs/stable/data.htm)

You could sample indices or data, with weight, with batch, with distributed, etc.

Here is some examples:

data = "I love you , and you love me , too .".split()

##### indices sampler #####

list(SequentialSampler(data))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

list(RandomSampler(data))
[8, 1, 6, 0, 3, 9, 5, 2, 7, 10, 4]

# if replacement=True, sample number could be any int. else should be smaller than len(weight).
list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 6, replacement=False))
[0, 3, 4, 5, 1, 2]


##### data sampler #####

list(SubsetRandomSampler(data))
['you', '.', 'I', 'too', 'and', ',', 'love', ',', 'love', 'you', 'me']


##### batch sampler #####

list(BatchSampler(RandomSampler(data), batch_size=3, drop_last=False))
[[0, 2, 5], [8, 10, 6], [1, 3, 4], [7, 9]]

list(BatchSampler(SubsetRandomSampler(data), batch_size=3, drop_last=False))
[['love', ',', 'me'], ['too', 'and', 'I'], [',', 'love', 'you'], ['you', '.']]


There is also a `DataLoader` to use. The doc says:
DataLoader supports automatically collating individual fetched data samples
into batches via arguments batch_size, drop_last, and batch_sampler.

For example:

list(DataLoader(data, batch_size= 3, sampler = RandomSampler(data), drop_last=True))
[['and', ',', 'you'], ['you', 'too', '.'], ['I', 'love', 'me']]


list(DataLoader(data,  batch_sampler = BatchSampler(RandomSampler(data), batch_size=3, drop_last=True)))
[[',', 'and', 'me'], ['love', 'you', ','], ['.', 'love', 'too']]


Here we just take good use of the excellent existed functions.
But we will make it a litte easier, we do not take too many choices.
We stick to the 80-20 principle, that is: For 80% normal users to 80% normal functions.


Our final choice is follow the `Sampler => BatchSampler => Dataloader` path
which is borrowed from https://github.com/huggingface/transformers/tree/master/examples/distillation.
They have used a `GroupedBatchSampler`, that is another kind of `BatchSampler`,
so split the processing to finer granularity might be a proper solution.


PyTorch also has a `torchtext` module, with several more abstractive functions.
For example, `BucketIterator`(https://pytorch.org/text/data.html#bucketiterator)
which to batch the data with similar lengths.
Besides, `Dataset`, `Pipeline`, `Fields` are convenient in some occasion.


"""


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
        self.identity = "data_manager_pt"
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
        self.dataset = MapStyleDataset(
            inputs, self.min_seq_len,
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
        super().__init__()
        self.sampler = RandomSampler


@Register.register
class SequenceDataManagerPt(BatchLoader):

    def __init__(self):
        super().__init__()
        self.sampler = SequentialSampler
