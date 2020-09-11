from dataclasses import dataclass
from typing import Iterator, Iterable, List, Tuple, Any

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data import BatchSampler, DataLoader


from hnlp.node import Node
from hnlp.register import Register


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


@dataclass
class DataManager(Node):

    name: str = "random"
    min_seq_len: int = 1
    max_seq_len: int = 512
    batch_size: int = 1
    drop_last: bool = False

    def __post_init__(self):
        super().__init__()
        self.identity = "data_manager"
        self.batch_input = True
        cls_name = "_".join([self.name, self.identity])
        Manager = Register.get(cls_name)
        if not Manager:
            raise NotImplementedError
        self.node = Manager(self.min_seq_len,
                            self.max_seq_len,
                            self.batch_size,
                            self.drop_last)


@dataclass
class BatchLoader:

    min_seq_len: int
    max_seq_len: int
    batch_size: int
    drop_last: bool

    def __call__(self, inputs:
                 List[List[str or int]] or
                 List[Tuple[List[str or int], Any]], *args):
        dataset = MapStyleDataset(
            inputs,
            self.min_seq_len,
            self.max_seq_len)
        batch_sampler = BatchSampler(
            self.sampler(dataset),
            batch_size=self.batch_size,
            drop_last=self.drop_last)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=dataset.batch_sequences)
        return loader


@Register.register
@dataclass
class RandomDataManager(BatchLoader):

    def __post_init__(self):
        self.sampler = RandomSampler


@Register.register
@dataclass
class SequenceDataManager(BatchLoader):

    def __post_init__(self):
        self.sampler = SequentialSampler


@dataclass
class MapStyleDataset(Dataset):

    data: List[List[str or int]] or List[Tuple[List[str or int], Any]]
    min_seq_len: int
    max_seq_len: int
    pad_token_id: int = 0

    def __post_init__(self):
        self.data = self.filter_sequences(self.data)
        self.length = len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return self.length

    def get_batch(self, batch_size: int):
        indexes = np.random.choice(
            self.length, batch_size, replace=False, p=None)
        chosen = [self.data[i] for i in indexes]
        return self.batch_sequences(chosen)

    def filter_sequences(self, data):
        def x_len(ele):
            if type(ele) == tuple:
                return len(ele[0])
            else:
                return len(ele)
        result = list(
            filter(lambda x:
                   self.min_seq_len <= x_len(x) <= self.max_seq_len, data))
        return result

    def batch_sequences(self, batch):
        if type(batch[0]) == tuple:
            tokens = [ele[0] for ele in batch]
            padded_tokens = self.padding_tokens(tokens)
            if len(batch[0]) > 1:
                labels = [ele[1] for ele in batch]
                return (padded_tokens, labels)
            return padded_tokens
        else:
            tokens = batch
            padded_tokens = self.padding_tokens(tokens)
            return padded_tokens

    def padding_tokens(self, tokens: list):
        def get_pad(ele):
            if type(ele[0]) == int:
                return self.pad_token_id
            else:
                return "[PAD]"
        max_len = max([len(item) for item in tokens])
        padded_tokens = [
            ele + [get_pad(ele)] * (max_len - len(ele)) for ele in tokens]
        return padded_tokens
