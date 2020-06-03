"""

Corpus => Preprocessor => Tokenizer => Dataset, Sampler => Dataloader

if params.n_gpu <= 1:
    sampler = RandomSampler(dataset)
else:
    sampler = DistributedSampler(dataset)

if params.group_by_size:
    groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
    sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
else:
    sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)


"""


from hnlp.dataprocessor.corpus import Corpus
from hnlp.dataprocessor.preprocessor import Preprocessor
from hnlp.dataprocessor.tokenizer import Tokenizer
from hnlp.dataprocessor.datamanager import DataManager

