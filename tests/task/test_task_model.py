import torch
from torch import Tensor

from hnlp.dataprocessor.corpus import Corpus
from hnlp.dataprocessor.preprocessor import Preprocessor
from hnlp.dataprocessor.tokenizer import Tokenizer
from hnlp.dataprocessor.datamanager import DataManager
from hnlp.task.model import Model


import pnlp

config = pnlp.read_yaml("tests/config.yaml")


def get_data_loader(data_path: str):
    vocab_path = "tests/task/vocab.txt"
    pipe = (Corpus("custom") >>
            Preprocessor("common") >>
            Tokenizer("bert", vocab_path) >>
            DataManager(batch_size=8, drop_last=True))
    dataloader = pipe.run(data_path)
    return dataloader


def test_model_fit():
    model = Model(
        "BertFcClassifier",
        model_path=config.get("pretrained").get("bert"),
        is_training=True)
    train_dataloader = get_data_loader("tests/task/train.txt")
    valid_dataloader = get_data_loader("tests/task/valid.txt")
    model.fit(train_dataloader, valid_dataloader)
