from dataclasses import dataclass
from collections.abc import Iterable
from typing import Dict
from gensim.models import fasttext, FastText
from addict import Dict as ADict

from hnlp.register import Register
from hnlp.config import default_model_home, default_fasttext_word2vec_config


@Register.register
@dataclass
class FastTextModel:

    name: str
    path: str
    config: Dict
    training_type: str

    def __post_init__(self):
        self.name = self.name or "word2vec"
        model_name = "/".join(("fasttext", self.name))
        model_path = default_model_home / model_name
        self.model_path = self.path or str(model_path)
        self.config = self.config or default_fasttext_word2vec_config
        self.config = ADict(self.config)
        self.training_type = self.training_type or "scratch"
        if self.training_type == "predict":
            self.load_model()

    def load_model(self):
        model = fasttext.load_facebook_model(self.model_path)
        self.model = model.wv

    def train(self, iter_corpus: Iterable):
        if self.training_type == "scratch":
            self.model = FastText(**self.config.model)
            self.model.build_vocab(sentences=iter_corpus)
            self.model.train(sentences=iter_corpus, **self.config.train)
        elif self.name == "continuous":
            self.model = fasttext.load_facebook_model(self.model_path)
            self.model.build_vocab(sentences=iter_corpus, update=True)
            self.model.train(sentences=iter_corpus, **self.config.train)
        else:
            info = "hnlp: invalid training type: {}".format(self.training_type)
            raise ValueError(info)
