from dataclasses import dataclass
from collections.abc import Iterable
from typing import Dict
from gensim.models import fasttext, FastText
from addict import Dict as ADict

from hnlp.register import Register
from hnlp.config import model_home, model_config


@Register.register
@dataclass
class FasttextPretrainedModel:

    """
    Parameters
    ----------
    name: model name, default is "word2vec"
    path: model path, default is "~/.hnlp/model/"
    config: model config, includes model, train, save config, default is "hnlp/config/"
    training_type: should be one of "scratch", "continuous", or "predict"

    References
    ----------------
    - https://radimrehurek.com/gensim/models/fasttext.html
    - https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html
    """

    name: str
    path: str
    config: Dict
    training_type: str

    def __post_init__(self):
        self.name = self.name or "word2vec"
        model_name = "/".join(("fasttext", self.name))
        model_path = model_home / model_name
        self.model_path = self.path or str(model_path)
        self.config = self.config or model_config.fasttext_word2vec
        self.config = ADict(self.config)
        self.training_type = self.training_type or "scratch"
        if self.training_type == "predict":
            self.load_model()

    def load_model(self):
        model = fasttext.load_facebook_model(self.model_path)
        self.model = model.wv

    def fit(self, iter_corpus: Iterable):
        if self.training_type == "scratch":
            self.model = FastText(**self.config.model)
            self.model.build_vocab(corpus_iterable=iter_corpus)
        elif self.training_type == "continuous":
            self.model = fasttext.load_facebook_model(self.model_path)
            self.model.build_vocab(corpus_iterable=iter_corpus, update=True)
        else:
            info = "hnlp: invalid training type: {}".format(self.training_type)
            raise ValueError(info)
        self.model.train(
            corpus_iterable=iter_corpus, total_examples=len(iter_corpus), **self.config.train
        )
        fasttext.save_facebook_model(self.model, self.model_path, **self.config.save)
