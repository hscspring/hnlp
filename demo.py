from hnlp.pretrained import Pretrained
from hnlp.dataset.tokenizer import SpaceTokenizer
from addict import Dict as ADict
from hnlp import Corpus

path = "/media/sf_kubuntu/lab/corpus/dialogTo20210601/dialog.json"
corpus = Corpus("custom", path)

tk = SpaceTokenizer()


def token_maker(corpus):
    for item in corpus:
        tokens = tk(item["text"])
        yield tokens


w2v_modelpath = "new.300.bin"
w2v_config = ADict()
w2v_config.model.vector_size = 300
w2v_config.model.negative = 10
w2v_config.train.epochs = 10

model = Pretrained(
    name="fasttext",
    model_path=w2v_modelpath,
    model_config=w2v_config,
    training_type="scratch",
)

iter_corpus = token_maker(corpus)
sentences = list(iter_corpus)
model.fit(sentences)
