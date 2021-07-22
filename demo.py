# from hnlp.pretrained import Pretrained
from hnlp.dataset.tokenizer import Tokenizer
from hnlp.dataset.tokenizer import BasicTokenizer
from hnlp.dataset.tokenizer import BertTokenizer
from hnlp import Corpus
from hnlp.register import Register

print("type of BasicTokenizer: ", type(BasicTokenizer))
print(Register._dict)

path = "tests/dataset/labeled_corpus.txt"
corpus = Corpus("labeled", path)
for item in corpus:
    print(item)

text = "我爱你"
vocab_file = "tests/dataset/vocab.txt"

tr = BertTokenizer(vocab_file, 10, lambda x: list(x))
print("bert tokenizer: ", tr(text))
print("bert encode: ", tr.encode(text))
tb = BasicTokenizer(vocab_file, 10, lambda x: list(x))
print("basic tokenzier: ", tb(text))
tk = Tokenizer("bert", vocab_file, 10, lambda x: list(x))
print("tokenzier: ", tk(text))


def token_maker(corpus):
    for item in corpus:
        tokens = tk(item["text"])
        yield tokens


# w2v_modelpath = "new.300.bin"
# w2v_config = ADict()
# w2v_config.model.vector_size = 300
# w2v_config.model.negative = 10
# w2v_config.train.epochs = 10

# model = Pretrained(
    # name="fasttext",
    # model_path=w2v_modelpath,
    # model_config=w2v_config,
    # training_type="scratch",
# )

# iter_corpus = token_maker(corpus)
# sentences = list(iter_corpus)
# model.fit(sentences)
