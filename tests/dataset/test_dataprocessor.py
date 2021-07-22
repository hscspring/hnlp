from dataclasses import dataclass

from hnlp.dataset.corpus import Corpus
from hnlp.dataset.preprocessor import Preprocessor
from hnlp.dataset.tokenizer import Tokenizer
from hnlp.dataset.datamanager_pt import DataManagerPt
from hnlp.node import Node, N

vocab_path = "tests/dataset/vocab.txt"


def test_normal_without_label():
    data_path = "tests/dataset/corpus/unlabeled/file1.txt"
    corpus = Corpus("unlabeled")
    preprocessor = Preprocessor("clean")
    tokenizer = Tokenizer("bert", vocab_path, return_numpy=False)
    datamanager = DataManagerPt()

    data = datamanager(tokenizer(preprocessor(corpus(data_path))))
    assert len(data) == 10
    i = 0
    for batch in data:
        assert type(batch) == list
        # batch size default is 1
        assert len(batch) == 1
        # this is batch data
        assert type(batch[0]) == list
        # tokens
        assert len(batch[0]) > 1
        i += 1
    assert i == 10


def test_normal_with_label():
    data_path = "tests/dataset/corpus/labeled/file1.json"
    corpus = Corpus("labeled")
    preprocessor = Preprocessor("clean")
    tokenizer = Tokenizer("bert", vocab_path, return_numpy=False)
    datamanager = DataManagerPt(batch_size=2)

    data = datamanager(tokenizer(preprocessor(corpus(data_path))))
    assert len(data) == 5
    i = 0
    for batch in data:
        assert type(batch) == tuple
        batch_x, batch_y = batch
        # batch size is 2
        assert len(batch_x) == 2
        assert len(batch_y) == 2
        # this is batch data
        assert type(batch_x[0]) == list
        # tokens
        assert len(batch_x[1]) > 1
        assert batch_y[0] in [1, 0]
        i += 1
    assert i == 5


def test_pipeline_without_label():
    data_path = "tests/dataset/corpus/unlabeled/file1.txt"
    pipe = (
        Corpus("unlabeled")
        >> Preprocessor("clean")
        >> Tokenizer("bert", vocab_path, return_numpy=False)
        >> DataManagerPt(batch_size=2)
    )
    data = pipe.run(data_path)
    assert len(data) == 5
    i = 0
    for batch in data:
        assert type(batch) == list
        # batch size default is 1
        assert len(batch) == 2
        # this is batch data
        assert type(batch[0]) == list
        # tokens
        assert len(batch[0]) > 1
        i += 1
    assert i == 5


def test_pipeline_with_label():
    data_path = "tests/dataset/corpus/labeled/file1.json"
    pipe = (
        Corpus("labeled")
        >> Preprocessor("clean")
        >> Tokenizer("bert", vocab_path, return_numpy=False)
        >> DataManagerPt(batch_size=3, drop_last=True)
    )
    data = pipe.run(data_path)
    assert len(data) == 3
    i = 0
    for batch in data:
        assert type(batch) == tuple
        batch_x, batch_y = batch
        # batch size is 2
        assert len(batch_x) == 3
        assert len(batch_y) == 3
        # this is batch data
        assert type(batch_x[0]) == list
        # tokens
        assert len(batch_x[1]) > 1
        assert batch_y[0] in [1, 0]
        i += 1
    assert i == 3


def test_functional_pipeline_without_label():
    data_path = "tests/dataset/corpus/unlabeled/file1.txt"
    corpus = Corpus("unlabeled")
    preprocessor = Preprocessor("clean")
    tokenizer = Tokenizer("bert", vocab_path, return_numpy=False)
    datamanager = DataManagerPt(batch_size=2)
    pipe = N(corpus) >> N(preprocessor) >> N(tokenizer) >> N(datamanager)
    data = pipe(data_path)
    assert len(data) == 5
    i = 0
    for batch in data:
        assert type(batch) == list
        # batch size default is 1
        assert len(batch) == 2
        # this is batch data
        assert type(batch[0]) == list
        # tokens
        assert len(batch[0]) > 1
        i += 1
    assert i == 5


def test_functional_pipeline_with_label():
    data_path = "tests/dataset/corpus/labeled/file1.json"
    corpus = Corpus("labeled")
    preprocessor = Preprocessor("clean")
    tokenizer = Tokenizer("bert", vocab_path, return_numpy=False)
    datamanager = DataManagerPt(batch_size=3, drop_last=True)
    pipe = N(corpus) >> N(preprocessor) >> N(tokenizer) >> N(datamanager)
    data = pipe(data_path)
    assert len(data) == 3
    i = 0
    for batch in data:
        assert type(batch) == tuple
        batch_x, batch_y = batch
        # batch size is 2
        assert len(batch_x) == 3
        assert len(batch_y) == 3
        # this is batch data
        assert type(batch_x[0]) == list
        # tokens
        assert len(batch_x[1]) > 1
        assert batch_y[0] in [1, 0]
        i += 1
    assert i == 3


def test_custom_node_without_label():
    data_path = "tests/dataset/corpus/unlabeled/file1.txt"

    @dataclass
    class CustomPreprocessor(Node):
        def __post_init__(self):
            super().__init__()
            self.node = lambda x: x

    pipe = (
        Corpus("unlabeled")
        >> CustomPreprocessor()
        >> Tokenizer("bert", vocab_path, return_numpy=False)
        >> DataManagerPt(batch_size=32)
    )
    data = pipe.run(data_path)
    assert len(data) == 1
    i = 0
    for batch in data:
        assert type(batch) == list
        # batch size default is 1
        assert len(batch) == 10
        # this is batch data
        assert type(batch[0]) == list
        # tokens
        assert len(batch[0]) > 1
        i += 1
    assert i == 1


def test_custom_node_with_label():
    data_path = "tests/dataset/corpus/labeled/file1.json"

    @dataclass
    class CustomPreprocessor(Node):
        def __post_init__(self):
            super().__init__()
            self.node = lambda x: x

    pipe = (
        Corpus("labeled")
        >> CustomPreprocessor()
        >> Tokenizer("bert", vocab_path, return_numpy=False)
        >> DataManagerPt(batch_size=5)
    )
    data = pipe.run(data_path)
    assert len(data) == 2
    i = 0
    for batch in data:
        assert type(batch) == tuple
        batch_x, batch_y = batch
        # batch size is 5
        assert len(batch_x) == 5
        assert len(batch_y) == 5
        # this is batch data
        assert type(batch_x[0]) == list
        # tokens
        assert len(batch_x[1]) > 1
        assert batch_y[0] in [1, 0]
        i += 1
    assert i == 2
