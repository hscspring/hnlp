import pandas as pd
from hnlp.dataset.corpus import Corpus


def test_unlabeled_file():
    data_path = "tests/dataset/corpus/unlabeled/file1.txt"
    corpus = Corpus("unlabeled")
    result = []
    for item in corpus(data_path):
        assert type(item) == str
        result.append(item)
    assert len(result) == 10


def test_labeled_file():
    data_path = "tests/dataset/corpus/labeled/file1.json"
    corpus = Corpus("labeled")
    result = []
    for item in corpus(data_path):
        assert len(item) == 2
        result.append(item)
    assert tuple(item)
    assert len(result) == 10


def test_unlabeled_multi_files():
    data_path = "tests/dataset/corpus/unlabeled/"
    corpus = Corpus("unlabeled", "*.txt")
    data = corpus(data_path)
    assert len(data) == 20
    assert len(corpus) == 20


def test_labeled_multiple_files():
    data_path = "tests/dataset/corpus/labeled/mfile.json"
    corpus = Corpus("labeled", keys=("text_a", "text_b", "label"))
    data = corpus(data_path)
    assert len(data) == 10
    assert len(data[0]) == 3
    assert tuple(data[0])
    assert len(corpus) == 10
    assert type(corpus[:2]) == pd.DataFrame
    assert type(corpus[0]) == pd.Series
