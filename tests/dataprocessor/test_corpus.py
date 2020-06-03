from hnlp.dataprocessor.corpus import Corpus


def test_corpus_single_file_without_label():
    data_path = "tests/dataprocessor/corpus_data_without_label.txt"
    corpus = Corpus("custom")
    result = []
    for item in corpus(data_path):
        assert type(item) == str
        assert len(item) > 2
        result.append(item)

    assert len(result) == 10


def test_corpus_single_file():
    data_path = "tests/dataprocessor/corpus_data.txt"
    corpus = Corpus("custom")
    result = []
    for item in corpus(data_path):
        assert type(item) == tuple
        assert len(item) == 2
        result.append(item)

    assert len(result) == 10


def test_corpus_multiple_files_single_element():
    data_path = "tests/dataprocessor/corpus_data.txt"
    corpus = Corpus("custom")
    result = []
    for sub_corpus in corpus([data_path]):
        assert len(sub_corpus) == 10
        result.append(sub_corpus)
        for item in sub_corpus:
            assert len(item) == 2

    assert len(result) == 1
    assert len(result[0]) == 10


def test_corpus_multiple_files():
    data_path1 = "tests/dataprocessor/corpus_data.txt"
    data_path2 = "tests/dataprocessor/corpus_data.txt"
    corpus = Corpus("custom")
    result = []
    for sub_corpus in corpus([data_path1, data_path2]):
        assert len(sub_corpus) == 10
        result.append(sub_corpus)
        for item in sub_corpus:
            assert len(item) == 2

    assert len(result) == 2
    assert len(result[0]) == 10
    assert len(result[0][0]) == 2
