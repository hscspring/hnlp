import pnlp
import numpy as np

from hnlp.dataset.tokenizer import Tokenizer
from hnlp.dataset.tokenizer import BasicTokenizer

vocab_file = "tests/dataset/vocab.txt"


def test_tokenizer_single_text():
    tokenizer = Tokenizer("bert", vocab_file, 10, return_numpy=False)
    text = "我爱你。"
    assert tokenizer.node.tokenize(text) == ["我", "爱", "你", "。"]
    assert tokenizer(text) == [101, 2768, 4262, 871, 510, 102]


def test_tokenizer_tuple():
    tokenizer = Tokenizer("bert", vocab_file, 10, return_numpy=False)
    inp = ("我爱你。", "你爱我。", 1)
    res = tokenizer(inp)
    print(res)
    assert res == (
        [101, 2768, 4262, 871, 510, 102], [101, 871, 4262, 2768, 510, 102],
        1,
    )


def test_tokenizer_list_str():
    tokenizer = Tokenizer("bert", vocab_file, 10, return_numpy=False)
    inp = ["我爱你。", "你爱我。"]
    assert tokenizer(inp) == [
        [101, 2768, 4262, 871, 510, 102],
        [101, 871, 4262, 2768, 510, 102],
    ]


def test_tokenizer_single_text_return_nump():
    tokenizer = Tokenizer("bert", vocab_file, 10)
    text = "我爱你。"
    assert tokenizer.node.tokenize(text) == ["我", "爱", "你", "。"]
    assert tokenizer(text).tolist() == [[101, 2768, 4262, 871, 510, 102, 0, 0, 0, 0]]


def test_tokenizer_single_element():
    tokenizer = Tokenizer("bert", vocab_file, 6)
    texts = ["我爱你。"]
    assert tokenizer(texts).tolist() == [[101, 2768, 4262, 871, 510, 102]]


def test_tokenizer_multiple_texts():
    tokenizer = Tokenizer("bert", vocab_file, 6)
    texts = ["我爱你。", "我爱你。"]
    assert tokenizer(texts).tolist() == [
        [101, 2768, 4262, 871, 510, 102],
        [101, 2768, 4262, 871, 510, 102],
    ]


def test_tokenizer_multiple_inputs():
    tokenizer = Tokenizer("bert", vocab_file, 6)
    texts = [("我爱你", "你爱我", "哈哈哈")]
    res = tokenizer(texts)
    assert len(res) == 3
    assert isinstance(res[0], np.ndarray)
    assert res[0].tolist() == [[101, 2768, 4262, 871, 102, 0]]


def test_tokenizer_multiple_inputs_with_label():
    tokenizer = Tokenizer("bert", vocab_file, 6)
    texts = [("我爱你", "你爱我", "哈哈哈", 1), ("我爱你", "你爱我", "哈哈哈", 0)]
    res = tokenizer(texts)
    assert len(res) == 4
    assert isinstance(res[3], np.ndarray)
    assert res[0].tolist() == [
        [101, 2768, 4262, 871, 102, 0],
        [101, 2768, 4262, 871, 102, 0],
    ]
    assert res[-1].tolist() == [1, 0]


def test_basic_tokenizer():
    tokenizer = BasicTokenizer(vocab_file, 6, lambda x: list(x))
    text = "我爱你。"
    assert tokenizer.tokenize(text) == ["我", "爱", "你", "。"]


def test_basic_tokenizer_with_custom_segmentor():
    tokenizer = BasicTokenizer(vocab_file, 6, pnlp.cut_zhchar)
    text = "我&你_love-.md"
    assert list(tokenizer.tokenize(text)) == [
        "我",
        "&",
        "你",
        "_",
        "love",
        "-",
        ".",
        "md",
    ]


def test_tokenizer_with_custom_segmentor():
    import jieba

    tk = Tokenizer(
        name="bert", vocab_file=vocab_file, max_seq_len=10, segmentor=jieba.lcut
    )
    text_list = ["我喜欢你，你也喜欢我。"]
    assert tk.node.vocab_size == 21127

    new_vocab_file = "tests/dataset/new_vocab.txt"
    tk.node.build_vocab(text_list, new_vocab_file, 2)
    assert tk.node.vocab_size == 109

    assert tk.node.tokenize(text_list[0]) == [
        "我",
        "喜欢",
        "你",
        "，",
        "你",
        "也",
        "喜欢",
        "我",
        "。",
    ]

    ids1 = tk(text_list)
    ids2 = tk(text_list[0])
    ids3 = tk.node.encode(text_list[0])

    assert ids1.tolist() == ids2.tolist()
    assert ids1.tolist() != ids3
