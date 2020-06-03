import pytest


from hnlp.dataprocessor.tokenizer import Tokenizer
from hnlp.dataprocessor.tokenizer import ChineseCharTokenizer
from hnlp.dataprocessor.tokenizer import BertChineseWordTokenizer


def test_tokenizer_single_text():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    text = "我爱你。"
    assert tokenizer.node.tokenize(text) == ["我", "爱", "你", "。"]
    assert tokenizer(text) == [101, 2769, 4263, 872, 511, 102]


def test_tokenizer_single_text_tuple_without_label():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    text = ("我爱你。", )
    assert tokenizer.node.tokenize(text[0]) == ["我", "爱", "你", "。"]
    assert tokenizer(text) == ([101, 2769, 4263, 872, 511, 102], )


def test_tokenizer_single_text_with_label():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    text = ("我爱你。", 1)
    assert tokenizer.node.tokenize(text[0]) == ["我", "爱", "你", "。"]
    assert tokenizer(text) == ([101, 2769, 4263, 872, 511, 102], 1)


def test_tokenizer_multiple_texts():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    texts = ["我爱你。", "我爱你。"]
    assert tokenizer(texts) == [[101, 2769, 4263, 872, 511, 102], [
        101, 2769, 4263, 872, 511, 102]]


def test_tokenizer_multiple_texts_tuple_without_labels():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    texts = [("我爱你。", ), ("我爱你。", )]
    assert tokenizer(texts) == [
        ([101, 2769, 4263, 872, 511, 102], ),
        ([101, 2769, 4263, 872, 511, 102], )]


def test_tokenizer_multiple_texts_with_labels():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    texts = [("我爱你。", 2), ("我爱你。", 2)]
    assert tokenizer(texts) == [
        ([101, 2769, 4263, 872, 511, 102], 2),
        ([101, 2769, 4263, 872, 511, 102], 2)]


def test_tokenizer_single_element():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    texts = ["我爱你。"]
    assert tokenizer(texts) == [[101, 2769, 4263, 872, 511, 102]]


def test_tokenizer_single_element_tuple_without_label():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    texts = [("我爱你。", )]
    assert tokenizer(texts) == [([101, 2769, 4263, 872, 511, 102], )]


def test_tokenizer_single_element_with_label():
    vocab_file = "tests/dataprocessor/vocab.txt"
    tokenizer = Tokenizer("bert", vocab_file)
    texts = [("我爱你。", "1")]
    assert tokenizer(texts) == [([101, 2769, 4263, 872, 511, 102], "1")]


def test_chinese_char_tokenizer_normal():
    tokenizer = ChineseCharTokenizer()
    text = "我爱你。"
    assert list(tokenizer.tokenize(text)) == ["我", "爱", "你", "。"]


def test_chinese_char_tokenizer_eng_num():
    tokenizer = ChineseCharTokenizer(False)
    text = "我  abc 123"
    assert list(tokenizer.tokenize(text)) == ["我", " ", " ", "abc", " ", "123"]


def test_chinese_char_tokenizer_pure_eng_num():
    tokenizer = ChineseCharTokenizer(False)
    text = "love abcdef 123"
    assert list(tokenizer.tokenize(text)) == [
        "love", " ", "abcdef", " ", "123"]


def test_chinese_char_tokenizer_special():
    tokenizer = ChineseCharTokenizer()
    text = "我&你_love-.md"
    assert list(tokenizer.tokenize(text)) == [
        "我", "&", "你", "_", "love", "-", ".", "md"]


def test_chinese_char_tokenizer_remove_blank():
    tokenizer = ChineseCharTokenizer()
    text = "我  爱 你"
    assert list(tokenizer.tokenize(text)) == ["我", "爱", "你"]


def test_bert_chinese_word_tokenizer_normal():
    vocab_file = "tests/dataprocessor/vocab.txt"
    import jieba
    tk = Tokenizer(
        name="bert_chinese_word",
        vocab_file=vocab_file,
        segmentor=jieba.lcut
    )
    text_list = ["我喜欢你，你也喜欢我。"]
    assert tk.node.vocab_size == 21128

    tk.node.build_vocab(text_list)
    assert tk.node.vocab_size == 21129

    assert tk.node.tokenize(text_list[0]) == [
        "我", "喜欢", "你", "，", "你", "也", "喜欢", "我", "。"]

    ids1 = tk(text_list)
    ids2 = tk(text_list[0])
    ids3 = tk.node.encode(text_list[0])

    assert ids1[0] == ids2
    assert ids1[0] == ids3
