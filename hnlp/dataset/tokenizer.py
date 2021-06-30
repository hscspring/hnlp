import collections
from dataclasses import dataclass
import re

from pnlp import cut_zhchar

from transformers import BasicTokenizer
from transformers import BertTokenizer as TransBertTokenizer
from transformers import RobertaTokenizer as TransRobertaTokenizer

from hnlp.node import Node
from hnlp.register import Register

# referenced from jieba
re_zh = re.compile(r"([\u4E00-\u9FD5+#&._%-]+)", re.UNICODE)
re_skip = re.compile(r"(\s)", re.U)


@dataclass
class Tokenizer(Node):

    name: str
    vocab_file: str = ""
    merges_file: str = ""
    segmentor: callable = lambda x: x

    def __post_init__(self):
        super().__init__()
        self.identity = "tokenizer"
        cls_name = "_".join([self.name, self.identity])
        DataTokenizer = Register.get(cls_name)
        if not DataTokenizer:
            raise NotImplementedError

        if self.name == "chinese_word":
            self.node = DataTokenizer(self.segmentor)
        elif self.name == "bert":
            self.node = DataTokenizer(self.vocab_file)
        elif self.name == "roberta":
            self.node = DataTokenizer(self.vocab_file, self.merges_file)
        elif self.name == "bert_chinese_word":
            self.node = BertChineseWordTokenizer(
                self.vocab_file, self.segmentor)
        else:
            self.node = DataTokenizer()


@Register.register
@dataclass
class BertTokenizer(TransBertTokenizer):

    vocab_file: str

    def __post_init__(self):
        super().__init__(
            vocab_file=self.vocab_file
        )

    def __call__(self, text: str):
        return self.encode(text)


@Register.register
@dataclass
class RobertaTokenizer(TransRobertaTokenizer):

    vocab_file: str
    merges_file: str

    def __post_init__(self):
        super().__init__(
            vocab_file=self.vocab_file,
            merges_file=self.merges_file
        )

    def __call__(self, text: str):
        return self.encode(text)


@Register.register
@dataclass
class BertChineseWordTokenizer(TransBertTokenizer):

    vocab_file: str
    segmentor: callable

    def __post_init__(self):
        super().__init__(
            vocab_file=self.vocab_file
        )
        self.do_basic_tokenize = False
        self.vocab_has_built = False

    def _tokenize(self, text: str):
        split_tokens = []
        for token in self.segmentor(text):
            split_tokens.append(token)
        return split_tokens

    def build_vocab(self, text_list: list, min_freq: int = 2):
        words = []
        for text in text_list:
            for w in self._tokenize(text):
                words.append(w)
        count = collections.Counter(words).most_common()
        size = self.vocab_size
        for i, (w, freq) in enumerate(count):
            if len(w) > 1 and freq >= min_freq:
                self.vocab[w] = size - 1 + i
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.vocab_has_built = True

    def __call__(self, text: str):
        if self.vocab_has_built:
            return self.encode(text)
        else:
            raise ValueError("hnlp: Please build vocab first.")


@Register.register
@dataclass
class ChineseCharTokenizer:

    remove_blank: bool = True

    def __post_init__(self):
        self.vocab = collections.OrderedDict()

    def tokenize(self, text: str):
        chars = cut_zhchar(text, self.remove_blank)
        return chars

    def build_vocab(self, text_list: list, min_freq: int = 2):
        pass

    def save_vocab(self, vocab_path: str):
        pass

    def encode(self, text: str):
        pass

    def decode(self, text: str):
        pass

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, text: str):
        pass


@Register.register
@dataclass
class ChineseWordTokenizer(BasicTokenizer):

    segmentor: callable

    def __post_init__(self):
        self.vocab = collections.OrderedDict()

    def tokenize(self, text: str):
        for token in segmentor(text):
            yield token

    def build_vocab(self, data, min_freq: int = 2):
        pass

    def save_vocab(self, vocab_path: str):
        pass

    def encode(self, text: str):
        pass

    def decode(self, text: str):
        pass

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, text: str):
        pass
