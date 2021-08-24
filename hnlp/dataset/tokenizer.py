"""
Tokenizer Module
========================
The core module of Tokenizer. Include BasicTokenizer, BertTokenizer
Tokenizer is the last node in the chain, so we convert the result to numpy

Note: We do not add model or training to our processing chain.
"""

import collections
from dataclasses import dataclass
from typing import List, Callable, Union, Dict, Tuple, Any
from numbers import Number
import numpy as np
import pnlp

from hnlp.node import Node
from hnlp.config import SpeToken
from hnlp.register import Register


@dataclass
class Tokenizer(Node):
    """
    Tokenizer middleware

    Parameters
    -----------
    name: a name for tokenizer, for example "basic", "bert"
    vocab_file: vocab file for the tokenizer
    max_seq_len: max sequence length for tokenizer to encode
    segmentor: how to segment the input text
    """

    name: str
    vocab_file: str = ""
    max_seq_len: int = 512
    segmentor: callable = lambda x: list(x)
    return_numpy: bool = True

    def __post_init__(self):
        super().__init__()
        self.identity = "tokenizer"
        self.node = super().get_cls(self.identity, self.name)(
            self.vocab_file, self.max_seq_len, self.segmentor
        )

    # over ride Node
    def call(self, inp: Union[str, List[str], List[Tuple[str, Any]]]):
        if self.return_numpy:
            # override
            return self.node.call(inp)
        else:
            # use Node call directly
            return super().call(inp)


@Register.register
@dataclass
class BasicTokenizer:

    vocab_file: str
    max_seq_len: int
    segmentor: Callable

    def __post_init__(self):
        vocab = pnlp.read_lines(self.vocab_file)
        unused = [SpeToken.unused.format(i) for i in range(1, 100)]
        others = list(SpeToken.values())[1:-1]
        self.default_tokens = [SpeToken.pad] + unused + others
        self.load_vocab(vocab)

    def load_vocab(self, vocab: List[str]):
        if self.check_vocab_contains_special(vocab):
            self.vocab = vocab
        else:
            self.vocab = self.default_tokens + vocab
        self.vocab_size = len(self.vocab)
        self.word2id = self.get_word2id(self.vocab)
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

    def get_word2id(self, vocab: List[str]) -> Dict[str, int]:
        res = {}
        i = 0
        for w in vocab:
            if w not in res:
                res[w] = i
                i += 1
            else:
                continue
        return res

    def check_vocab_contains_special(self, vocab: List[str]) -> bool:
        return all(
            (SpeToken.pad in vocab, SpeToken.unk in vocab, SpeToken.mask in vocab)
        )

    def tokenize(self, text: str) -> List[str]:
        res = []
        for token in self.segmentor(text):
            res.append(token)
        return res

    def _encode(self, text: str) -> List[int]:
        ids = []
        unk_id = self.word2id.get(SpeToken.unk)
        for token in self.tokenize(text):
            id = self.word2id.get(token, unk_id)
            ids.append(id)
        return ids

    def encode(self, inputs: Union[str, List[str]]) -> List[List[int]]:
        if isinstance(inputs, str):
            return self._encode(inputs)
            # inputs = [inputs]
        res = []
        for inp in inputs:
            ids = self._encode(inp)
            res.append(ids)
        return res

    def padding(self, ids: List[List[int]]) -> List[List[int]]:
        pad_id = self.word2id.get(SpeToken.pad)
        # arr = np.zeros((len(ids), self.max_seq_len), dtype=np.int32)
        res = []
        for i, sen_ids in enumerate(ids):
            length = len(sen_ids)
            if length < self.max_seq_len:
                pad = [pad_id] * (self.max_seq_len - length)
                sen_ids.extend(pad)
            else:
                sen_ids = sen_ids[: self.max_seq_len]
            res.append(sen_ids)
            # arr[i] = sen_ids
        return res

    def decode(self, ids: List[int]) -> str:
        """
        Decode ids to tokens.
        """
        res = []
        for i in ids:
            word = self.id2word.get(i)
            res.append(word)
        return "".join(res)

    def build_vocab(self, text_list: list, vocab_file: str, min_freq: int = 2):
        """
        Vocab builder

        Parameters
        ------------
        text_list: A list of text
        vocab_file: Where the vocab_file should locate
        min_freq: Minimal frequence for a token,
            any token whose frequence < min_freq will be dropped
        """
        vocab = self.default_tokens
        words = []
        for text in text_list:
            for token in self.tokenize(text):
                words.append(token)
        count = collections.Counter(words).most_common()
        for _i, (token, freq) in enumerate(count):
            if freq >= min_freq:
                vocab.append(token)
        pnlp.write_file(vocab_file, vocab)
        self.load_vocab(vocab)

    def __call__(self, inp: Union[str, List[str]]):
        return self.encode(inp)

    def call(
        self, inp: Union[str, List[str], List[Tuple[str, Any]]]
    ) -> np.array:
        if isinstance(inp, list) and isinstance(inp[0], tuple):
            res = []
            for i in range(len(inp[0])):
                inputs = [v[i] for v in inp]
                if isinstance(inputs[0], str):
                    arr = self.call(inputs)
                elif isinstance(inputs[0], Number):
                    arr = np.array(inputs, dtype=np.float32)
                else:
                    info = "hnlp: invalid type of iterable item"
                    raise ValueError(info)
                res.append(arr)
            return res
        else:
            # Here we make signle text in a list, unlike the __call__
            if isinstance(inp, str):
                inp = [inp]
            ids = self.encode(inp)
            padded = self.padding(ids)
            return np.array(padded, dtype=np.int32)


@Register.register
@dataclass
class BertTokenizer(BasicTokenizer):
    def _encode(self, text: str) -> List[int]:
        cls_id = self.word2id.get(SpeToken.cls)
        sep_id = self.word2id.get(SpeToken.sep)
        ids = super()._encode(text)
        return [cls_id] + ids + [sep_id]
