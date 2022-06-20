"""
Tokenizer Module
========================
The core module of Tokenizer. Include BasicTokenizer, BertTokenizer
Tokenizer is the last node in the chain, so we convert the result to numpy

Note: We do not add model or training to our processing chain.
"""

import collections
from typing import List, Callable, Union, Dict, Tuple, Any
from pathlib import Path
import numpy as np
import pnlp

from transformers import AutoTokenizer

from hnlp.node import Node
from hnlp.config import SpeToken, default_config
from hnlp.register import Register


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

    def __init__(
            self,
            name: str,
            vocab_file: Union[str, Path] = "",
            max_seq_len: int = 0,
            segmentor: callable = lambda x: list(x),
            return_numpy: bool = False,
    ):

        super().__init__()
        self.return_numpy = return_numpy
        self.identity = "tokenizer"

        if not vocab_file:
            vocab_file = default_config.vocab_file

        self.node = super().get_cls(
            self.identity,
            name)(
            vocab_file,
            max_seq_len,
            segmentor)

    # over ride Node's call function
    def call(
        self,
        inp: Union[str, List[str], List[Tuple[str, Any]]],
        *args
    ):
        if self.return_numpy:
            # override
            return self.node.call(inp)
        else:
            # use Node call directly, that is just Node's __call__
            return super().call(inp)


class TokenizerMixin:

    ...


@Register.register
class BasicTokenizer(TokenizerMixin):

    def __init__(self, vocab_file: str, max_seq_len: int, segmentor: Callable):
        self.vocab_file = vocab_file
        self.max_seq_len = max_seq_len
        self.segmentor = segmentor
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
        return all((SpeToken.pad in vocab, SpeToken.unk in vocab, SpeToken.mask
                    in vocab))

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
        res = self.encode(inp)
        return res

    def call(
        self,
        inp: Union[str, List[str], List[Tuple[str, Any]]]
    ) -> np.array:
        if isinstance(inp, list) and isinstance(inp[0], tuple):
            res = []
            for i in range(len(inp[0])):
                inputs = [v[i] for v in inp]
                if isinstance(inputs[0], str):
                    arr = self.call(inputs)
                # should be labels
                elif isinstance(inputs[0], int):
                    arr = np.array(inputs, dtype=np.int16)
                else:
                    arr = inputs
                res.append(arr)
            return res
        else:
            # Here we make single text in a list, unlike the __call__
            if isinstance(inp, str):
                inp = [inp]
            ids = self.encode(inp)
            padded = self.padding(ids)
            return np.array(padded, dtype=np.int32)

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
                sen_ids = sen_ids[:self.max_seq_len]
            res.append(sen_ids)
            # arr[i] = sen_ids
        return res


@Register.register
class BertTokenizer(BasicTokenizer):

    def __init__(self, vocab_file: str, max_seq_len: int, segmentor: Callable = None):
        super().__init__(vocab_file, max_seq_len, segmentor)
        self.max_seq_len = max_seq_len
        vocab_path = Path(vocab_file).parent.as_posix()
        self.bert = AutoTokenizer.from_pretrained(vocab_path)

    def tokenize(self, text: str) -> List[str]:
        return self.bert.tokenize(text)

    def _encode(self, text: str) -> List[int]:
        if self.max_seq_len > 0:
            return self.bert.encode(
                text, padding="max_length", max_length=self.max_seq_len, truncation=True
            )
        else:
            return self.bert.encode(text)
