import torch

from hnlp.dataprocessor.corpus import Corpus
from hnlp.dataprocessor.preprocessor import Preprocessor
from hnlp.dataprocessor.tokenizer import Tokenizer
from hnlp.dataprocessor.datamanager import DataManager
from hnlp.pretrained.processor import PretrainedBasicProcessor


def test_pretrained_processor():
    pp = PretrainedBasicProcessor()
    batch_size, seq_len = 2, 4
    batch = [
        [1, 2, 3, 4],
        [0, 1, 2, 3]
    ]

    outputs = pp(batch)

    for output in outputs:
        assert len(output) == 4
        assert output["input_ids"].shape == torch.Size([batch_size, seq_len])
        assert output["attention_mask"].shape == torch.Size([batch_size, seq_len])
        assert output["token_type_ids"].shape == torch.Size([batch_size, seq_len])
        assert output["position_ids"].shape == torch.Size([batch_size, seq_len])


def test_pretrained_processor_input_dataloader():
    pp = PretrainedBasicProcessor()

    batch_size = 10
    seq_len = 32

    data_path = "tests/dataprocessor/corpus_data_without_label.txt"
    vocab_path = "tests/dataprocessor/vocab.txt"
    pipe = (Corpus("custom") >>
            Preprocessor("common") >>
            Tokenizer("bert", vocab_path) >>
            DataManager(batch_size=batch_size))
    data = pipe.run(data_path)

    outputs = pp(data)

    for output in outputs:
        print(output)
        assert len(output) == 4
        assert output["input_ids"].shape == torch.Size([batch_size, seq_len])
        assert output["attention_mask"].shape == torch.Size([batch_size, seq_len])
        assert output["token_type_ids"].shape == torch.Size([batch_size, seq_len])
        assert output["position_ids"].shape == torch.Size([batch_size, seq_len])